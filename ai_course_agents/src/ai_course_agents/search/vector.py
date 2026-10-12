"""Семантический поиск: модель эмбеддингов (fastembed) + векторная БД (Qdrant, embedded)
+ лексическое совпадение по основам слов. Гибрид объединяется методом RRF.

Как это работает:
  1. Индексация (один раз, search/index.py): текст-описание каждого места → вектор
     из 384–768 чисел (эмбеддинг) → точка в коллекции Qdrant вместе с payload
     (id, название, вид, адрес, координаты, сам текст описания).
  2. Поиск (на каждый запрос) — два независимых списка кандидатов:
     • векторный (dense): запрос → вектор той же моделью → Qdrant находит точки
       с наибольшим косинусным сходством, применяя фильтры по виду и георадиусу;
     • лексический: основы слов запроса («пельмени» → «пельмен») ищутся в текстах
       описаний (название + слова кухни из словаря).
     Списки объединяются RRF (Reciprocal Rank Fusion): у каждого места
     счёт = Σ 1 / (60 + место в списке). Кто высоко хотя бы в одном списке — поднимается,
     кто высоко в обоих — оказывается наверху.

Зачем гибрид, а не только векторы (замер scripts/eval_search.py, docs/PLAN.md):
маленькая мультиязычная модель плохо понимает однословные запросы. На «пельмени»
сходство с «Ресторан «Пельмень»» всего 0.45, а короткие описания вроде «Паб «Визирь»»
оказываются «хабами», близкими почти к любому запросу. Лексический список ловит
явные совпадения слов, векторный — смысл («грузинская кухня» → «Гурия», «Эрти»).

Почему Qdrant в embedded-режиме: это обычная папка на диске (data/qdrant), отдельный
сервер не нужен, а фильтры по полям и по георадиусу — те же, что у серверного Qdrant.
Ограничение режима: папку может держать открытой только один процесс — поэтому индекс
строится, когда сервер остановлен.

Почему fastembed: модели в формате ONNX работают на CPU без PyTorch (установка ~100 МБ
вместо 1–2 ГБ) и быстрее аналогов на sentence-transformers.
"""

from __future__ import annotations

import asyncio
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

# Модели скачиваются с Hugging Face. По умолчанию большие файлы идут через протокол Xet,
# и с этой сети (проверено) загрузка .onnx зависает на 0 байт. Обычный HTTP работает,
# поэтому отключаем Xet. setdefault — не перебиваем значение, если его задали снаружи.
# Строка стоит ДО импорта fastembed: huggingface_hub читает переменную при импорте.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from fastembed import TextEmbedding                  # noqa: E402 — импорт после настройки окружения
from qdrant_client import QdrantClient, models       # noqa: E402

from ai_course_agents.poi.repo import distance_m     # noqa: E402

BATCH = 64          # сколько текстов модель кодирует за раз при индексации
DENSE_TOP = 50      # сколько кандидатов берём из векторного поиска для слияния
RRF_K = 60          # константа RRF: сглаживает разницу между 1-м и 2-м местом (стандартное значение)
# Вес лексического списка в RRF. Явное совпадение слова запроса с названием или кухней —
# более сильное свидетельство, чем векторная близость маленькой модели: при равных весах
# «хабы» вроде «Паб «Визирь»» (сходство 0.73 с «пельмени») стояли вровень с «Пельменем».
# Значение подобрано по scripts/eval_search.py.
LEXICAL_WEIGHT = 2.0

# Слова запроса, которые ничего не говорят о самом месте: «где поесть пельменей» → «пельмен».
STOPWORDS = {"где", "что", "как", "хочу", "можно", "есть", "поесть", "покушать", "выпить",
             "попить", "купить", "найти", "найди", "место", "места", "заведение", "кухня",
             "кухни", "рядом", "около", "недорого", "вкусно", "вкусные", "вкусный", "ресторан",
             "кафе", "бар", "город", "ростов", "ростове", "сходить", "любой", "какой", "какие"}


def collection_name(model_name: str) -> str:
    """Своя коллекция на каждую модель: векторы разных моделей несравнимы между собой.
    «sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2» → «poi__paraphrase_multilingual_minilm_l12_v2»."""
    short = model_name.split("/")[-1].lower()
    return "poi__" + re.sub(r"[^a-z0-9]+", "_", short).strip("_")


def query_stems(query: str) -> list[str]:
    """Основы значимых слов запроса для лексического поиска.

    Грубый «стемминг» отрезанием окончания (как street_stem в poi/repo.py):
    «пельмени» → «пельмен», «шаурма» → «шаур», «блины» → «блин», «пиво» → «пив».
    Слова короче 4 букв и служебные слова (STOPWORDS) отбрасываем.
    """
    stems = []
    for w in re.findall(r"[а-яёa-z]+", query.lower()):      # слова кириллицей и латиницей
        if len(w) < 4 or w in STOPWORDS:
            continue
        cut = 2 if len(w) >= 6 else 1                        # длинным словам — 2 буквы, коротким — 1
        stems.append(w[:-cut])
    return stems


class SemanticIndex:
    """Модель эмбеддингов + коллекция Qdrant + тексты для лексики. Один экземпляр на процесс."""

    def __init__(self, model_name: str, qdrant_dir: Path, models_dir: Path,
                 client: QdrantClient | None = None) -> None:
        self.model_name = model_name
        self.collection = collection_name(model_name)
        # Модель скачивается с Hugging Face при первом запуске и кэшируется в models_dir.
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(models_dir))
        # QdrantClient(path=...) — embedded-режим: база в папке, без сервера.
        # client можно передать готовый — так несколько моделей делят одну папку (оценка).
        self.client = client or QdrantClient(path=str(qdrant_dir))
        # Кэш векторов запросов: одинаковые запросы («пельмени») не кодируем повторно.
        # lru_cache на методе экземпляра — через обёртку, чтобы кэш был у каждого объекта свой.
        self._embed_cached = lru_cache(maxsize=1024)(self._embed_query)
        # Тексты описаний в памяти (для лексического поиска) — загружаются при первом поиске.
        self._docs: list[dict[str, Any]] | None = None

    # ── индексация ──────────────────────────────────────────────────────────

    def build(self, places: list[dict[str, Any]], texts: list[str]) -> None:
        """Пересоздать коллекцию: places[i] (строка базы) описывается текстом texts[i]."""
        # Размер вектора узнаём у самой модели: закодируем пустую строку.
        dim = len(next(iter(self.model.embed([""]))))
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)          # индекс строим с нуля
        self.client.create_collection(
            self.collection,
            # COSINE — косинусная близость: важна «направленность» смысла, а не длина вектора.
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        # embed() — генератор векторов (numpy-массивов) в том же порядке, что тексты.
        vectors = self.model.embed(texts, batch_size=BATCH)
        points = (
            models.PointStruct(
                id=p["id"],                                        # id точки = id места в базе
                vector=v.tolist(),
                payload={
                    "id": p["id"], "name": p.get("name"), "kind": p["kind"],
                    "street": p.get("street"), "housenumber": p.get("housenumber"),
                    "opening_hours": p.get("opening_hours"), "cuisine": p.get("cuisine"),
                    "lat": p["lat"], "lon": p["lon"],
                    # location в формате {lat, lon} — по нему работает фильтр geo_radius
                    "location": {"lat": p["lat"], "lon": p["lon"]},
                    "text": t,                                     # описание — для лексики и отладки
                })
            for p, t, v in zip(places, texts, vectors))
        self.client.upload_points(self.collection, points=points, batch_size=256)
        self._docs = None                                          # тексты поменялись — перечитать

    # ── поиск ───────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> tuple[float, ...]:
        # query_embed — кодирование именно запроса (у некоторых моделей он кодируется иначе,
        # чем документ). Возвращаем кортеж: он неизменяемый, его можно хранить в lru_cache.
        return tuple(next(iter(self.model.query_embed(query))).tolist())

    def _load_docs(self) -> list[dict[str, Any]]:
        """Все payload коллекции в память (≈6.8 тыс. записей, единицы МБ) — для лексики."""
        if self._docs is None:
            docs, offset = [], None
            while True:                                            # scroll — постраничный обход коллекции
                batch, offset = self.client.scroll(self.collection, limit=1000, offset=offset,
                                                   with_payload=True, with_vectors=False)
                docs += [{**pt.payload, "_text": pt.payload["text"].lower()} for pt in batch]
                if offset is None:                                 # страниц больше нет
                    break
            self._docs = docs
        return self._docs

    def _dense(self, query: str, kinds: list[str] | None, lat: float | None, lon: float | None,
               radius_m: int | None) -> list[dict[str, Any]]:
        """Векторные кандидаты из Qdrant с фильтрами: [{...payload, score}] по убыванию сходства."""
        must: list[models.Condition] = []                 # все условия фильтра (логическое И)
        if kinds:
            # MatchAny — kind равен любому из перечисленных
            must.append(models.FieldCondition(key="kind", match=models.MatchAny(any=kinds)))
        if lat is not None and lon is not None and radius_m:
            # Только точки в радиусе radius_m метров от (lat, lon)
            must.append(models.FieldCondition(
                key="location",
                geo_radius=models.GeoRadius(center=models.GeoPoint(lat=lat, lon=lon),
                                            radius=float(radius_m))))
        hits = self.client.query_points(
            self.collection, query=list(self._embed_cached(query.strip().lower())),
            query_filter=models.Filter(must=must) if must else None,
            limit=DENSE_TOP, with_payload=True).points
        return [{**h.payload, "score": round(h.score, 3)} for h in hits]

    def _lexical(self, query: str, kinds: list[str] | None, lat: float | None, lon: float | None,
                 radius_m: int | None) -> list[dict[str, Any]]:
        """Лексические кандидаты: места, в описании которых есть основы слов запроса.

        Совпадения взвешиваются по надёжности источника: в фактах (название, кухня из базы) —
        вес 2, только в догадке LLM («предположительно: …») — вес 1. Так подтверждённая
        «Пельмешка» стоит выше «Самовара», про который пельмени лишь предположены, но и
        «Самовар» в выдаче остаётся. Без этого обогащение подняло находки по догадкам выше
        фактов: строгая P@5 упала с 0.92 до 0.82 (scripts/eval_search.py, docs/PLAN.md).
        При равном весе выше то место, где слово запроса есть прямо в названии.
        """
        stems = query_stems(query)
        if not stems:
            return []
        # (?<![а-яёa-z]) — основа должна начинаться с начала слова: «пив» найдёт «пиво»,
        # «пивной», но не «напиток».
        patterns = [re.compile(r"(?<![а-яёa-z])" + re.escape(s)) for s in stems]
        found = []
        for d in self._load_docs():
            if kinds and d["kind"] not in kinds:
                continue
            if lat is not None and lon is not None and radius_m                     and distance_m(lat, lon, d["lat"], d["lon"]) > radius_m:
                continue
            # partition делит текст на «до» и «после» пометки догадки (если её нет — guess пуст)
            facts, _, guess = d["_text"].partition("предположительно:")
            weight = sum(2 if p.search(facts) else 1 if p.search(guess) else 0 for p in patterns)
            if weight:
                in_name = any(p.search((d.get("name") or "").lower()) for p in patterns)
                found.append((weight, in_name, d))
        found.sort(key=lambda x: (-x[0], not x[1]))       # по убыванию веса, затем «в названии»
        return [d for _, _, d in found[:DENSE_TOP]]

    def search_sync(self, query: str, *, kinds: Iterable[str] | None = None,
                    lat: float | None = None, lon: float | None = None,
                    radius_m: int | None = None, limit: int = 10,
                    mode: str = "hybrid") -> list[dict[str, Any]]:
        """Синхронный поиск. mode="hybrid" (по умолчанию) или "dense" (только векторы — для оценки).

        Возвращает [{...payload, score, match}]: score — косинусное сходство из векторного
        поиска (None, если место нашлось только по словам), match — как нашлось:
        «по смыслу», «по словам» или «по словам и смыслу».
        """
        kinds = list(kinds) if kinds else None
        dense = self._dense(query, kinds, lat, lon, radius_m)
        if mode == "dense":
            return [{**d, "match": "по смыслу"} for d in dense[:limit]]
        lexical = self._lexical(query, kinds, lat, lon, radius_m)

        # RRF: складываем вес/(K + ранг) из каждого списка, где место встретилось.
        fused: dict[int, dict[str, Any]] = {}
        for source, items, weight in (("dense", dense, 1.0), ("lexical", lexical, LEXICAL_WEIGHT)):
            for rank, item in enumerate(items, start=1):
                entry = fused.setdefault(item["id"], {"item": item, "rrf": 0.0, "from": set()})
                entry["rrf"] += weight / (RRF_K + rank)
                entry["from"].add(source)
                if source == "dense":
                    entry["item"] = item                  # у векторного кандидата есть score
        ranked = sorted(fused.values(), key=lambda e: -e["rrf"])[:limit]
        labels = {frozenset({"dense"}): "по смыслу", frozenset({"lexical"}): "по словам",
                  frozenset({"dense", "lexical"}): "по словам и смыслу"}
        result = []
        for e in ranked:
            item = {k: v for k, v in e["item"].items() if k != "_text"}   # служебное поле не отдаём
            item.setdefault("score", None)
            item["match"] = labels[frozenset(e["from"])]
            result.append(item)
        return result

    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """То же для асинхронного кода. Кодирование запроса и перебор текстов нагружают CPU,
        поэтому выполняем их в отдельном потоке: цикл событий в это время обслуживает
        других пользователей, а не ждёт процессор."""
        return await asyncio.to_thread(self.search_sync, query, **kwargs)
