"""Оценка семантического поиска: качество и скорость нескольких моделей эмбеддингов.

    uv run python scripts/eval_search.py                 # все модели, описания как в индексе
    uv run python scripts/eval_search.py --no-guesses    # без догадок LLM (базовая линия)
    uv run python scripts/eval_search.py --models <m1> <m2>

Как считаем качество. 12 запросов «как от пользователя» («пельмени», «суши»…), поиск
только по общепиту. Для каждого запроса заранее задано правило, какие места считать
подходящими: по словам в названии (хинкал, пицц…) или по тегу кухни из базы. Правила
не зависят от модели и от догадок LLM — это «линейка», одинаковая для всех вариантов.
  • P@5   — доля подходящих среди первых 5 результатов (главная метрика: что увидит человек);
  • hit@1 — подходящий ли самый первый результат.
Оговорка: правило видит только то, что явно есть в названии или теге. Если поиск нашёл
настоящую пельменную с названием «Сказка», правило её не засчитает — P@5 занижен
одинаково для всех вариантов, сравнивать их это не мешает.

Режимы: dense — только векторный поиск; hybrid — векторы + лексика по основам слов (RRF).

Скорость: время загрузки модели, построения индекса (≈6.8 тыс. описаний) и задержка
одного запроса (кодирование + поиск) без кэша, p50/p95 по 60 запросам.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # запуск без установки пакета

from ai_course_agents.config import get_settings          # noqa: E402
from ai_course_agents.poi.kinds import FOOD_KINDS          # noqa: E402
from ai_course_agents.search.index import corpus           # noqa: E402
from ai_course_agents.search.vector import SemanticIndex   # noqa: E402

MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "minishlab/potion-multilingual-128M",
]

# запрос → (регулярка по названию, теги кухни, виды мест) — что считать подходящим
QUERIES: dict[str, tuple[str, set[str], set[str]]] = {
    "пельмени":                 (r"пельме|варенич", {"russian"}, set()),
    "суши и роллы":             (r"суши|ролл|sushi|самура|япон|токио|сакур|васаби", {"sushi", "japanese"}, set()),
    "пицца":                    (r"пицц|pizza", {"pizza", "italian_pizza"}, set()),
    "шаурма":                   (r"шаурм|шаверм|донер|кебаб|kebab", {"shawarma", "kebab"}, set()),
    "выпить кофе":              (r"кофе|coffee|кофей", {"coffee_shop"}, set()),
    "грузинская кухня хинкали": (r"хинкал|хачапур|грузин|кинза|сакартвело|тбилиси", {"georgian"}, set()),
    "бургеры":                  (r"бургер|burger", {"burger"}, set()),
    "торты и пирожные":         (r"торт|кондитер|пирожн|десерт", {"cake", "кондитерская"}, {"confectionery"}),
    "попить пива":              (r"пив|beer|пенн|паб|pub|хмел|brew", set(), {"pub"}),
    "плов":                     (r"плов|узбек|чайхан|самарканд|ташкент", {"uzbek"}, set()),
    "блины":                    (r"блин", {"pancake", "crepe", "savory_pancakes"}, set()),
    "рыба и морепродукты":      (r"рыб|улов|морепрод|краб|устриц|fish", {"fish", "seafood", "fish_and_chips"}, set()),
}


def relevant(hit: dict, rule: tuple[str, set[str], set[str]]) -> bool:
    """Подходит ли найденное место под правило запроса."""
    pattern, tags, kinds = rule
    name = (hit.get("name") or "").lower()
    place_tags = {t.strip().lower() for t in (hit.get("cuisine") or "").split(";")}
    return bool(re.search(pattern, name)) or bool(tags & place_tags) or hit["kind"] in kinds


def evaluate(index: SemanticIndex, mode: str) -> dict:
    p5, h1, examples = [], [], {}
    for query, rule in QUERIES.items():
        hits = index.search_sync(query, kinds=FOOD_KINDS, limit=5, mode=mode)
        marks = [relevant(h, rule) for h in hits]
        p5.append(sum(marks) / 5)
        h1.append(marks[0] if marks else False)
        examples[query] = [(h["name"], h["score"], ok) for h, ok in zip(hits, marks)]
    # Задержка без кэша: каждый раз сбрасываем кэш векторов запросов
    lat = []
    for _ in range(5):
        for query in QUERIES:
            index._embed_cached.cache_clear()
            t = time.perf_counter()
            index.search_sync(query, kinds=FOOD_KINDS, limit=5, mode=mode)
            lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    return {"P@5": statistics.mean(p5), "hit@1": statistics.mean(h1),
            "p50_ms": lat[len(lat) // 2], "p95_ms": lat[int(len(lat) * 0.95) - 1],
            "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--no-guesses", action="store_true")
    parser.add_argument("--show", action="store_true", help="напечатать топ-5 по каждому запросу")
    args = parser.parse_args()
    settings = get_settings()

    places, texts = asyncio.run(corpus(use_guesses=not args.no_guesses))
    print(f"описаний: {len(texts)}; догадки LLM: {'нет' if args.no_guesses else 'да'}")
    # Отдельная папка для оценки: не трогаем рабочий индекс сервиса
    client = QdrantClient(path=str(settings.data_dir / "qdrant_eval"))
    rows = []
    for model in args.models:
        t0 = time.perf_counter()
        index = SemanticIndex(model, settings.qdrant_dir, settings.models_dir, client=client)
        t1 = time.perf_counter()
        index.build(places, texts)
        t2 = time.perf_counter()
        for mode in ("dense", "hybrid"):          # только векторы против гибрида
            r = evaluate(index, mode)
            rows.append((f"{model.split('/')[-1]} [{mode}]", t1 - t0, t2 - t1, r))
            if args.show:
                print(f"── {model.split('/')[-1]} [{mode}]")
                for q, ex in r["examples"].items():
                    print(f"  {q}: " + " | ".join(f"{'✓' if ok else '·'}{n}" for n, s, ok in ex))

    print(f"\n{'модель [режим]':<53}{'загрузка, с':>12}{'индекс, с':>11}{'P@5':>7}{'hit@1':>7}{'p50, мс':>9}{'p95, мс':>9}")
    for name, load_s, build_s, r in rows:
        print(f"{name:<53}{load_s:>12.1f}{build_s:>11.1f}{r['P@5']:>7.2f}{r['hit@1']:>7.2f}"
              f"{r['p50_ms']:>9.1f}{r['p95_ms']:>9.1f}")


if __name__ == "__main__":
    main()
