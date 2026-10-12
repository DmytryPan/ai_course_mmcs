"""Асинхронный доступ к базе мест Ростова (poi.sqlite).

Слой «репозиторий»: только запросы к данным, без знания об агенте, LLM и вебе.
Функции возвращают обычные dict/list — как их показать модели, решает agent/tools.py.

Почему aiosqlite, а не sqlite3 из sdk_tasks.py: сервис асинхронный (FastAPI + ADK),
и синхронный запрос к базе остановил бы цикл событий — на время запроса «замёрзли» бы
все пользователи. aiosqlite выполняет запросы в отдельном потоке и отдаёт await.

База открывается только на чтение (mode=ro): сервис её никогда не меняет, и случайная
запись из инструмента невозможна в принципе.

Логика поиска адреса, ближайших мест и расстояний перенесена из week2/practice/sdk_tasks.py
(address_point, nearest, distance_m), но без глобального журнала CALLS: в многопользовательском
сервисе общий на всех список вызовов перемешал бы чужие запросы.
"""

from __future__ import annotations

import math                # hypot, cos, radians — для расстояний
import re                  # регулярные выражения — для разбора названия улицы
from pathlib import Path
from typing import Any     # Any — «значение любого типа» в аннотациях

import aiosqlite           # асинхронная обёртка над стандартным sqlite3

# Сколько метров в одном градусе. По широте — почти константа (~111.3 км).
# По долготе — меньше и зависит от широты: градус долготы «сжимается» к полюсам
# пропорционально cos(широты). Для Ростова (≈47.23° с. ш.) это ~75.6 км.
LAT_M = 111_320                                   # метров в градусе широты
LON_M = LAT_M * math.cos(math.radians(47.23))     # ... и в градусе долготы на широте Ростова

# Какие колонки отдаём. Для списков мест — только нужное для ответа и карты;
# телефон, сайт, удобства — только в полной карточке (details), чтобы не раздувать
# результаты поиска лишними токенами.
LIST_COLUMNS = "id, kind, name, brand, street, housenumber, opening_hours, cuisine, lat, lon"
CARD_COLUMNS = ("id, kind, name, brand, street, housenumber, opening_hours, is_24_7, cuisine, "
                "internet, wheelchair, outdoor_seating, takeaway, phone, website, lat, lon")


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    """Расстояние по прямой в метрах. Для масштабов города плоская формула точна до метров."""
    # Разницу координат в градусах переводим в метры по каждой оси,
    # а дальше — теорема Пифагора: hypot(dx, dy) = √(dx² + dy²).
    return int(math.hypot((lat1 - lat2) * LAT_M, (lon1 - lon2) * LON_M))


def street_stem(street: str) -> str:
    """Основа названия улицы для поиска по LIKE в любом падеже.

    «на Белорусской улице» → «Белорусск», «проспект Ворошиловский» → «Ворошиловск».
    Берём самое длинное русское слово (не «улица/проспект/переулок/бульвар»)
    и отрезаем два последних символа — окончание, которое меняется по падежам.
    Короткие слова (меньше 7 букв, как «Марта» в «8 Марта») не режем: основа «Мар»
    совпала бы с половиной города. Если русских слов нет — строка как есть.
    """
    # re.findall вытаскивает все русские слова длиной от 4 букв: «на», «8» и т.п. отпадают.
    # Из них выкидываем служебные слова (улица, проспект…): они есть в половине адресов.
    words = [w for w in re.findall(r"[А-Яа-яЁё]{4,}", street)
             if not re.match(r"улиц|проспек|переул|бульвар|площад|шоссе", w.lower())]
    if not words:
        return street.strip()               # нечего резать — ищем по строке как есть
    longest = max(words, key=len)           # самое длинное слово — обычно это и есть название
    return longest[:-2] if len(longest) >= 7 else longest   # [:-2] — без двух последних букв


class PoiRepo:
    """Запросы к poi.sqlite. Один экземпляр на процесс; соединение открывается на каждый запрос.

    Соединение на запрос — осознанное упрощение: открыть локальный файл SQLite стоит
    доли миллисекунды, зато нет общего соединения, за которое конкурировали бы
    одновременные запросы разных пользователей.
    """

    def __init__(self, db_path: Path) -> None:
        if not db_path.exists():
            # Падаем сразу при старте, а не на первом вопросе пользователя.
            raise FileNotFoundError(f"нет базы мест: {db_path}")
        # URI-форма пути с mode=ro — открыть только на чтение. as_uri() даёт file:///C:/...
        self._uri = f"{db_path.resolve().as_uri()}?mode=ro"

    async def _rows(self, sql: str, params: tuple | list = ()) -> list[dict[str, Any]]:
        """Выполнить SELECT и вернуть все строки списком словарей."""
        # async with — соединение закроется само при выходе из блока, даже при ошибке.
        # uri=True — сказать sqlite, что передан URI (с параметром mode=ro), а не путь.
        async with aiosqlite.connect(self._uri, uri=True) as con:
            con.row_factory = aiosqlite.Row          # строки как словари: row["name"]
            # params подставляются вместо знаков ? в sql — безопасно, без склейки строк.
            async with con.execute(sql, params) as cur:
                return [dict(r) for r in await cur.fetchall()]   # Row → обычный dict

    async def _one(self, sql: str, params: tuple | list = ()) -> dict[str, Any] | None:
        """То же, но одна строка: первая из результата или None, если пусто."""
        got = await self._rows(sql, params)
        return got[0] if got else None

    # ── поиск и подсчёт ──────────────────────────────────────────────────────

    # @staticmethod — метод не использует self: это просто функция, сгруппированная с классом.
    @staticmethod
    def _filters(kind: str | None, street: str | None, brand: str | None, cuisine: str | None,
                 name: str | None, only_24_7: bool) -> tuple[str, list[Any]]:
        """Собирает WHERE из заданных фильтров. Значения — только через параметры (?),
        никакой подстановки строк в SQL: так невозможна SQL-инъекция из аргументов модели."""
        # "1 = 1" — всегда истинное условие-заглушка: к нему можно приклеивать
        # " AND ..." сколько угодно раз, не думая, первое это условие или нет.
        where, params = ["1 = 1"], []
        if kind:
            where.append("kind = ?"); params.append(kind)              # точное совпадение вида
        if street:
            # LIKE '%Пушкинск%' — подстрока в любом месте названия улицы
            where.append("street LIKE ?"); params.append(f"%{street_stem(street)}%")
        if brand:
            # COLLATE NOCASE в SQLite работает только для латиницы, поэтому ищем и по brand,
            # и по name: у сетей бренд часто записан только в одном из полей.
            where.append("(brand LIKE ? OR name LIKE ?)"); params += [f"%{brand}%"] * 2
        if cuisine:
            # cuisine бывает списком через «;» (pizza;italian) — подстрока это покрывает
            where.append("cuisine LIKE ?"); params.append(f"%{cuisine}%")
        if name:
            where.append("name LIKE ?"); params.append(f"%{name}%")
        if only_24_7:
            where.append("is_24_7 = 1")       # без параметра: значение фиксированное
        return " AND ".join(where), params    # "1 = 1 AND kind = ? AND street LIKE ?", [...]

    # Звёздочка * в сигнатуре: все аргументы после неё — только по имени
    # (repo.search(kind="cafe"), а не repo.search("cafe")). Так вызовы читаемее и не перепутать порядок.
    async def search(self, *, kind: str | None = None, street: str | None = None,
                     brand: str | None = None, cuisine: str | None = None,
                     name: str | None = None, only_24_7: bool = False,
                     limit: int = 10) -> tuple[int, list[dict[str, Any]]]:
        """Места по фильтрам. Возвращает (сколько всего подошло, первые limit строк).

        Общее число нужно модели, чтобы честно сказать «показаны 10 из 37», а не выдать
        первые десять за полный список. Места с названием идут первыми: безымянная
        точка в ответе пользователю почти бесполезна.
        """
        where, params = self._filters(kind, street, brand, cuisine, name, only_24_7)
        # Два запроса: сначала считаем всё подходящее, потом берём первые limit строк.
        total = (await self._one(f"SELECT COUNT(*) AS n FROM poi WHERE {where}", params))["n"]
        rows = await self._rows(
            # ORDER BY name IS NULL — выражение 0/1: места с названием (0) идут раньше
            # безымянных (1); затем по алфавиту.
            f"SELECT {LIST_COLUMNS} FROM poi WHERE {where} "
            "ORDER BY name IS NULL, name LIMIT ?", [*params, limit])
        return total, rows

    async def count(self, *, kind: str, street: str | None = None,
                    only_24_7: bool = False) -> int:
        """Сколько мест вида kind (с фильтрами по улице и круглосуточности)."""
        where, params = self._filters(kind, street, None, None, None, only_24_7)
        return (await self._one(f"SELECT COUNT(*) AS n FROM poi WHERE {where}", params))["n"]

    async def top_brands(self, *, kind: str, limit: int = 5) -> list[dict[str, Any]]:
        """Крупнейшие сети среди мест вида kind: [{brand, n}] по убыванию n."""
        return await self._rows(
            # COALESCE(brand, 'без бренда') — подставить текст, если brand пустой (NULL),
            # чтобы безбрендовые места собрались в одну группу с понятным названием.
            "SELECT COALESCE(brand, 'без бренда') AS brand, COUNT(*) AS n FROM poi "
            "WHERE kind = ? GROUP BY 1 ORDER BY n DESC LIMIT ?", (kind, limit))

    # ── адреса и расстояния ──────────────────────────────────────────────────

    async def geocode(self, street: str, housenumber: str | None = None) -> dict[str, Any] | None:
        """Координаты адреса по базе мест (другого геокодера у нас нет).

        С номером дома: ищем место с этим адресом; «44» не нашли — пробуем «44А», «44/1».
        Без номера дома: центр улицы — среднее координат мест на ней.

        Под одну основу могут попасть РАЗНЫЕ улицы: «Садов» — это и Большая Садовая
        в центре, и Садовая улица на востоке города. Усреднять их нельзя (получится точка
        посередине, ни к чему не относящаяся), поэтому центр считаем по одной улице —
        той, где больше всего мест, — а остальные возвращаем в other_streets,
        чтобы агент мог уточнить у пользователя.

        Возвращает {street, housenumber, lat, lon, precision, other_streets?} или None.
        """
        like = f"%{street_stem(street)}%"
        # Все улицы, подходящие под основу, с числом мест и центром каждой (GROUP BY street),
        # самая «населённая» — первой. Её и считаем той, что имел в виду пользователь.
        streets = await self._rows(
            "SELECT street, COUNT(*) AS n, AVG(lat) AS lat, AVG(lon) AS lon FROM poi "
            "WHERE street LIKE ? GROUP BY street ORDER BY n DESC", (like,))
        if not streets:
            return None                                   # такой улицы в базе нет совсем
        others = [s["street"] for s in streets[1:]]       # остальные совпавшие улицы
        extra = {"other_streets": others} if others else {}   # ключ добавляем, только если есть что сказать
        if housenumber:
            hn = housenumber.strip()
            # Две попытки по очереди: точный номер ("44" = "44"),
            # потом номер с продолжением ("44%" LIKE: 44А, 44/1, 44к2).
            for sql_hn, op in ((hn, "="), (f"{hn}%", "LIKE")):
                row = await self._one(
                    f"SELECT street, housenumber, lat, lon FROM poi "
                    f"WHERE street LIKE ? AND housenumber {op} ? LIMIT 1", (like, sql_hn))
                if row:
                    # {**row, ...} — новый словарь: все поля row плюс дописанные ключи
                    return {**row, "precision": "дом", **extra}
        # Номер дома не задан или не найден — отдаём центр самой подходящей улицы.
        best = streets[0]
        return {"street": best["street"], "housenumber": None, "lat": best["lat"],
                "lon": best["lon"], "precision": "центр улицы", **extra}

    async def nearest(self, lat: float, lon: float, *, kind: str, limit: int = 3,
                      max_distance_m: int | None = None) -> list[dict[str, Any]]:
        """Ближайшие к точке места вида kind, с полем distance_m, по возрастанию расстояния.

        Расстояния считаем в питоне по всем местам вида: их максимум ~900 (пункты выдачи),
        это миллисекунды. Пространственный индекс здесь был бы преждевременной оптимизацией.
        """
        rows = await self._rows(f"SELECT {LIST_COLUMNS} FROM poi WHERE kind = ?", (kind,))
        for r in rows:
            r["distance_m"] = distance_m(lat, lon, r["lat"], r["lon"])   # дописываем расстояние в строку
        if max_distance_m is not None:
            rows = [r for r in rows if r["distance_m"] <= max_distance_m]  # отсечь дальние
        # sorted(..., key=...) — сортировка по расстоянию; [:limit] — первые limit штук
        return sorted(rows, key=lambda r: r["distance_m"])[:limit]

    async def by_ids(self, ids: list[int]) -> list[dict[str, Any]]:
        """Места по списку id (порядок не гарантируется)."""
        if not ids:
            return []
        # Для IN (...) нужно столько знаков ?, сколько id: ",".join("?" * 3) → "?,?,?"
        marks = ",".join("?" * len(ids))
        return await self._rows(f"SELECT {LIST_COLUMNS} FROM poi WHERE id IN ({marks})", ids)

    async def named_places(self, kinds: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """Все места с названием или брендом (для построения поискового индекса и обогащения).

        Безымянные места (~2 тыс. площадок, банкоматов и т.п.) в семантический индекс
        не идут: по смыслу их искать не по чему, а «Детская площадка» ×900 только засорит выдачу.
        kinds — ограничить видами (например, только общепит).
        """
        sql = f"SELECT {LIST_COLUMNS} FROM poi WHERE (name IS NOT NULL OR brand IS NOT NULL)"
        params: list[Any] = []
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params += list(kinds)
        return await self._rows(sql + " ORDER BY id", params)

    async def details(self, place_id: int) -> dict[str, Any] | None:
        """Полная карточка места по id или None, если такого id нет."""
        return await self._one(f"SELECT {CARD_COLUMNS} FROM poi WHERE id = ?", (place_id,))
