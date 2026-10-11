"""Инструменты агента: тонкий слой между моделью и репозиторием poi.repo.

Как ADK превращает функцию в инструмент (FunctionTool):
  • имя функции → имя инструмента;
  • аннотации типов → JSON Schema аргументов (Literal → "enum", Optional → можно null);
  • ВСЯ докстрока (вместе с секцией Args) → описание инструмента. Его и читает модель,
    выбирая инструмент, поэтому докстроки написаны для модели, а не для программиста:
    что делает, когда звать, когда НЕ звать, что вернёт, смысл аргументов (урок задачи 4);
  • параметр tool_context ADK подставляет сам и в схему не включает.

Общие правила всех инструментов (уроки исходного задания и ошибок прошлых лет):
  • инструмент не бросает исключений: любая ошибка возвращается как {"error": "..."},
    модель видит её как наблюдение и может исправиться (декоратор @observed_errors);
  • в ответ кладётся не всё подряд, а компактные записи (до 10–20 мест, без пустых полей):
    каждый токен результата модель перечитывает на всех следующих шагах;
  • найденные места запоминаются в состоянии сессии (state["last_places"]) — на них
    опираются уточнения «а какой из них ближе?», даже когда старая история обрезана.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Optional

from google.adk.tools import ToolContext

from ai_course_agents.config import get_settings
from ai_course_agents.poi.kinds import KIND_LABELS, KindName
from ai_course_agents.poi.repo import PoiRepo, distance_m

log = logging.getLogger(__name__)

MAX_LIMIT = 20          # больше 20 строк модели не нужно: она их всё равно не перечислит
LAST_PLACES_KEY = "last_places"

_repo: PoiRepo | None = None


def get_repo() -> PoiRepo:
    """Репозиторий создаётся лениво, один на процесс. Тесты могут подменить его set_repo()."""
    global _repo
    if _repo is None:
        _repo = PoiRepo(get_settings().poi_db)
    return _repo


def set_repo(repo: PoiRepo) -> None:
    global _repo
    _repo = repo


# ── общие помощники ─────────────────────────────────────────────────────────

def observed_errors(func: Callable[..., Awaitable[dict]]) -> Callable[..., Awaitable[dict]]:
    """Превращает исключение инструмента в наблюдение {"error": ...} вместо падения прогона.

    functools.wraps копирует имя, докстроку и __wrapped__: по ним ADK строит схему,
    поэтому обёрнутая функция выглядит для модели точно так же, как исходная.
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return await func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — нарочно ловим всё: это граница с моделью
            log.exception("инструмент %s упал", func.__name__)
            return {"error": f"{type(exc).__name__}: {exc}. Проверь аргументы и попробуй иначе."}
    return wrapper


def _check_kind(kind: str) -> dict | None:
    """Схема уже ограничивает kind перечнем, но модель может её нарушить — проверяем сами."""
    if kind not in KIND_LABELS:
        return {"error": f"вида «{kind}» нет в базе. Допустимые значения kind: "
                         + ", ".join(KIND_LABELS)}
    return None


def _clamp(limit: int) -> int:
    return max(1, min(int(limit), MAX_LIMIT))


def _place(row: dict[str, Any]) -> dict[str, Any]:
    """Компактная запись места для модели: только непустые поля, координаты до 5 знаков (~1 м)."""
    address = ", ".join(x for x in (row.get("street"), row.get("housenumber")) if x)
    item = {
        "id": row["id"],
        "name": row.get("name") or "без названия",
        "kind": KIND_LABELS.get(row.get("kind", ""), row.get("kind")),
        "address": address or None,
        "hours": row.get("opening_hours"),
        "cuisine": row.get("cuisine"),
        "distance_m": row.get("distance_m"),
        "lat": round(row["lat"], 5),
        "lon": round(row["lon"], 5),
    }
    return {k: v for k, v in item.items() if v is not None}


def _remember(tool_context: ToolContext | None, places: list[dict[str, Any]]) -> None:
    """Кладёт найденные места в состояние сессии. ADK сохранит его вместе с историей."""
    if tool_context is not None and places:
        tool_context.state[LAST_PLACES_KEY] = [
            {"id": p["id"], "name": p["name"], "lat": p["lat"], "lon": p["lon"]} for p in places]


# ── инструменты ─────────────────────────────────────────────────────────────

@observed_errors
async def search_places(kind: KindName, street: Optional[str] = None,
                        brand: Optional[str] = None, cuisine: Optional[str] = None,
                        name: Optional[str] = None, only_24_7: bool = False,
                        limit: int = 10, tool_context: ToolContext = None) -> dict:
    """Список мест Ростова заданного вида с фильтрами по улице, сети, кухне, названию.

    Зови, когда нужен ПЕРЕЧЕНЬ мест по точным признакам: «какие аптеки на Пушкинской»,
    «пункты выдачи Ozon», «круглосуточные заправки», «пиццерии» (cuisine=pizza).
    НЕ зови, чтобы узнать «сколько» (для этого count_places), чтобы найти ближайшее к точке
    (nearest_places) или когда пользователь описывает место смыслом/словами, а не точным
    признаком — «где поесть пельменей», «уютное место с кофе».

    Возвращает {"found_total": сколько всего подошло, "shown": сколько в списке,
    "places": [{id, name, kind, address, hours, cuisine, lat, lon}]}. Если found_total
    больше shown — скажи пользователю, что показана часть.

    Args:
        kind: вид места латиницей из перечня (pharmacy, cafe, restaurant, outpost …).
        street: улица, если вопрос про конкретную улицу: «Пушкинская», «Доватора».
        brand: сеть или бренд: «Ozon», «Магнит», «Пятёрочка».
        cuisine: кухня латиницей, как в OpenStreetMap: pizza, sushi, burger, coffee_shop,
            russian, georgian, japanese, italian. Заполнена только у части заведений.
        name: часть названия места, если пользователь его знает.
        only_24_7: True только если спрашивают про круглосуточные места.
        limit: сколько мест вернуть, 1–20, по умолчанию 10.
    """
    if err := _check_kind(kind):
        return err
    total, rows = await get_repo().search(kind=kind, street=street, brand=brand, cuisine=cuisine,
                                          name=name, only_24_7=only_24_7, limit=_clamp(limit))
    places = [_place(r) for r in rows]
    _remember(tool_context, places)
    return {"found_total": total, "shown": len(places), "places": places}


@observed_errors
async def count_places(kind: KindName, street: Optional[str] = None,
                       only_24_7: bool = False) -> dict:
    """Сколько в Ростове мест заданного вида: одно число, без списка.

    Зови на вопросы «сколько», «какое количество», «много ли». НЕ зови, если нужен
    перечень мест (для этого search_places).

    Возвращает {"kind": вид по-русски, "count": число}.

    Args:
        kind: вид места латиницей из перечня (pharmacy, cafe, outpost …).
        street: считать только на этой улице, например «Пушкинская».
        only_24_7: True — считать только круглосуточные.
    """
    if err := _check_kind(kind):
        return err
    n = await get_repo().count(kind=kind, street=street, only_24_7=only_24_7)
    return {"kind": KIND_LABELS[kind], "count": n}


@observed_errors
async def top_brands(kind: KindName, limit: int = 5) -> dict:
    """Самые крупные сети (бренды) среди мест заданного вида в Ростове.

    Зови на вопросы «какие сети», «чьих аптек больше всего», «какая сеть крупнейшая».
    НЕ зови для подсчёта всех мест вида (count_places) или списка адресов (search_places).

    Возвращает {"brands": [{brand, n}]} по убыванию n; места без бренда собраны
    в строку «без бренда».

    Args:
        kind: вид места латиницей из перечня.
        limit: сколько брендов вернуть, по умолчанию 5.
    """
    if err := _check_kind(kind):
        return err
    return {"brands": await get_repo().top_brands(kind=kind, limit=_clamp(limit))}


@observed_errors
async def geocode_address(street: str, housenumber: Optional[str] = None) -> dict:
    """Координаты адреса в Ростове (широта и долгота) по базе мест города.

    Зови ПЕРВЫМ, когда пользователь назвал адрес или улицу и нужно что-то рядом с ними:
    «я на Белорусской, 44», «рядом с Пушкинской». Полученные lat и lon передавай
    дальше в nearest_places, rank_by_distance или semantic_search — дословно.
    НЕ зови, если координаты уже есть в вопросе или в прошлых результатах.

    Возвращает {"street", "housenumber", "lat", "lon", "precision"}: precision = «дом»
    (точный адрес) или «центр улицы» (номер дома не указан или не найден — тогда
    расстояния приблизительные, скажи об этом). Если есть "other_streets" — под название
    подошли и другие улицы (например, «Садовая» и «Большая Садовая»): скажи пользователю,
    какую взял, и предложи уточнить. Или {"error"}, если улицы нет в базе.

    Args:
        street: название улицы в любом падеже, без номера дома: «Белорусская», «Садовая».
        housenumber: номер дома, если назван: «44», «12А».
    """
    point = await get_repo().geocode(street, housenumber)
    if point is None:
        return {"error": f"улица «{street}» в базе не найдена. Уточни название у пользователя."}
    point["lat"], point["lon"] = round(point["lat"], 6), round(point["lon"], 6)
    return point


@observed_errors
async def nearest_places(lat: float, lon: float, kind: KindName, limit: int = 3,
                         max_distance_m: Optional[int] = None,
                         tool_context: ToolContext = None) -> dict:
    """Ближайшие к точке места заданного вида, с расстоянием в метрах.

    Зови, когда нужно «ближайшее», «рядом», «в шаговой доступности» и известны координаты
    точки: из вопроса или из geocode_address. НЕ зови без координат — сначала
    geocode_address; и не зови для общего перечня мест города (search_places).

    Возвращает {"places": [{id, name, address, hours, distance_m, lat, lon}]}
    по возрастанию distance_m (расстояние по прямой).

    Args:
        lat: широта точки в градусах, например 47.2357.
        lon: долгота точки в градусах, например 39.7015.
        kind: вид места латиницей из перечня.
        limit: сколько ближайших вернуть, 1–20, по умолчанию 3.
        max_distance_m: не дальше этого расстояния в метрах, если пользователь его задал.
    """
    if err := _check_kind(kind):
        return err
    rows = await get_repo().nearest(lat, lon, kind=kind, limit=_clamp(limit),
                                    max_distance_m=max_distance_m)
    places = [_place(r) for r in rows]
    _remember(tool_context, places)
    return {"places": places}


@observed_errors
async def rank_by_distance(lat: float, lon: float, place_ids: list[int],
                           tool_context: ToolContext = None) -> dict:
    """Расстояние от точки до уже найденных мест и их порядок от ближнего к дальнему.

    Зови на уточнения про места из прошлых ответов: «а какой из них ближе к Садовой?»,
    «что из этого рядом со мной?». id мест бери из прошлых результатов инструментов.
    НЕ зови для поиска новых мест — для этого nearest_places.

    Возвращает {"places": [{id, name, address, distance_m, lat, lon}]} по возрастанию
    distance_m; id, которых нет в базе, пропускаются.

    Args:
        lat: широта точки отсчёта.
        lon: долгота точки отсчёта.
        place_ids: id мест из прошлых результатов, например [811, 731].
    """
    rows = await get_repo().by_ids([int(i) for i in place_ids][:MAX_LIMIT])
    for r in rows:
        r["distance_m"] = distance_m(lat, lon, r["lat"], r["lon"])
    places = [_place(r) for r in sorted(rows, key=lambda r: r["distance_m"])]
    _remember(tool_context, places)
    return {"places": places}


@observed_errors
async def place_details(place_id: int) -> dict:
    """Полная карточка одного места: часы работы, телефон, сайт, кухня, удобства.

    Зови, когда пользователь спрашивает подробности о конкретном найденном месте:
    «до скольки работает», «какой телефон», «есть ли wi-fi». id бери из прошлых
    результатов. НЕ зови для поиска мест.

    Возвращает карточку места. Поля 24/7, wifi, wheelchair (доступно для колясок),
    outdoor_seating (веранда), takeaway (с собой): true / false; если поля нет —
    в базе нет данных (это НЕ означает «нет»).

    Args:
        place_id: id места из прошлых результатов, например 1728.
    """
    row = await get_repo().details(int(place_id))
    if row is None:
        return {"error": f"места с id={place_id} нет в базе"}
    card = _place(row)
    for field, key in (("is_24_7", "24/7"), ("internet", "wifi"), ("wheelchair", "wheelchair"),
                       ("outdoor_seating", "outdoor_seating"), ("takeaway", "takeaway")):
        if row.get(field) is not None:
            card[key] = bool(row[field])
    for field in ("brand", "phone", "website"):
        if row.get(field):
            card[field] = row[field]
    return card


# Набор инструментов агента. semantic_search добавится на этапе 2.
TOOLS = [search_places, count_places, top_brands, geocode_address,
         nearest_places, rank_by_distance, place_details]
