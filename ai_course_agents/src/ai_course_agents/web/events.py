"""Разбор событий ADK для интерфейса: краткие сводки шагов и история чата.

Общий код для двух мест:
  • web/stream.py — живой поток SSE во время ответа;
  • web/chats.py  — восстановление истории чата из сохранённых событий сессии
    (после перезагрузки страницы пользователь видит те же ответы и шаги агента).
"""

from __future__ import annotations

from typing import Any

from google.adk.events import Event

# Человеческие названия инструментов для «Хода мысли» в интерфейсе.
TOOL_TITLES = {
    "search_places": "Поиск мест по фильтрам",
    "count_places": "Подсчёт мест",
    "top_brands": "Крупнейшие сети",
    "geocode_address": "Координаты адреса",
    "nearest_places": "Ближайшие места",
    "rank_by_distance": "Сортировка по расстоянию",
    "place_details": "Карточка места",
    "semantic_search": "Поиск по смыслу",
}


def summarize_tool_result(response: dict[str, Any]) -> dict[str, Any]:
    """Результат инструмента → {summary, places, error} для интерфейса.

    summary — одна строка для шага («нашлось 8 мест: Пельмень, Пельмешка…»);
    places — точки для карты (id, название, координаты, адрес, часы);
    error — текст ошибки, если инструмент вернул {"error": ...}.
    Модель при этом получает полный результат — эта сводка только для человека.
    """
    if not isinstance(response, dict):
        return {"summary": str(response)[:200], "places": [], "error": None}
    if "error" in response:
        return {"summary": "ошибка", "places": [], "error": str(response["error"])}
    places = [p for p in response.get("places", []) if "lat" in p and "lon" in p]
    if places:
        names = ", ".join(p.get("name", "?") for p in places[:3])
        total = response.get("found_total")
        count = f"{len(places)} из {total}" if total and total > len(places) else str(len(places))
        summary = f"нашлось {count}: {names}{'…' if len(places) > 3 else ''}"
    elif "count" in response:
        summary = f"{response.get('kind', '')}: {response['count']}"
    elif "lat" in response and "lon" in response:            # geocode_address
        summary = f"{response.get('street', '')}" + (
            f", {response['housenumber']}" if response.get("housenumber") else " (центр улицы)")
        # Точку адреса тоже покажем на карте — пользователю видно, откуда считали расстояния.
        places = [{"id": None, "name": f"📍 {summary}", "lat": response["lat"],
                   "lon": response["lon"], "is_origin": True}]
    elif "brands" in response:
        summary = ", ".join(f"{b['brand']} ({b['n']})" for b in response["brands"][:3])
    elif "places" in response:
        summary = "ничего не нашлось"
    else:                                                     # карточка места
        summary = response.get("name", "готово")
        if "lat" in response:
            places = [response]
    return {"summary": summary, "places": places, "error": None}


def build_history(events: list[Event]) -> list[dict[str, Any]]:
    """События сессии ADK → список сообщений для интерфейса.

    [{role: "user", text}, {role: "assistant", text, steps: [...], places: [...]}, ...]
    Один ответ ассистента = все события одного invocation_id (один Runner.run_async).
    """
    messages: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None       # сообщение ассистента, которое сейчас собираем
    steps_by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        parts = event.content.parts if event.content and event.content.parts else []
        if event.author == "user" and any(p.text for p in parts):
            # Реплика пользователя (результаты инструментов тоже идут от «user», но без text)
            messages.append({"role": "user", "text": " ".join(p.text for p in parts if p.text)})
            current = None
            continue
        if current is None or current["invocation_id"] != event.invocation_id:
            current = {"role": "assistant", "invocation_id": event.invocation_id, "text": "",
                       "thoughts": "", "steps": [], "places": []}
            messages.append(current)
        for p in parts:
            if p.function_call:
                step = {"id": p.function_call.id, "name": p.function_call.name,
                        "title": TOOL_TITLES.get(p.function_call.name, p.function_call.name),
                        "args": dict(p.function_call.args or {})}
                steps_by_id[p.function_call.id] = step
                current["steps"].append(step)
            elif p.function_response:
                info = summarize_tool_result(p.function_response.response)
                step = steps_by_id.get(p.function_response.id)
                if step is not None:
                    step.update(summary=info["summary"], error=info["error"])
                if info["places"]:
                    current["places"] = info["places"]       # на карте — результат последнего шага
            elif p.text and p.thought:
                current["thoughts"] += p.text
            elif p.text:
                current["text"] = p.text                     # итоговый текст ответа (не partial)
    return messages
