"""Колбэки агента: гардрейл и обрезка истории.

Оба — before_model_callback: ADK вызывает их перед КАЖДЫМ запросом к LLM и передаёт
llm_request — то, что сейчас уйдёт модели (инструкция, история, схемы инструментов).
Колбэк может:
  • изменить llm_request (так работает обрезка истории);
  • вернуть готовый LlmResponse — тогда ADK НЕ зовёт модель, а использует этот ответ
    как будто его написала модель (так работает гардрейл: отказ за 0 токенов).
Вернуть None — «ничего не делаю, зови модель как обычно».

Порядок в agent.py: сначала гардрейл (если он ответил — дальше ничего не нужно),
потом обрезка истории.
"""

from __future__ import annotations

import re

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

# Сколько последних реплик пользователя (вместе с ответами и шагами агента между ними)
# модель видит в истории. Старше — отрезаем: каждый токен истории оплачивается
# на каждом шаге каждого следующего ответа. Ссылки на старые места не теряются —
# последние найденные места лежат в state и попадают в инструкцию (agent.py).
MAX_USER_TURNS = 6

# ── гардрейл ────────────────────────────────────────────────────────────────

# Явные посторонние просьбы. Список ловит именно ПРОСЬБЫ (написать стихи, перевести,
# решить), а не темы: «Сколько аптек?» и «а из них круглосуточных?» сюда не попадут.
# Жадное правило «нет слова про места → постороннее» в чате не годится: уточнения
# («а второй?», «а ближе?») не содержат ни одного признака темы (урок задачи 8 практики).
# «Рецепт» нарочно не в списке: «аптека, где продают без рецепта» — нормальный вопрос.
OFF_TOPIC = re.compile(
    r"стих|поэм|сочини|сочинение|эссе|анекдот|шутк|переведи|перевод\w* на "
    r"|уравнени|интеграл|реши задач|напиши (код|программ|функци|скрипт)",
    re.IGNORECASE)

REFUSAL = ("Я отвечаю только на вопросы о местах Ростова-на-Дону: кафе и рестораны, аптеки, "
           "магазины, пункты выдачи и т.п. Спросите, например: «Где поесть пельменей?» или "
           "«Ближайшая аптека к Большой Садовой, 50».")


def _last_user_text(llm_request: LlmRequest) -> str | None:
    """Текст последнего сообщения, ЕСЛИ это реплика пользователя.

    На первом шаге ответа последнее сообщение — вопрос пользователя. На следующих
    шагах последним идёт результат инструмента (function_response) — тогда None:
    проверять вопрос второй раз незачем, он уже прошёл гардрейл.
    """
    if not llm_request.contents:
        return None
    last = llm_request.contents[-1]
    if last.role != "user" or any(p.function_response for p in last.parts or []):
        return None
    return " ".join(p.text for p in last.parts or [] if p.text)


def guardrail(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
    """Отсекает посторонние просьбы ДО вызова модели: ответ-отказ без единого токена LLM."""
    text = _last_user_text(llm_request)
    if text and OFF_TOPIC.search(text):
        # Веб-слой узнаёт такой ответ по тому, что за него не было ни одного запроса к LLM.
        return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=REFUSAL)]))
    return None


# ── обрезка истории ─────────────────────────────────────────────────────────

def _is_user_turn(content: types.Content) -> bool:
    """Начало «хода» пользователя: его текстовая реплика (не результат инструмента).

    В истории ADK результаты инструментов тоже идут с role="user" — отличаем их
    по наличию function_response.
    """
    parts = content.parts or []
    return content.role == "user" and any(p.text for p in parts) \
        and not any(p.function_response for p in parts)


def trim_history(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
    """Оставляет в истории последние MAX_USER_TURNS реплик пользователя со всем, что после них.

    Резать можно только по границе хода пользователя: если отрезать посередине,
    у модели останется function_response без породившего его function_call —
    DeepSeek такой запрос отвергнет с ошибкой.
    """
    starts = [i for i, c in enumerate(llm_request.contents) if _is_user_turn(c)]
    if len(starts) > MAX_USER_TURNS:
        cut = starts[-MAX_USER_TURNS]                 # индекс начала самого старого оставляемого хода
        llm_request.contents = llm_request.contents[cut:]
    return None                                       # None — модель вызывается как обычно
