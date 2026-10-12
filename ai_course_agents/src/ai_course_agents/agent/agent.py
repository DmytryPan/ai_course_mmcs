"""Сборка агента: модель DeepSeek через LiteLLM, инструкция, инструменты.

Агент — это только конфигурация (как Agent в Agents SDK из практики). Цикл «запрос к
модели → вызов инструмента → снова запрос» выполняет Runner; его создаёт вызывающий код:
консольный скрипт (agent/__main__.py) или веб-слой (web/).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo        # часовые пояса; на Windows данные даёт пакет tzdata

from google.adk.agents import LlmAgent                         # агент ADK на основе LLM
from google.adk.agents.readonly_context import ReadonlyContext  # то, что ADK передаёт в провайдер инструкции
from google.adk.models.lite_llm import LiteLlm                 # адаптер ADK → LiteLLM → любой провайдер

from ai_course_agents.agent.tools import LAST_PLACES_KEY, TOOLS
from ai_course_agents.config import Settings, get_settings
from ai_course_agents.poi.kinds import kinds_cheatsheet

APP_NAME = "rostov_guide"     # имя приложения для ADK: под ним хранятся сессии
TZ = ZoneInfo("Europe/Moscow")  # Ростов-на-Дону живёт по московскому времени

# Шаблон системной инструкции. {now}, {kinds}, {last_places} заполняет build_instruction
# через str.format перед каждым запросом к модели. Каждое правило — ответ на конкретную
# ошибку, которую агенты делают без него (см. практику недели 2).
INSTRUCTION = """\
Ты — «Гид по Ростову»: помощник, который отвечает на вопросы о местах Ростова-на-Дону
(кафе, рестораны, аптеки, магазины, пункты выдачи, банки, парки и т.д.)
по базе из OpenStreetMap — 8860 мест. Доступ к базе — только через инструменты.

Правила:
1. Факты (названия, адреса, часы работы, числа, расстояния) бери ТОЛЬКО из результатов
   инструментов. Не отвечай о местах по памяти и не выдумывай; не округляй числа.
2. Вид места (kind) передавай латиницей из шпаргалки ниже. Если подходящего вида нет —
   скажи, что в базе таких мест нет.
3. Адрес или улица и «что-то рядом» → сначала geocode_address, затем nearest_places
   с полученными lat и lon. Координаты и id бери из результатов дословно.
   Если geocode_address вернул other_streets — в ответе назови, от какой улицы считал,
   перечисли другие подходящие улицы и предложи уточнить.
4. Уточнения про уже найденные места («а какой из них ближе», «а до скольки работает
   второй») решай по id из прошлых результатов: rank_by_distance, place_details.
   Не ищи заново то, что уже нашёл.
5. Данные неполные: у многих мест нет названия, часов или адреса. Пустое поле —
   это «нет данных», а не «нет». Так и говори пользователю.
6. Если инструмент вернул error — исправь аргументы и попробуй ещё раз; если не выходит —
   честно скажи, что не получилось.
7. Вопросы не о местах Ростова (стихи, переводы, задачи, общие знания) вежливо отклоняй.

Формат ответа — по-русски, кратко, в markdown:
- место: **название** — адрес; часы работы; расстояние, если считали;
- если показана только часть найденного, скажи сколько всего;
- без вступлений вроде «Конечно!».

Сейчас: {now}.

Шпаргалка видов мест (kind — по-русски):
{kinds}
{last_places}"""

# Шпаргалка «kind — по-русски» строится один раз при импорте: справочник видов
# не меняется, незачем собирать эту строку на каждый запрос.
_KINDS = kinds_cheatsheet()


def build_instruction(ctx: ReadonlyContext) -> str:
    """Инструкция, которую ADK запрашивает перед каждым обращением к модели.

    Функция, а не строка, потому что части инструкции живые:
      • текущее время — без него модель не ответит, работает ли место сейчас;
      • последние найденные места из состояния сессии — опора для уточнений
        «а какой из них…», даже если старая часть истории обрезана.
    Если instruction — функция, ADK не подставляет {переменные} сам, поэтому
    форматируем строку здесь.
    """
    now = datetime.now(TZ)                                # текущее время в Москве/Ростове
    weekdays = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    # ctx.state — состояние сессии (только чтение). Туда инструменты кладут last_places.
    # `or []` — на первом вопросе ключа ещё нет, get вернёт None.
    last = ctx.state.get(LAST_PLACES_KEY) or []
    last_text = ""                                        # пусто — блок не попадёт в инструкцию
    if last:
        # По строке на место: id нужен модели для rank_by_distance/place_details.
        rows = "\n".join(f"- id={p['id']}: {p['name']} (lat={p['lat']}, lon={p['lon']})"
                         for p in last)
        last_text = f"\nМеста из последнего результата поиска (для уточнений):\n{rows}"
    return INSTRUCTION.format(
        # weekday() даёт 0 для понедельника; %d.%m.%Y %H:%M → «12.10.2026 14:05»
        now=f"{weekdays[now.weekday()]}, {now:%d.%m.%Y %H:%M}",
        kinds=_KINDS, last_places=last_text)


def build_model(settings: Settings) -> LiteLlm:
    """Модель через LiteLLM. Строка «deepseek/…» — провайдер DeepSeek в LiteLLM: он знает
    адрес API и переводит reasoning_content (размышления) в Part(thought=True) для ADK."""
    # Всё, кроме model, ADK передаёт в litellm.acompletion как есть — здесь только ключ.
    return LiteLlm(model=settings.llm_model, api_key=settings.deepseek_api_key)


def build_agent(settings: Settings | None = None) -> LlmAgent:
    """Собрать агента. settings можно передать явно (тесты), иначе берутся из .env."""
    settings = settings or get_settings()
    return LlmAgent(
        name="rostov_guide",          # имя агента — оно же author в событиях ADK
        description="Отвечает на вопросы о местах Ростова-на-Дону по базе OpenStreetMap.",
        model=build_model(settings),
        instruction=build_instruction,  # функция: ADK вызовет её перед каждым запросом к модели
        tools=TOOLS,                    # обычные async-функции; ADK обернёт их в FunctionTool
    )
