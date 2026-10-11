"""Пробный прогон связки ADK → LiteLLM → DeepSeek (контрольная точка этапа 0).

    uv run python scripts/smoke_llm.py          # настоящий запрос к DeepSeek (нужен ключ в .env)
    uv run python scripts/smoke_llm.py --mock   # без сети: LiteLLM подставляет готовый ответ

Режим --mock проверяет всё, кроме самого DeepSeek: что ADK собирает запрос,
LiteLLM его принимает и возвращает ответ в формате, который ADK умеет разобрать.
"""

from __future__ import annotations

import asyncio
import sys

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from ai_course_agents.config import get_settings


async def main(mock: bool) -> None:
    settings = get_settings()
    if not mock and not settings.deepseek_api_key:
        sys.exit("DEEPSEEK_API_KEY пуст: впиши ключ в ai_course_agents/.env или запусти с --mock")

    # LiteLlm(model, **kwargs): все лишние аргументы ADK передаёт в litellm.acompletion.
    # api_key — ключ DeepSeek; mock_response — встроенная в LiteLLM заглушка: вместо
    # запроса в сеть возвращается этот текст, оформленный как настоящий ответ модели.
    extra = {"mock_response": "Привет! Связка ADK → LiteLLM работает."} if mock else {}
    model = LiteLlm(model=settings.llm_model, api_key=settings.deepseek_api_key or "mock", **extra)

    agent = LlmAgent(name="smoke", model=model,
                     instruction="Отвечай по-русски одним коротким предложением.")

    # Runner — цикл агента в ADK; InMemorySessionService хранит историю в памяти процесса.
    sessions = InMemorySessionService()
    runner = Runner(app_name="smoke", agent=agent, session_service=sessions)
    session = await sessions.create_session(app_name="smoke", user_id="dev")

    question = types.Content(role="user", parts=[types.Part(text="Скажи, что ты на связи.")])
    # StreamingMode.SSE — ответ приходит кусками (partial=True), как это будет в веб-сервисе.
    config = RunConfig(streaming_mode=StreamingMode.SSE, max_llm_calls=3)

    print(f"модель: {settings.llm_model}{' (mock)' if mock else ''}")
    async for event in runner.run_async(user_id="dev", session_id=session.id,
                                        new_message=question, run_config=config):
        for part in (event.content.parts if event.content else []):
            if part.text:
                kind = "💭 мысль" if part.thought else ("… кусок" if event.partial else "✅ ответ")
                print(f"{kind}: {part.text!r}")
        if event.usage_metadata and not event.partial:
            u = event.usage_metadata
            print(f"токены: вход {u.prompt_token_count}, выход {u.candidates_token_count}")


if __name__ == "__main__":
    asyncio.run(main(mock="--mock" in sys.argv))
