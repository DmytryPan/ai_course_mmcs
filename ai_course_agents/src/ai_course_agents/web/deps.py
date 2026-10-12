"""Общие зависимости веб-слоя: сервисы приложения и «текущий пользователь».

FastAPI-зависимость (Depends) — функция, которую FastAPI вызывает перед обработчиком
и передаёт её результат в параметр. Так проверка входа пишется один раз и подключается
к любому маршруту одной строкой: `user: dict = Depends(current_user)`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, status
from google.adk.runners import Runner
from google.adk.sessions.base_session_service import BaseSessionService

from ai_course_agents.config import Settings
from ai_course_agents.web.db import AppDB
from ai_course_agents.web.security import COOKIE_NAME, read_token


@dataclass
class Services:
    """Всё долгоживущее, что создаётся один раз при старте (web/app.py → lifespan)."""

    settings: Settings
    db: AppDB                        # пользователи и чаты
    sessions: BaseSessionService     # история диалогов ADK
    runner: Runner                   # цикл агента
    # Сколько прогонов агента одновременно ждут LLM на весь процесс. Остальные ждут
    # своей очереди — так всплеск пользователей не упрётся в лимиты API DeepSeek
    # и не съест всю память. Создаётся в lifespan (семафор должен жить в цикле событий).
    llm_slots: asyncio.Semaphore
    # Замок на каждый чат: в одном чате не может идти два ответа одновременно — иначе
    # две параллельные записи в одну сессию ADK перемешали бы историю.
    chat_locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    def chat_lock(self, chat_id: str) -> asyncio.Lock:
        """Замок чата; создаётся при первом обращении (setdefault — атомарно для asyncio)."""
        return self.chat_locks.setdefault(chat_id, asyncio.Lock())


def get_services(request: Request) -> Services:
    """Сервисы приложения лежат в app.state — FastAPI доступен через request.app."""
    return request.app.state.services


async def current_user(request: Request) -> dict:
    """Пользователь из cookie с токеном. Нет токена, он поддельный/просрочен или
    пользователя удалили → 401, и фронтенд покажет экран входа."""
    services = get_services(request)
    token = request.cookies.get(COOKIE_NAME)
    user_id = read_token(token, services.settings.jwt_secret) if token else None
    user = await services.db.user_by_id(user_id) if user_id else None
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "нужно войти")
    return user
