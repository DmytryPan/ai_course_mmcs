"""Фабрика веб-приложения FastAPI.

    uv run ai-course-agents                                   # штатный запуск (см. __init__.main)
    uv run uvicorn ai_course_agents.web.app:create_app --factory --port 8000

Что где:
  /api/auth/*    — регистрация и вход (web/auth.py)
  /api/chats/*   — чаты и история (web/chats.py), ответ агента потоком SSE (web/stream.py)
  /api/health    — жив ли сервис и готов ли поиск
  /              — страница интерфейса (web/static/index.html)

Lifespan — код, который FastAPI выполняет один раз при старте (до первого запроса)
и один раз при остановке. Всё тяжёлое и долгоживущее создаётся там, а не на каждый
запрос: модель эмбеддингов (~секунда загрузки и сотни МБ памяти), Qdrant, Runner ADK.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService

from ai_course_agents.agent import tools
from ai_course_agents.agent.agent import APP_NAME, build_agent
from ai_course_agents.config import Settings, get_settings
from ai_course_agents.search.vector import SemanticIndex
from ai_course_agents.web import auth, chats, stream
from ai_course_agents.web.db import AppDB
from ai_course_agents.web.deps import Services

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Собрать приложение. settings можно передать явно — так тесты подставляют свои пути."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # ── старт ──
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        db = AppDB(settings.app_db)
        await db.init()                                    # таблицы users и chats

        # Семантический индекс: загрузка модели — работа CPU, уводим в поток,
        # чтобы не блокировать цикл событий. Индекс отдаём инструментам агента.
        index = await asyncio.to_thread(
            SemanticIndex, settings.embed_model, settings.qdrant_dir, settings.models_dir)
        if not index.client.collection_exists(index.collection):
            log.warning("индекс семантического поиска не построен — semantic_search будет "
                        "возвращать ошибку. Постройте: uv run python -m ai_course_agents.search.index")
        tools.set_index(index)

        # История диалогов ADK в SQLite: переживает перезапуск сервера.
        sessions = SqliteSessionService(str(settings.sessions_db))
        runner = Runner(app_name=APP_NAME, agent=build_agent(settings), session_service=sessions)
        app.state.services = Services(
            settings=settings, db=db, sessions=sessions, runner=runner,
            llm_slots=asyncio.Semaphore(settings.max_concurrent_runs))
        app.state.index_ready = index.client.collection_exists(index.collection)
        log.info("сервис готов: модель %s, индекс %s", settings.llm_model,
                 "есть" if app.state.index_ready else "НЕТ")
        yield                                              # ← здесь приложение обслуживает запросы
        # ── остановка ──
        await runner.close()
        index.client.close()                               # освободить папку Qdrant

    app = FastAPI(title="Гид по Ростову", lifespan=lifespan)
    app.include_router(auth.router)
    app.include_router(chats.router)
    app.include_router(stream.router)

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "search_index": app.state.index_ready,
                "model": settings.llm_model}

    # Статика фронтенда (этап 4): /static/app.js, /static/styles.css …
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
