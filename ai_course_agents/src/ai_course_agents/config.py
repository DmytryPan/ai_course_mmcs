"""Настройки сервиса: всё, что зависит от окружения, читается здесь и только здесь.

Источник значений — файл ai_course_agents/.env (см. .env.example) и переменные
окружения процесса. Остальной код получает настройки через get_settings() и не
обращается к os.environ напрямую: так видно, от чего зависит сервис, и тесты
могут подменить настройки целиком.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# src/ai_course_agents/config.py → parents[0] = пакет, [1] = src, [2] = проект ai_course_agents,
# [3] = корень репозитория курса (там лежит week2/practice/poi.sqlite).
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
REPO_ROOT = PROJECT_DIR.parent


@dataclass(frozen=True)
class Settings:
    """Неизменяемый набор настроек. frozen=True — чтобы никто не поменял их на лету."""

    # ── LLM ────────────────────────────────────────────────────────────────
    deepseek_api_key: str
    # Строка модели в формате LiteLLM «провайдер/модель». Провайдер deepseek/ сам знает
    # адрес API и умеет разбирать reasoning_content (размышления модели) — их мы
    # показываем пользователю как «ход мысли».
    llm_model: str
    # Тариф для подсчёта цены прогона, ₽ за миллион токенов: вход / вход из кэша / выход.
    # DeepSeek кэширует повторяющееся начало запроса (инструкция + схемы инструментов)
    # и берёт за такие токены в разы меньше — это заметная часть каждого запроса.
    price_in_rub: float
    price_cached_rub: float
    price_out_rub: float
    # Предохранитель: сколько запросов к LLM разрешено за один ответ (RunConfig.max_llm_calls).
    max_llm_calls: int
    # Сколько прогонов агента могут одновременно ждать LLM на весь процесс.
    max_concurrent_runs: int

    # ── Данные ─────────────────────────────────────────────────────────────
    poi_db: Path          # база мест Ростова (только чтение)
    data_dir: Path        # всё, что сервис создаёт сам: сессии, пользователи, индекс, журналы

    # ── Веб ────────────────────────────────────────────────────────────────
    jwt_secret: str
    jwt_ttl_hours: int
    host: str
    port: int

    @property
    def sessions_db(self) -> Path:
        """История диалогов ADK (SqliteSessionService)."""
        return self.data_dir / "sessions.sqlite"

    @property
    def app_db(self) -> Path:
        """Пользователи и список чатов."""
        return self.data_dir / "app.sqlite"

    @property
    def qdrant_dir(self) -> Path:
        """Векторная БД Qdrant в embedded-режиме: просто папка на диске."""
        return self.data_dir / "qdrant"

    @property
    def runs_dir(self) -> Path:
        """Журналы прогонов агента в JSONL, по файлу на день."""
        return self.data_dir / "runs"

    @property
    def enrichment_file(self) -> Path:
        """Кэш описаний заведений, сгенерированных LLM (коммитится в git)."""
        return self.data_dir / "enrichment.json"


def _jwt_secret(data_dir: Path) -> str:
    """Секрет для подписи JWT.

    Если JWT_SECRET задан в .env — берём его. Иначе генерируем случайный один раз
    и сохраняем в data/.jwt_secret: без этого при каждом перезапуске сервера
    все пользователи разлогинивались бы (старые токены перестали бы проходить проверку).
    """
    if value := os.getenv("JWT_SECRET"):
        return value
    path = data_dir / ".jwt_secret"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    data_dir.mkdir(parents=True, exist_ok=True)
    value = secrets.token_urlsafe(48)
    path.write_text(value, encoding="utf-8")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Читает .env один раз за процесс (lru_cache) и собирает Settings.

    load_dotenv НЕ перезаписывает переменные, уже заданные в окружении процесса:
    так в тестах или на сервере можно переопределить любое значение без правки .env.
    """
    load_dotenv(PROJECT_DIR / ".env")
    data_dir = Path(os.getenv("DATA_DIR", PROJECT_DIR / "data")).resolve()
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "deepseek/deepseek-flash"),
        price_in_rub=float(os.getenv("PRICE_IN_RUB", "30")),
        price_cached_rub=float(os.getenv("PRICE_CACHED_RUB", "3")),
        price_out_rub=float(os.getenv("PRICE_OUT_RUB", "120")),
        max_llm_calls=int(os.getenv("MAX_LLM_CALLS", "12")),
        max_concurrent_runs=int(os.getenv("MAX_CONCURRENT_RUNS", "8")),
        poi_db=Path(os.getenv("POI_DB", REPO_ROOT / "week2" / "practice" / "poi.sqlite")).resolve(),
        data_dir=data_dir,
        jwt_secret=_jwt_secret(data_dir),
        jwt_ttl_hours=int(os.getenv("JWT_TTL_HOURS", "168")),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
