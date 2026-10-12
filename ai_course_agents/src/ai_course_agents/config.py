"""Настройки сервиса: всё, что зависит от окружения, читается здесь и только здесь.

Источник значений — файл ai_course_agents/.env (см. .env.example) и переменные
окружения процесса. Остальной код получает настройки через get_settings() и не
обращается к os.environ напрямую: так видно, от чего зависит сервис, и тесты
могут подменить настройки целиком.
"""

# Отложенное вычисление аннотаций типов: можно писать `str | None` и ссылаться
# на классы, объявленные ниже, — аннотации не вычисляются при импорте модуля.
from __future__ import annotations

import os                               # чтение переменных окружения: os.getenv("ИМЯ", "по умолчанию")
import secrets                          # криптостойкий генератор случайных строк (для секрета JWT)
from dataclasses import dataclass       # класс-«запись» с полями без ручного __init__
from functools import lru_cache         # кэш результата функции: get_settings() вычисляется один раз
from pathlib import Path                # пути к файлам, одинаково работают на Windows и Linux

from dotenv import load_dotenv          # пакет python-dotenv: читает .env в os.environ

# Где лежат файлы проекта. Path(__file__) — путь к этому файлу (config.py);
# .resolve() делает его абсолютным; .parent — папка, в которой он лежит.
#   config.py → parent = src/ai_course_agents (пакет)
#             → parents[1] от пакета = ai_course_agents (корень uv-проекта, там .env)
#             → parent проекта = корень репозитория курса (там week2/practice/poi.sqlite)
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
REPO_ROOT = PROJECT_DIR.parent


# frozen=True — объект нельзя изменить после создания (settings.port = 1 бросит ошибку):
# настройки читаются один раз при старте и дальше только используются.
@dataclass(frozen=True)
class Settings:
    """Неизменяемый набор настроек сервиса."""

    # ── LLM ────────────────────────────────────────────────────────────────
    deepseek_api_key: str               # ключ DeepSeek API; пустая строка — ключа нет
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
    # Если агент зациклится, ADK остановит его на этом числе, а не будет тратить деньги вечно.
    max_llm_calls: int
    # Сколько прогонов агента могут одновременно ждать LLM на весь процесс (этап 3, семафор).
    max_concurrent_runs: int

    # ── Данные ─────────────────────────────────────────────────────────────
    poi_db: Path          # база мест Ростова (только чтение)
    data_dir: Path        # всё, что сервис создаёт сам: сессии, пользователи, индекс, журналы

    # ── Веб ────────────────────────────────────────────────────────────────
    jwt_secret: str       # секрет, которым подписываются токены входа (этап 3)
    jwt_ttl_hours: int    # сколько часов живёт токен входа; по умолчанию неделя
    host: str             # на каком адресе слушает веб-сервер (127.0.0.1 — только эта машина)
    port: int             # порт веб-сервера

    # Пути к файлам внутри data_dir. @property — вычисляемое поле: обращаемся как к полю
    # (settings.sessions_db), а значение каждый раз собирается из data_dir. Так все пути
    # гарантированно лежат в одной папке, и при смене DATA_DIR переезжают вместе.
    @property
    def sessions_db(self) -> Path:
        """История диалогов ADK (SqliteSessionService)."""
        return self.data_dir / "sessions.sqlite"   # оператор / у Path склеивает пути

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
    # «Морж» := присваивает и сразу проверяет: если переменная задана и не пустая — вернуть её.
    if value := os.getenv("JWT_SECRET"):
        return value
    path = data_dir / ".jwt_secret"
    if path.exists():                                   # секрет уже сгенерирован раньше
        return path.read_text(encoding="utf-8").strip()
    data_dir.mkdir(parents=True, exist_ok=True)         # создать data/, если её ещё нет
    value = secrets.token_urlsafe(48)                   # 48 случайных байт → ~64 символа
    path.write_text(value, encoding="utf-8")            # запомнить на следующие запуски
    return value


# lru_cache(maxsize=1): первый вызов выполняет функцию и запоминает результат,
# все следующие вызовы сразу возвращают тот же объект Settings. Поэтому get_settings()
# можно звать где угодно, не боясь, что .env будет перечитываться на каждый запрос.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Читает .env один раз за процесс и собирает Settings.

    load_dotenv НЕ перезаписывает переменные, уже заданные в окружении процесса:
    так в тестах или на сервере можно переопределить любое значение без правки .env.
    """
    load_dotenv(PROJECT_DIR / ".env")    # пары KEY=VALUE из .env → os.environ
    # Папка данных: из переменной DATA_DIR или ai_course_agents/data по умолчанию.
    data_dir = Path(os.getenv("DATA_DIR", PROJECT_DIR / "data")).resolve()
    # Все значения из окружения приходят строками — числа приводим через int()/float().
    # Второй аргумент os.getenv — значение по умолчанию, если переменной нет.
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
        jwt_ttl_hours=int(os.getenv("JWT_TTL_HOURS", "168")),      # 168 ч = 7 дней
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
