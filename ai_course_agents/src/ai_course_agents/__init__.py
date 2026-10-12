"""«Гид по Ростову» — агент на Google ADK над базой мест Ростова-на-Дону.

Точка входа `uv run ai-course-agents` (pyproject.toml → [project.scripts]) — функция main().
"""


def main() -> None:
    """Запустить веб-сервис. Импорты внутри функции: пакет можно импортировать
    (например, в тестах) без загрузки uvicorn и всего веб-стека."""
    import logging

    import uvicorn

    from ai_course_agents.config import get_settings

    # Журнал в консоль: время, уровень, модуль, сообщение.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # LiteLLM пишет INFO-строку на каждый запрос к модели и рекламный баннер на каждую
    # ошибку — в журнале сервиса это шум. Оставляем только предупреждения и ошибки.
    import litellm
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    settings = get_settings()
    print(f"Гид по Ростову: http://{settings.host}:{settings.port}")
    # factory=True — uvicorn сам вызовет create_app() и получит приложение.
    uvicorn.run("ai_course_agents.web.app:create_app", factory=True,
                host=settings.host, port=settings.port, log_level="info")
