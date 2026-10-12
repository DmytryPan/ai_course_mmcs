"""Построение поискового индекса (CLI).

    uv run python -m ai_course_agents.search.index                 # модель из настроек
    uv run python -m ai_course_agents.search.index --model <имя>   # другая модель fastembed
    uv run python -m ai_course_agents.search.index --no-guesses    # без догадок LLM (для сравнения)

Берёт все места с названием или брендом (~6.8 тыс.), собирает для каждого текст-описание
(search/documents.py), кодирует тексты моделью эмбеддингов и складывает в Qdrant (data/qdrant).
Запускать при ОСТАНОВЛЕННОМ сервере: embedded-Qdrant держит папку одним процессом.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from ai_course_agents.config import get_settings
from ai_course_agents.poi.repo import PoiRepo
from ai_course_agents.search.documents import describe
from ai_course_agents.search.vector import SemanticIndex


def load_guesses(path: Path) -> dict[int, str]:
    """Догадки LLM из data/enrichment.json → {id места: текст догадки}. Нет файла — пусто."""
    if not path.exists():
        return {}
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    # Ключи в JSON — строки, а id в базе — числа; записи с guess=null пропускаем.
    return {int(pid): v["guess"] for pid, v in items.items() if v.get("guess")}


async def corpus(use_guesses: bool = True) -> tuple[list[dict[str, Any]], list[str]]:
    """Места для индекса и их тексты-описания (в одном порядке)."""
    settings = get_settings()
    places = await PoiRepo(settings.poi_db).named_places()
    guesses = load_guesses(settings.enrichment_file) if use_guesses else {}
    texts = [describe(p, guesses.get(p["id"])) for p in places]
    return places, texts


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Построить семантический индекс мест")
    parser.add_argument("--model", default=settings.embed_model, help="модель fastembed")
    parser.add_argument("--no-guesses", action="store_true", help="не использовать догадки LLM")
    args = parser.parse_args()

    places, texts = asyncio.run(corpus(use_guesses=not args.no_guesses))
    print(f"мест в индексе: {len(places)}; пример описания:\n  {texts[0]}")
    started = time.perf_counter()
    index = SemanticIndex(args.model, settings.qdrant_dir, settings.models_dir)
    loaded = time.perf_counter()
    index.build(places, texts)
    done = time.perf_counter()
    print(f"модель загружена за {loaded - started:.1f} с, индекс построен за {done - loaded:.1f} с "
          f"→ коллекция {index.collection} в {settings.qdrant_dir}")


if __name__ == "__main__":
    main()
