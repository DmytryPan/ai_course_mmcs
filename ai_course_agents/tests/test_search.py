"""Тесты семантического поиска на рабочем индексе (data/qdrant).

Нужен построенный индекс: `uv run python -m ai_course_agents.search.index`.
Если его нет — тесты пропускаются (skip), а не падают: на чистой машине сначала
строится индекс, это отдельный шаг. Сервер при запуске тестов должен быть остановлен —
embedded-Qdrant держит папку одним процессом.
"""

import pytest

from ai_course_agents.config import get_settings
from ai_course_agents.poi.kinds import FOOD_KINDS
from ai_course_agents.search.vector import SemanticIndex, collection_name, query_stems

settings = get_settings()


def test_query_stems_drop_stopwords_and_endings():
    # «где», «поесть» — служебные; у «пельменей» отрезаются 2 буквы, у «пива» — 1
    assert query_stems("Где поесть пельменей и пива?") == ["пельмен", "пив"]


@pytest.fixture(scope="module")
def index():
    if not (settings.qdrant_dir / "collection" / collection_name(settings.embed_model)).exists():
        pytest.skip("индекс не построен: uv run python -m ai_course_agents.search.index")
    ix = SemanticIndex(settings.embed_model, settings.qdrant_dir, settings.models_dir)
    yield ix
    ix.client.close()


def test_pelmeni_finds_dumpling_places_first(index):
    """Сценарий из задания: человек забыл название и пишет «пельмени»."""
    top = [h["name"] for h in index.search_sync("пельмени", kinds=FOOD_KINDS, limit=3)]
    assert {"Пельмень", "Пельмешка"} <= set(top)


def test_geo_radius_limits_results(index):
    """С точкой и радиусом поиск не выходит за радиус (центр Пушкинской, 1.5 км)."""
    hits = index.search_sync("суши", kinds=FOOD_KINDS, lat=47.228084, lon=39.7134,
                             radius_m=1500, limit=10)
    from ai_course_agents.poi.repo import distance_m
    assert hits and all(distance_m(47.228084, 39.7134, h["lat"], h["lon"]) <= 1500 for h in hits)


def test_hybrid_reports_how_place_matched(index):
    hits = index.search_sync("шаурма", kinds=FOOD_KINDS, limit=5)
    assert all(h["match"] in {"по смыслу", "по словам", "по словам и смыслу"} for h in hits)
    assert any("Шаурма" in (h["name"] or "") for h in hits)
