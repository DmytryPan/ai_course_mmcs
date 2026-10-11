"""Тесты слоя доступа к базе: числа сверены с базой и с ответами практики недели 2."""

import pytest

from ai_course_agents.config import get_settings
from ai_course_agents.poi.repo import PoiRepo, street_stem


@pytest.fixture(scope="module")
def repo() -> PoiRepo:
    return PoiRepo(get_settings().poi_db)


@pytest.mark.parametrize("street, stem", [
    ("на Белорусской улице", "Белорусск"),     # падеж и слово «улица» не мешают
    ("проспект Будённовский", "Будённовск"),
    ("улица 8 Марта", "Марта"),                # короткое слово не режем: «Мар» — полгорода
])
def test_street_stem(street, stem):
    assert street_stem(street) == stem


async def test_count_24_7_pharmacies(repo):
    assert await repo.count(kind="pharmacy", only_24_7=True) == 19   # как в задаче 1 практики


async def test_chain_address_to_nearest_pharmacy(repo):
    """Цепочка задачи 2 практики: Белорусская, 44 → «Лаки Фарма» (id 1728), 123 м."""
    home = await repo.geocode("Белорусской", "44")
    assert home["precision"] == "дом"
    best = (await repo.nearest(home["lat"], home["lon"], kind="pharmacy", limit=1))[0]
    assert (best["id"], best["distance_m"]) == (1728, 123)
    assert (await repo.details(1728))["opening_hours"] == "Mo-Su 08:00-21:00"


async def test_geocode_does_not_average_different_streets(repo):
    """«Садовая» — это две улицы; центр берём по одной, вторую отдаём в other_streets."""
    point = await repo.geocode("Садовая")
    assert point["street"] == "Большая Садовая улица"
    assert point["other_streets"] == ["Садовая улица"]
    assert 39.70 < point["lon"] < 39.74          # центр города, а не среднее двух улиц (~39.76)


async def test_geocode_unknown_street(repo):
    assert await repo.geocode("Хогвартсская", "7") is None


async def test_search_reports_total_and_limit(repo):
    total, rows = await repo.search(kind="cafe", street="Пушкинской", limit=2)
    assert total == 4 and len(rows) == 2
