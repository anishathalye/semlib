import asyncio
from collections.abc import Callable

import pydantic
import pytest

from semlib.compare import _DEFAULT_TEMPLATE, _DEFAULT_TEMPLATE_BY
from semlib.sort import Sort, sort, sort_sync
from semlib.sort.algorithm import BordaCount, QuickSort
from tests.conftest import LLMMocker


def test_sort(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    measurements: list[str] = ["1 mile", "1.4 kilometers", "3 inches", "3 centimeters", "1 foot"]

    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(a="1 mile", b="1.4 kilometers"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE.format(a="1 mile", b="1 foot"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE.format(a="1 mile", b="3 inches"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE.format(a="1 mile", b="3 centimeters"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE.format(a="1.4 kilometers", b="1 foot"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE.format(a="1.4 kilometers", b="3 inches"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE.format(a="1.4 kilometers", b="3 centimeters"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE.format(a="1 foot", b="3 inches"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE.format(a="1 foot", b="3 centimeters"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE.format(a="3 inches", b="3 centimeters"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE.format(b="1 mile", a="1.4 kilometers"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(b="1 mile", a="1 foot"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(b="1 mile", a="3 inches"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(b="1 mile", a="3 centimeters"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(b="1.4 kilometers", a="1 foot"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(b="1.4 kilometers", a="3 inches"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(b="1.4 kilometers", a="3 centimeters"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(b="1 foot", a="3 inches"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(b="1 foot", a="3 centimeters"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(b="3 inches", a="3 centimeters"): '{"choice": "B"}',
        }
    )

    sorter = Sort()
    with mocker.patch_llm():
        sort_measurements: list[str] = asyncio.run(sorter.sort(measurements, algorithm=QuickSort(randomized=False)))
    assert sort_measurements == ["3 centimeters", "3 inches", "1 foot", "1.4 kilometers", "1 mile"]
    assert sorter.total_cost() == 9.0

    with mocker.patch_llm():
        sort_measurements = sort_sync(measurements, algorithm=BordaCount())
    assert sort_measurements == ["3 centimeters", "3 inches", "1 foot", "1.4 kilometers", "1 mile"]


@pytest.mark.asyncio
async def test_sort_equal(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(a="12 inches", b="1 meter"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(a="1 foot", b="1 meter"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(a="1 foot", b="12 inches"): '{"choice": "neither"}',
        }
    )

    items = ["1 meter", "12 inches", "1 foot"]
    with mocker.patch_llm():
        result = await sort(items, algorithm=QuickSort(randomized=False), task="choose_greater_or_abstain")
    assert result == ["12 inches", "1 foot", "1 meter"]


def test_sort_by(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE_BY.format(criteria="wavelength", a="red", b="green"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE_BY.format(criteria="wavelength", a="red", b="blue"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE_BY.format(criteria="wavelength", a="green", b="blue"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE_BY.format(criteria="wavelength", a="green", b="red"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE_BY.format(criteria="wavelength", a="blue", b="red"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE_BY.format(criteria="wavelength", a="blue", b="green"): '{"choice": "B"}',
        }
    )
    colors = ["red", "green", "blue"]
    with mocker.patch_llm():
        result = sort_sync(colors, by="wavelength")
    assert result == ["blue", "green", "red"]


@pytest.mark.asyncio
async def test_sort_raises_quicksort(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(a="12 inches", b="1 meter"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(a="1 foot", b="1 meter"): '{"choice": "1 meter"}',  # invalid
            _DEFAULT_TEMPLATE.format(a="1 foot", b="12 inches"): '{"choice": "neither"}',
        }
    )

    items = ["1 meter", "12 inches", "1 foot"]
    with mocker.patch_llm(), pytest.raises(pydantic.ValidationError):
        await sort(items, algorithm=QuickSort(randomized=False))


@pytest.mark.asyncio
async def test_sort_raises_borda_count(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(a="12 inches", b="1 meter"): '{"choice": "invalid"}',
            _DEFAULT_TEMPLATE.format(a="1 meter", b="12 inches"): '{"choice": "invalid"}',
        }
    )

    items = ["1 meter", "12 inches"]
    with mocker.patch_llm(), pytest.raises(pydantic.ValidationError):
        await sort(items, algorithm=BordaCount(), max_concurrency=1)
