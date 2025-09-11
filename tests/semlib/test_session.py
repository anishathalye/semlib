from collections.abc import Callable

import pydantic
import pytest

from semlib import InMemoryCache, Session
from semlib._internal.constants import DEFAULT_MODEL
from semlib.compare import _DEFAULT_TEMPLATE
from tests.conftest import LLMMocker


@pytest.mark.asyncio
async def test_session(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    session = Session(cache=InMemoryCache())

    measurements: list[str] = ["1 mile", "1.4 kilometers", "1 foot", "3 inches", "3 centimeters"]

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
            "Convert the length 1 mile to centimeters, rounding to the nearest integer.": '{"centimeters": 160934}',
            "Convert the length 1.4 kilometers to centimeters, rounding to the nearest integer.": '{"centimeters": 140000}',
            "Convert the length 1 foot to centimeters, rounding to the nearest integer.": '{"centimeters": 30}',
            "Convert the length 3 inches to centimeters, rounding to the nearest integer.": '{"centimeters": 8}',
            "Convert the length 3 centimeters to centimeters, rounding to the nearest integer.": '{"centimeters": 3}',
        }
    )

    with mocker.patch_llm():
        sort_measurements: list[str] = await session.sort(measurements)

    assert sort_measurements == ["3 centimeters", "3 inches", "1 foot", "1.4 kilometers", "1 mile"]
    assert session.total_cost() == 20

    # running again should hit the cache and incur no additional cost
    with mocker.patch_llm():
        sort_measurements = await session.sort(measurements)
    assert session.total_cost() == 20

    # unless we clear the cache
    session.clear_cache()
    with mocker.patch_llm():
        sort_measurements = await session.sort(measurements)
    assert session.total_cost() == 40

    class Length(pydantic.BaseModel):
        centimeters: int

    with mocker.patch_llm():
        measurements_cm: list[Length] = await session.map(
            sort_measurements,
            template="Convert the length {} to centimeters, rounding to the nearest integer.",
            return_type=Length,
        )
    assert session.total_cost() == 45

    assert measurements_cm == [
        Length(centimeters=3),
        Length(centimeters=8),
        Length(centimeters=30),
        Length(centimeters=140_000),
        Length(centimeters=160_934),
    ]

    assert session.model == DEFAULT_MODEL
