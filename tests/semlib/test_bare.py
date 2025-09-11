import pytest

from semlib.bare import Bare


def test_bare_invalid_extract() -> None:
    b = Bare(int)
    with pytest.raises(TypeError, match="expected instance of"):
        b._extract("hi")  # noqa: SLF001
