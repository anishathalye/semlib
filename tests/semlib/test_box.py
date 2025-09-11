from semlib.box import Box


def test_box() -> None:
    box1: Box[int] = Box(42)
    assert box1.value == 42
    box2: Box[str] = Box("hello")
    assert box2.value == "hello"
