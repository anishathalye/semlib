from typing import Any, cast

import pydantic


class Bare[T]:
    """A marker to indicate that a function should return a bare value of type `T`.

    This can be passed to the `return_type` parameter of functions like [prompt][semlib._internal.base.Base.prompt]. For
    situations where you want to extract a single value of a given base type (like `int` or `list[float]`), this is more
    convenient than the alternative of defining a Pydantic model with a single field for the purpose of extracting that
    value.

    Examples:
        Extract a bare value using prompt:
        >>> await session.prompt("What is 2+2?", return_type=Bare(int))
        4

        Influence model output using `class_name` and `field_name`:
        >>> await session.prompt(
        ...     "Give me a list",
        ...     return_type=Bare(
        ...         list[int], class_name="list_of_three_values", field_name="primes"
        ...     ),
        ... )
        [3, 7, 11]
    """

    def __init__(self, typ: type[T], /, class_name: str | None = None, field_name: str | None = None):
        """Initialize a Bare instance.

        Args:
            typ: The type of the bare value to extract.
            class_name: Name for a dynamically created Pydantic model class. If not provided, defaults to
                the name of `typ`. This name is visible to the LLM and may affect model output.
            field_name: Name for the field in the dynamically created Pydantic model that holds the bare value.
                If not provided, defaults to "value". This name is visible to the LLM and may affect model output.
        """
        self._typ = typ
        self._class_name = class_name if class_name is not None else typ.__name__
        self._field_name = field_name if field_name is not None else "value"
        field_definitions: Any = {self._field_name: (self._typ, ...)}
        self._model: type[pydantic.BaseModel] = pydantic.create_model(self._class_name, **field_definitions)

    def _extract(self, obj: Any) -> T:
        if isinstance(obj, self._model):
            return cast(T, getattr(obj, self._field_name))
        msg = f"expected instance of {self._model.__name__}, got {type(obj).__name__}"
        raise TypeError(msg)
