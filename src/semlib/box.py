class Box[T]:
    """A container that holds a value of type `T`.

    This can be used to tag values, so that they can be distinguished from other values of the same underlying type.
    Such a tag can be useful in the context of methods like [reduce][semlib.reduce.Reduce.reduce], where you can use
    this marker to distinguish leaf nodes from internal nodes in an associative reduce.
    """

    def __init__(self, value: T) -> None:
        """Initialize a Box instance.

        Args:
            value: The value to be contained in the Box.
        """
        self._value = value

    @property
    def value(self) -> T:
        """Get the value contained in the Box.

        Returns:
            The value contained in the Box.
        """
        return self._value
