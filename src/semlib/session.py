from semlib.apply import Apply
from semlib.extrema import Extrema
from semlib.filter import Filter
from semlib.find import Find
from semlib.reduce import Reduce
from semlib.sort import Sort


# don't need to include
# - Map (is a subclass of Filter)
# - Compare (is a subclass of Sort)
class Session(Sort, Filter, Extrema, Find, Apply, Reduce):
    """A session provides a context for performing Semlib operations.

    Sessions provide defaults (e.g., default model), manage caching, implement concurrency control, and track costs.

    All of a Session's methods have analogous standalone functions (e.g., the [Session.map][semlib.map.Map.map] method
    has an analogous [map][semlib.map.map] function). It is recommended to use a Session for any non-trivial use of the
    semlib library.
    """
