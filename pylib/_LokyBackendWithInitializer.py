from typing import Any, Callable

from joblib.parallel import LokyBackend


# adapted from https://github.com/joblib/joblib/issues/381#issuecomment-2774934859
class LokyBackendWithInitializer(LokyBackend):
    """
    A specialization of the LokyBackend with an initializer.

    It enables initializing worker processes with custom function calls before
    actually starting the parallel compute.
    """

    def __init__(
        self,
        *,
        initializer: Callable[[], None],
        initargs: tuple[Any, ...] = tuple(),
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.initializer = initializer
        self.initargs = initargs

    def configure(self, *args: Any, **kwargs: Any) -> int:
        return super().configure(
            *args,
            initializer=self.initializer,
            initargs=self.initargs,
            **kwargs,
        )
