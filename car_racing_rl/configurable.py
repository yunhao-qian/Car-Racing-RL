import copy
from abc import ABC, abstractmethod
from typing import Unpack


class Configurable[T](ABC):

    def __init__(self, **kwargs: Unpack[T]) -> None:
        self._config = self.default_config()
        self._config.update(kwargs)

    @property
    def config(self) -> T:
        return copy.deepcopy(self._config)

    @staticmethod
    @abstractmethod
    def default_config() -> T: ...
