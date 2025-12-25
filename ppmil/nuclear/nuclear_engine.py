from abc import ABC, abstractmethod
from .. import GTO

class NuclearEngine(ABC):
    """
    Abstract class for Two Electron Integral Engine
    """

    @abstractmethod
    def nuclear_primitive(self, g1:GTO, g2:GTO):
        pass