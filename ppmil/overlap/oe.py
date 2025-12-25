from abc import ABC, abstractmethod
from .. import GTO

class OverlapEngine(ABC):
    """
    Abstract class for Two Electron Integral Engine
    """

    @abstractmethod
    def overlap_primitive(self, g1:GTO, g2:GTO, g3:GTO, g4:GTO):
        pass