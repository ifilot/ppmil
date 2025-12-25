from abc import ABC, abstractmethod
from .import GTO

class ElectronRepulsionEngine(ABC):
    """
    Abstract class for Two Electron Integral Engine
    """

    @abstractmethod
    def eri_primitive(self, g1:GTO, g2:GTO, g3:GTO, g4:GTO):
        pass