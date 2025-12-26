from abc import abstractmethod
from ..util.gto import GTO

class ElectronRepulsionEngine:
    """
    Abstract class for Two Electron Integral Engine
    """

    @abstractmethod
    def repulsion_primitive(self, g1:GTO, g2:GTO, g3:GTO, g4:GTO):
        pass