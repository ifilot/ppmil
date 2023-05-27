from .gto import GTO

class CGF:
    """
    Contracted Gaussian Type Orbital
    """
    def __init__(self, _p=[0,0,0]):
        """
        Default constructor
        """
        self.gtos = []
        self.p = _p

    def __reduce__(self):
        """
        Used to pickle the class
        """
        return (self.__class__, tuple([self.p]), (self.p, self.gtos))

    def __str__(self):
        """
        Get string representation of the Contracted Gaussian Functional
        """
        res = "CGF; R=(%f,%f,%f)\n" % tuple(self.p)
        for i,gto in enumerate(self.gtos):
            res += " %02i | %s" % (i+1, str(gto))
        return res

    def add_gto(self, c, alpha, l, m, n):
        """
        Add Gaussian Type Orbital to Contracted Gaussian Function
        """
        self.gtos.append(GTO(c, alpha, self.p, [l, m, n]))

    def get_amp_f(self, x, y, z):
        """
        Get the amplitude of the wave function at position r
        """
        return self.cgf.get_amp_f(x, y, z)

    def get_amp(self, r):
        """
        Get the amplitude of the wave function at position r
        """
        return self.cgf.get_amp_f(r[0], r[1], r[2])

    def get_grad_f(self, x, y, z):
        """
        Get the gradient (3-vector) of the wave function at position r
        """
        return self.cgf.get_grad_f(x, y, z)

    def get_grad(self, r):
        """
        Get the gradient (3-vector) of the wave function at position r
        """
        return self.cgf.get_grad_f(r[0], r[1], r[2])