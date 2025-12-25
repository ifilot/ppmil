from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ppmil")
except PackageNotFoundError:
    __version__ = "0.0.0"  # optional fallback