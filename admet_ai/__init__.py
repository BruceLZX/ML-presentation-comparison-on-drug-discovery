"""Import all submodules of admet_ai."""

__version__ = "2.0.0"

from admet_ai.admet_model import ADMETModel
from admet_ai.admet_predict import admet_predict

__all__ = ["ADMETModel", "admet_predict", "__version__"]
