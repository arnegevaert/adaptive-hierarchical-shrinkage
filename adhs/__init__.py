from ._adhs import ShrinkageClassifier, ShrinkageRegressor
from ._cross_val_shrinkage import cross_val_shrinkage
from importlib.metadata import version

__version__ = version("adhs")
