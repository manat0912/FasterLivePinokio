import sys
import inspect
sys.path.insert(0, 'c:/pinokio/api/FasterLivePinokio/app')
from src.models.pytorch_modules.warping_network import WarpingNetwork
from src.models.pytorch_modules.spade_generator import SPADEGenerator

print("WarpingNetwork:")
print(inspect.signature(WarpingNetwork.__init__))
print("SPADEGenerator:")
print(inspect.signature(SPADEGenerator.__init__))
