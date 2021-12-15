import torch

from dmgcn import DMGCN
from utils import  count_model_parameter
model = DMGCN()

print('num paras',count_model_parameter(model))

