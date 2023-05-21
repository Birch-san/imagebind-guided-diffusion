import torch.nn.functional as F
from torch import FloatTensor

def spherical_dist_loss(x: FloatTensor, y: FloatTensor) -> FloatTensor:
  x = F.normalize(x, dim=-1)
  y = F.normalize(y, dim=-1)
  return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)