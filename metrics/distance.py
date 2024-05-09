import torch
from torch.nn.functional import normalize

def cross_euclidean_distance(x:torch.Tensor, y:torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.cdist(x, y, p=2).cpu()

def cross_cosine_distances(x:torch.Tensor, y:torch.Tensor, **kwargs) -> torch.Tensor:
    return (1.0 - normalize(x, p=2, dim=-1)@(normalize(y, p=2, dim=-1).T)).cpu() 

