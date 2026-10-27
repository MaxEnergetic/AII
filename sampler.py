import torch
from torch.utils.data import Sampler 

class NoReplacementSampler(Sampler):
    def __init__(self,dataset, seed = 42):
        self.size = len(dataset)
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        indices = torch.randperm(
            self.size,
            generator =self.generator
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.size

