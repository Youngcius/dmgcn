import dgl
import torch
from torch import nn


def batcher():
    def batcher_dev(batch):
        graphs, labels = zip(*batch)
        labels = torch.stack(labels, 0)
        batch_graphs = dgl.batch(graphs)
        return batch_graphs, labels

    return batcher_dev


def count_model_parameter(model: nn.Module):
    return sum([p.numel() for p in model.parameters()])


