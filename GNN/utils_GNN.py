"""
Filename: GNN/utils.py
Description: Module containing functions using by PNA GNN model and turing traning and evaluation.
"""
import torch
from torch_geometric.nn.models import PNA
import torch.nn as nn
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import degree
import numpy as np

torch.set_printoptions(precision=8)

# TODO: Read and make it more readable
class ArrayLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, lr_array, last_epoch=-1):
        self.lr_array = lr_array
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Handle the case where the current epoch exceeds the length of the lr_array
        if self.last_epoch < len(self.lr_array):
            return [self.lr_array[self.last_epoch] for _ in self.optimizer.param_groups]
        else:
            # Keep the last specified learning rate if out of bounds
            return [self.lr_array[-1] for _ in self.optimizer.param_groups]

def custom_scheduler(x,lr,wrm_rt,wrm_hght,min_lr_rt,n_epchs):
    lr_tmp = 0.0
    first_part = int(wrm_rt*n_epchs)
    gamma = (min_lr_rt/wrm_hght)**(1./((1.0-wrm_rt)*n_epchs))
    if x <= first_part:
        lr_tmp = lr*(wrm_hght-1.0) * x / first_part + lr
    else:
        lr_tmp = wrm_hght*lr*(gamma)**(x-first_part)
    return lr_tmp

def compute_histogram(data, *, num_bins=-1, max_degree=None):
    deg = degree(data.edge_index[1], dtype=torch.long, num_nodes=data.num_nodes)
    if max_degree is None:
        max_degree = deg.max().item()
    histogram = torch.histc(deg.float(), bins=num_bins, min=0, max=max_degree)
    return histogram

def merge_histograms(dataset, *, num_bins=None, max_degree=None):
    # Determine the maximum degree if not given
    if max_degree is None:
        max_degree = max(degree(data.edge_index[1], dtype=torch.long, num_nodes=data.num_nodes).max().item() for data in dataset)
    if num_bins is None:
        # Sturges' formula
        num_bins_sturges = torch.ceil(torch.log2(torch.tensor(len(dataset))) + 1).to(torch.long)

    # Calculate histograms and sum them
    aggregated_histogram = torch.zeros(num_bins_sturges)
    for data in dataset:
        histogram = compute_histogram(data, num_bins=num_bins_sturges, max_degree=max_degree)
        aggregated_histogram += histogram
    
    # Normalized the aggregated histogram in PNAConv
    return aggregated_histogram

# copied from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience = 7, verbose = False, delta = 0, trace_func = print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model, *, path = './models/checkpoint.pt',):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

class NodeEdgePNA(torch.nn.Module):
    """
    Principal Neighbourhood Aggregation (PNA) architecture. Inspired from "https://arxiv.org/abs/2003.00982" and "https://arxiv.org/abs/2004.05718".

        Args:
            node_in_channels: embedding dimension of the node features, i.e magnetization.
            edge_in_channels: embedding dimension of the edge features, i.e nearest-neighbor correlation functions.
            out_channels: node feature space output dimensionality of the GNN.
            hidden_channels: number of hidden channels in the message-passing mechanism.
            deg_histogram: degree of destination nodes in the graph. Used for normalization.
            num_layers: the number of GNN layers, i.e the number of message-passing procedures.
            hidden_nodes: enhancement factor of the node embedding space.
            hidden_edges: enhancement factor of the edge embedding space.

        Returns:
            A tensor representing the targets either at the node level (transverse fields) and at the edge level (lattice hoppings).
    """

    def __init__(self, node_in_channels, edge_in_channels, out_channels, hidden_channels, deg_histogram, *, 
                 num_layers = 3, hidden_edges = 2, hidden_nodes = 4):
        super(NodeEdgePNA,self).__init__()

        # Initial edge and node feature transformation
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_channels, hidden_edges*edge_in_channels), 
            nn.ReLU(), 
            nn.Linear(hidden_edges*edge_in_channels, edge_in_channels)
            )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_channels, hidden_nodes*node_in_channels), 
            nn.ReLU(), 
            nn.Linear(hidden_nodes*node_in_channels, node_in_channels)
            )
        
        self.PNA = PNA(
            in_channels = node_in_channels, # or -1 to derive the size from the first input(s) to the forward method
            edge_dim = edge_in_channels,
            out_channels = out_channels,
            hidden_channels = hidden_channels,
            num_layers = num_layers,
            # Normalization layer (similar to batch norm), good default:
            norm = GraphNorm(in_channels = hidden_channels),
            # This is something akin to a skip connection, good default to turn this on:
            jk = "cat",
            # PNA supports a few ways of aggregating/scaling messages, we just use all of them:
            aggregators = ["sum", "mean", "min", "max", "var", "std"],
            scalers = ["identity", "amplification", "attenuation", "linear", "inverse_linear"],
            # Histogram of node degrees, used for normalization. Mostly matters if node degrees vary a lot,
            # which is not the case. We consider the 'in-node' degree.
            deg = deg_histogram,
            dropout=0.0 # could be useful during training
            )
        # Output layers for node and edge predictions
        
        self.edge_pred_mlp = nn.Sequential(
            nn.Linear(edge_in_channels + 2 * out_channels, hidden_edges*out_channels), 
            nn.ReLU(),
            nn.Linear(hidden_edges*out_channels, hidden_edges*out_channels),
            nn.ReLU(),
            nn.Linear(hidden_edges*out_channels, 1)
            )
        self.double()

    def forward(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        # dim x = [batch_size*#nodes,node_feat_dim]
        # dim edge_index = [2,batch_size*#edges]
        # dim edge_attr = [batch_size*#edges,edge_feat_dim]

        # print(f"Inital values x.shape = {x.shape}, edge_index.shape = {edge_index.shape} and edge_attr.shape = {edge_attr.shape}")
        # Transform initial edge and node features
        edge_attr = self.edge_mlp(edge_attr) # [batch_size*#edges,edge_feat_dim]
        x = self.node_mlp(x) # [batch_size*#nodes,node_feat_dim]

        x = self.PNA(x, edge_index, edge_attr=edge_attr) # [#batch_size*#nodes, out_channels]
        
        x = x.tanh()
        # print(f"x.shape = {x.shape}")

        # Prepare for edge-level prediction
        # For each edge, concatenate the features of the connected nodes and the edge itself
        row, col = edge_index
        edge_feat = torch.cat([x[row], x[col], edge_attr], dim=-1) # cat [(#batches*#edges,out_channels),(#batches*#edges,out_channels),(#batches*#edges,edge_feat_dim)]

        # Edge-level prediction
        out_edges = self.edge_pred_mlp(edge_feat)

        return out_edges