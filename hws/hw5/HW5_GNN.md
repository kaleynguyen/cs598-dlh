# HW5 MPNN

## Overview

In this question, we will try the Graph Neural Network (GNN).

---


```python
import os
import pickle
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
```


```python
# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
```


```python
DATA_PATH = "../HW5_GNN-lib/data/"
```

---

We will use [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html), which is a geometric deep learning extension library for PyTorch.

## 1 Graph [10 points]

First, let us learn the fundamental concepts of PyTorch Geometric.

A graph is used to model pairwise relations (edges) between objects (nodes).
A single graph in PyTorch Geometric is described by an instance of `torch_geometric.data.Data`, which holds the following attributes by default:

- `data.x`: Node feature matrix with shape `[num_nodes, num_node_features]`
- `data.edge_index`: Graph connectivity in COO format with shape `[2, num_edges]` and type `torch.long`
- `data.edge_attr`: Edge feature matrix with shape `[num_edges, num_edge_features]`
- `data.y`: Target to train against (may have arbitrary shape), *e.g.*, node-level targets of shape `[num_nodes, *]` or graph-level targets of shape `[1, *]`
- `data.pos`: Node position matrix with shape `[num_nodes, num_dimensions]`

Note that none of these attributes is required.

We show a simple example of an unweighted and undirected graph with three nodes and four edges. Each node contains exactly one feature:

<img width="500" src="img/graph-1.png">


```python
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)
print("num_nodes:", data.num_nodes)
print("num_edges:", data.num_edges)
print("num_node_features:", data.num_node_features)
```

Note that although the graph has only two edges, we need to define four index tuples to account for both directions of a edge.

Now, create a `torch_geometric.data.Data` instance for the following graph.

<img width="250" src="img/graph-2.png">

Assign the graph-level target $1$ to the graph.


```python
"""
TODO: create a `torch_geometric.data.Data` instance for the graph above. Set the graph-level target to 1.
"""

data = None

# your code here
raise NotImplementedError
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

assert data.num_nodes == 4
assert data.num_edges == 4
assert data.y == 1
assert data.num_node_features == 0
assert data.num_edge_features == 0


```

## 2 Dataset [20 points]

For this question, we will use the [MUTAG dataset](http://networkrepository.com/Mutag.php). Each graph in the dataset represents a chemical compound and graph labels represent their mutagenic effect on a specific gram negative bacterium. The dataset includes 188 graphs. Graph nodes have 7 labels and each graph is labelled as belonging to 1 of 2 classes.


```python
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root=DATA_PATH, name='MUTAG')
print("len:", len(dataset))
print("num_classes:", dataset.num_classes)
print("num_node_features:", dataset.num_node_features)
```

Let us take one graph as an example.


```python
data = dataset[0]
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of features: {data.num_node_features}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
```

We can see that the first graph in the dataset contains 17 nodes, each one having 7 features. There are 38/2 = 19 undirected edges and the graph is assigned to exactly one class. In addition, the data object is holding exactly one graph-level target.


```python
def graph_stat(dataset):
    """
    TODO: calculate the statistics of the ENZYMES dataset.
    
    Outputs:
        min_num_nodes: min number of nodes
        max_num_nodes: max number of nodes
        mean_num_nodes: average number of nodes
        min_num_edges: min number of edges
        max_num_edges: max number of edges
        mean_num_edges: average number of edges
    """
    
    # your code here
    raise NotImplementedError
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

assert np.allclose(graph_stat(dataset), (10, 28, 17.93, 20, 66, 39.58), atol=1e-2)


```

Neural networks are usually trained in a batch-wise fashion. PyTorch Geometric achieves parallelization over a mini-batch by creating sparse block diagonal adjacency matrices (defined by `edge_index`) and concatenating feature and target matrices in the node dimension. This composition allows differing number of nodes and edges over examples in one batch:

$\begin{split}\mathbf{A} = \begin{bmatrix} \mathbf{A}_1 & & \\ & \ddots & \\ & & \mathbf{A}_n \end{bmatrix}, \qquad \mathbf{X} = \begin{bmatrix} \mathbf{X}_1 \\ \vdots \\ \mathbf{X}_n \end{bmatrix}, \qquad \mathbf{Y} = \begin{bmatrix} \mathbf{Y}_1 \\ \vdots \\ \mathbf{Y}_n \end{bmatrix}\end{split}$

Luckily, PyTorch Geometric contains its own `torch_geometric.data.DataLoader`, which already takes care of this concatenation process. Let us learn about it in an example:


```python
from torch_geometric.data import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)

loader_iter = iter(loader)
batch = next(loader_iter)
print(batch)
print(batch.num_graphs)
```

That is, each batch contains $32$ graphs whose nodes and edges are stacked into one matrix.

Now, let us create a 80/20 train/test split, and load them into the dataloaders.


```python
# shuffle
dataset = dataset.shuffle()
# split
split_idx = int(len(dataset) * 0.8)
train_dataset = dataset[:split_idx]
test_dataset = dataset[split_idx:]

print("len train:", len(train_dataset))
print("len test:", len(test_dataset))
```


```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
```

## 3 Graph Neural Network [50 points]

After learning about the fundamental concepts in PyTorch Geometric, let us try implement our first graph neural network.

We will use a simple Graph Convolution Network (GCN) to assign each enzyme to one of the 6 EC top-level classes.

For a high-level explanation on GCN, have a look at its [blog](http://tkipf.github.io/graph-convolutional-networks/) post.

We will first implement a GCN layer. A GCN layer is given an adjacency matrix $A\in\mathbb{R}^{N\times N}$ and a node feature matrix $X\in\mathbb{R}^{N\times D_{in}}$, where $N$ is the number of nodes and $D_{in}$ is the input dimension. The graph convolution network will calculate its output by:

$$
X' = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}X\Theta
$$

where $\hat{A}=A+I$ denotes the adjacency matrix with self-loop, $\hat{D}_{ii}=\sum_{j=0}^{N-1}\hat{A}_{ij}$ is its diagonal degree matrix and $\Theta\in\mathbb{R}^{D_{in}\times D_{out}}$ is the model parameter.


```python
import math


class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        # Initialize the parameters.
        stdv = 1. / math.sqrt(out_channels)
        self.theta.data.uniform_(-stdv, stdv)
    
    def forward(self, x, edge_index):
        """
        TODO:
            1. Generate the adjacency matrix with self-loop \hat{A} using edge_index.
            2. Calculate the diagonal degree matrix \hat{D}.
            3. Calculate the output X' with torch.mm using the equation above.
        """
        # your code here
        raise NotImplementedError
        return ret
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''


```

<img width="500" src="img/graph-3.png">

The GCN will have the following steps:

- Embed each node by performing multiple rounds of message passing
- Aggregate node embeddings into a unified graph embedding (readout layer)
- Train a final classifier on the graph embedding


There exists multiple readout layers in literature, but the most common one is to simply take the average of node embeddings:

$$
\mathbf{x}_{\mathcal{G}} = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \mathcal{x}^{(L)}_v
$$

PyTorch Geometric provides this functionality via `torch_geometric.nn.global_mean_pool`.


```python
# from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        """
        TODO:
            1. Define the first convolution layer using `GCNConv()`. Set `out_channels` to 64;
            2. Define the first activation layer using `nn.ReLU()`;
            3. Define the second convolution layer using `GCNConv()`. Set `out_channels` to 64;
            4. Define the second activation layer using `nn.ReLU()`;
            5. Define the third convolution layer using `GCNConv()`. Set `out_channels` to 64;
            6. Define the dropout layer using `nn.Dropout()`;
            7. Define the linear layer using `nn.Linear()`. Set `output_size` to 2.

        Note that for MUTAG dataset, the number of node features is 7, and the number of classes is 2.

        """
        
        # your code here
        raise NotImplementedError

    def forward(self, x, edge_index, batch):
        """
        TODO:
            1. Pass the data through the frst convolution layer;
            2. Pass the data through the first activation layer;
            3. Pass the data through the second convolution layer;
            4. Pass the data through the second activation layer;
            5. Pass the data through the third convolution layer;
            6. Obtain the graph embeddings using the readout layer with `global_mean_pool()`;
            7. Pass the graph embeddgins through the dropout layer;
            8. Pass the graph embeddings through the linear layer.
            
        Arguments:
            x: [num_nodes, 7], node features
            edge_index: [2, num_edges], edges
            batch: [num_nodes], batch assignment vector which maps each node to its 
                   respective graph in the batch. 
                   Can be used in global_mean_pool

        Outputs:
            probs: probabilities of shape (batch_size, 2)
        """
        
        # your code here
        raise NotImplementedError
        
GCN()
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''


```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''


```

## 4 Training and Inferencing [20 points]

Let us train our network for a few epochs to see how well it performs on the training as well as test set.


```python
gcn = GCN()

# optimizer
optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01)
# loss
criterion = torch.nn.CrossEntropyLoss()

def train(train_loader):
    gcn.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        """
        TODO: train the model for one epoch.
        
        Note that you can acess the batch data using `data.x`, `data.edge_index`, `data.batch`, `data,y`.
        """
        
        # your code here
        raise NotImplementedError

def test(loader):
    gcn.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = gcn(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(200):
    train(train_loader)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch + 1:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

test_acc = test(test_loader)


```


```python

```


```python

```
