import networkx as nx
import torch
import numpy as np


### not sure if i should modify fil edges to fit with the removeing of 'time' after incorporating the edges (-1 for all edges)
def fil_edges(Mat):
    edges_index=[]
    X1=[]
    X2=[]
    n=len(Mat)
    for k in range(len(Mat)):
        for j in range(len(Mat[0])):
            if k+j+1<n:
                if Mat[j][j+k+1]>0:
                    X1.append(j)
                    X2.append(j+k+1)
    edges_index.append(X1)
    edges_index.append(X2)
    edges_index=torch.tensor(edges_index)
    #edges_index=edges_index.float()
    return edges_index




def get_edge_index(
    X: torch.Tensor | None = None, device: str = "cuda", path: str | None = None
) -> torch.Tensor:
    """
    Get the edge index of the graph.

    Args:
        X: The input data.
        device: The device to place the edge_index tensor on.
        path: The path to the edge_index file.

    Returns:
        The edge index of the graph.
    """
    if path:
        if path.endswith('.csv'):

            adj_25=np.genfromtxt(path, delimiter=',')
            edges_index=fil_edges(adj_25)
            fc_edge_index = torch.tensor(edges_index, dtype = torch.long)
            #torch.save(fc_edge_index,"adj.pt")
            print(f"Load edge index from matrix adjacency from path: {path}")
            return fc_edge_index.to(device)

        else:    
            try:
                edge_index = torch.load(path)
                if not isinstance(edge_index, torch.Tensor):
                    edge_index = torch.tensor(edge_index, dtype=torch.long)
                else:
                    edge_index = edge_index.long()

                if edge_index.dim() != 2 or edge_index.size(0) != 2:
                    raise ValueError(
                        f"Edge index must have shape [2, num_edges], got {edge_index.shape}"
                    )

                print(f"Loaded edge index from {path}")
                return edge_index.to(device)
            except FileNotFoundError:
                print(f"Edge index file not found at {path}")
            except Exception as e:
                print(f"Error loading edge index: {str(e)}")

    if X is None:
        raise ValueError("X must be provided if path is non-existent")

    print("Building fully connected edge index")
    return build_fully_connected_edge_index(X, device)


def build_fully_connected_edge_index(X: torch.Tensor, device: str) -> torch.Tensor:
    """
    Build a fully connected edge index for the graph.

    Args:
        X: The input data.
        device: The device to place the edge_index tensor on.

    Returns:
        The fully connected edge index of the graph.
    """
    edge_index = (
        torch.tensor(
            [[i, j] for i in range(X.shape[1]) for j in range(X.shape[1])],
            dtype=torch.long,  # edge_index must be long type
        )
        .t()
        .to(device)
    )

    return edge_index


def edge_index_to_networkx(
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor | None = None,
    directed: bool = False,
) -> nx.Graph | nx.DiGraph:
    """
    Convert a PyTorch Geometric edge_index to a NetworkX graph using torch operations.

    Args:
        edge_index : torch.Tensor
            Edge index tensor of shape [2, num_edges]
        edge_weights : torch.Tensor, optional
            Edge weights tensor of shape [num_edges]
        directed : bool, default=False
            If True, creates a directed graph

    Returns:
        networkx.Graph or networkx.DiGraph
            The converted NetworkX graph
    """
    G = nx.DiGraph() if directed else nx.Graph()

    edges = torch.stack([edge_index[0], edge_index[1]], dim=1).tolist()

    if edge_weights is not None:
        edges = [(i, j, {"weight": float(w)}) for (i, j), w in zip(edges, edge_weights)]

    G.add_edges_from(edges)
    return G


def networkx_to_edge_index(
    G: nx.Graph | nx.DiGraph, return_weights: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a NetworkX graph to a PyTorch Geometric edge_index using torch operations.

    Args:
        G : networkx.Graph or networkx.DiGraph
            Input NetworkX graph
        return_weights : bool, default=False
            If True, also returns edge weights

    Returns:
        Edge index tensor of shape [2, num_edges]
        If return_weights=True, also returns edge weights tensor
    """
    edges = list(G.edges())

    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        if return_weights:
            weights = torch.empty(0, dtype=torch.float)
            return edge_index, weights
        return edge_index

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    if return_weights:
        weights = torch.tensor(
            [G[i][j].get("weight", 1.0) for i, j in edges], dtype=torch.float
        )
        return edge_index, weights

    return edge_index


def build_random_edge_index(
    num_nodes: int, num_edges: int, device: str
) -> torch.Tensor:
    """
    Build a random edge index for the graph.

    Args:
        num_nodes: The number of nodes in the graph.
        num_edges: The number of edges in the graph.
    Returns:
        The random edge index of the graph.
    """
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    return edge_index
