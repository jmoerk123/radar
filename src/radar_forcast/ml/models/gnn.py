from typing import Any, Literal

import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn


class GNNModel(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        c_out: int,
        num_layers: int = 2,
        layer_name: Literal["GCN", "GAT", "GraphConv"] = "GCN",
        dp_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """GNNModel.

        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)

        """
        super().__init__()
        self._gnn_layer_by_name = {
            "GCN": geom_nn.GCNConv,
            "GAT": geom_nn.GATConv,
            "GraphConv": geom_nn.GraphConv,
        }
        gnn_layer = self._gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for _ in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)

        """
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x
