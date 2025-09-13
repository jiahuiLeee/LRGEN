import torch
import torch.nn as nn
from dgl.nn import RelGraphConv
import dgl

class RGCN(nn.Module):
    def __init__(self, in_size, hidden_size, out_feats, num_relations, num_bases, n_layers):
        """_summary_

        Args:
            in_feats (int): _description_
            hidden_size (int): _description_
            out_feats (int): _description_
            relation_mapping (_type_): _description_
            n_layers (int): _description_
        """
        super(RGCN, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(hidden_size)
            )
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                RelGraphConv(
                    in_feat=hidden_size, 
                    out_feat=hidden_size, 
                    num_rels=num_relations, 
                    regularizer='basis', 
                    num_bases=num_bases,
                    activation=torch.relu,
                    )
            )
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_size, out_feats),
            nn.ReLU(inplace=False),
            # nn.BatchNorm1d(out_feats),
            # nn.Dropout(0.1)
        )


    def forward(self, graph):
        graph = dgl.to_homogeneous(graph, ndata=['feat'])
        x = graph.ndata['feat']
        etypes = graph.edata[dgl.ETYPE]
        x = self.input_layer(x)
        for conv in self.convs:
            x = conv(graph, x, etypes)
        x = self.out_layer(x)
        return x