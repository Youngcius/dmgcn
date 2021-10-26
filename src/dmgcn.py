import torch
import dgl
from torch import nn
import layers


class DMGCN(nn.Module):
    """
    Deep Molecular Graph Convolutional Network
    """

    def __init__(self, dim_node=128, dim_edge=128, dim_node_dict=20, dim_edge_dict=400, cutoff_low=0, cutoff_high=30,
                 n_centers=300, n_conv=3, norm=False, dim_node_app=0, dim_edge_app=0, dist=True, angle=True, multi_read=False,
                 embed_init=None, conv_init='normal', read_init='uniform'):
        """
        :param dim_node: dimension of node types after embedding
        :param dim_edge: dimension of edge types after embedding
        :param dim_node_dict: dimension of maximum node types before embedding
        :param dim_edge_dict: dimension of maximum edge types before embedding
        :param cutoff_low: inf-cutoff for RBF embedding of distance parameters
        :param cutoff_high: sup-cutoff for RBF embedding of distance parameters
        :param n_centers: number of centers for RBF embedding
        :param n_conv: number of convolutional layers
        :param norm: whether use norm and anti-norm techniques
        :param dim_node_app: dimension of appended node features, g.ndata['feat']
        :param dim_edge_app: dimension of appended edge features, g.edata['feat']
        :param dist: whether include the distance features for edges
        :param multi_read: whether readout results of each convolutional layer
        """
        super(DMGCN, self).__init__()
        self.name = 'EnhancedDTNN'
        self.n_conv = n_conv
        self.norm = norm
        self.multi_read = multi_read
        self.dist = dist
        self.angle = angle
        self.node_embedding = layers.NodeEmbedding(dim_dict=dim_node_dict, dim_embedded=dim_node,
                                                   dim_node_app=dim_node_app)
        self.edge_embedding = layers.EdgeEmbedding(dim_dict=dim_edge_dict, dim_embedded=dim_edge,
                                                   dim_edge_app=dim_edge_app)
        if self.dist:  # 是否有距离参数可供嵌入
            self.rbf_embedding = layers.RBFLayer(cutoff_low, cutoff_high, n_centers)
        else:
            n_centers = 0

        dim_node = dim_node + dim_node_app  # total dim of nodes
        dim_edge = dim_edge + dim_edge_app + n_centers  # total dim of edges
        # print('distance:', self.dist)
        # print(dim_node, dim_edge)

        self.conv_layers = nn.ModuleList([layers.DMGCNLayer(dim_node, dim_edge) for i in range(n_conv)])
        # Readout: 对于每个节点特征进行线性变换（双层FC网络）
        if self.multi_read:
            self.read_layer = layers.ReadLayer(dim_in=dim_node * self.n_conv, dim_hidden=dim_node)
        else:
            self.read_layer = layers.ReadLayer(dim_in=dim_node, dim_hidden=dim_node)

        self.reset_embedding_parameter(embed_init)  # 嵌入层Xavier初始化效果并不太好
        self.reset_conv_parameter(conv_init)
        self.reset_readlayer_parameter(read_init)

    def forward(self, g: dgl.DGLGraph):
        """
        :param g: DGLGraph instance
                    node features: ndata['Z']; optional: ndata['feat']
                    edge features: edata['type]; optional: edata['feat'], edata['dist']
        """
        self.node_embedding(g, name='h')  # 类型嵌入并与电荷特征做 concatenate, g.ndata['h'] size: [N, 128+]
        self.edge_embedding(g, name='h')  # 类型嵌入, 'h' g.edage['h'] size: [M, 128+]
        if self.dist:  # 是否有距离参数可供嵌入
            self.rbf_embedding(g)  # 距离的RBF嵌入, g.edata['rbf'] size: [M, K]
            g.edata['h'] = torch.cat((g.edata['h'], g.edata['rbf']), 1)  # g.edata['feat'] size: [M, 128+K]
        for i in range(self.n_conv):  # 多层结果都用于读出时
            g.ndata['h{}'.format(i)] = self.conv_layers[i](g)
        if self.multi_read:
            g.ndata['h'] = torch.cat([g.ndata['h{}'.format(i)] for i in range(self.n_conv)], dim=1)

        self.read_layer(g)  # 最终得到各个节点标量特征 ‘E’/'h'
        res = dgl.sum_nodes(g, 'h').flatten()  # [batch size,]
        if self.norm:
            # 反归一化
            res = res * self.std + self.mean
        return res

    def reset_conv_parameter(self, conv_init=None):
        """
        初始化卷积层权重参数，全部 ~ Normal Xavier 随机初始化
        """

        if conv_init == 'normal':
            print('using Xavier Normal initialization for Conv layer')
            for i in range(self.n_conv):
                nn.init.xavier_normal_(self.conv_layers[i].fc_node_1.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_normal_(self.conv_layers[i].fc_edge_1.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_normal_(self.conv_layers[i].fc_node_2.weight)
                nn.init.xavier_normal_(self.conv_layers[i].fc_edge_2.weight)
                nn.init.xavier_normal_(self.conv_layers[i].fc_combine.weight, gain=nn.init.calculate_gain('tanh'))
        elif conv_init == 'uniform':
            print('using Xavier Uniform initialization for Conv layer')
            for i in range(self.n_conv):
                nn.init.xavier_uniform_(self.conv_layers[i].fc_node_1.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.conv_layers[i].fc_edge_1.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.conv_layers[i].fc_node_2.weight)
                nn.init.xavier_uniform_(self.conv_layers[i].fc_edge_2.weight)
                nn.init.xavier_uniform_(self.conv_layers[i].fc_combine.weight, gain=nn.init.calculate_gain('tanh'))
        else:
            pass

    def reset_embedding_parameter(self, embed_init=None):
        """
        初始化嵌入层参数
        """
        if embed_init == 'normal':
            print('using Xavier Normal initialization for Embed layer')
            nn.init.xavier_normal_(self.node_embedding.embedding.weight)
            nn.init.xavier_normal_(self.edge_embedding.embedding.weight)
        elif embed_init == 'uniform':
            print('using Xavier Uniform initialization for Embed layer')
            nn.init.xavier_uniform_(self.node_embedding.embedding.weight)
            nn.init.xavier_uniform_(self.edge_embedding.embedding.weight)
        else:
            pass

    def reset_readlayer_parameter(self, read_init=None):
        # nn.init.xavier_normal_(self.read_layer.fc1.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.read_layer.fc2.weight)
        # 发现uniform更好，但是normal更稳定
        if read_init == 'normal':
            print('using Xavier Normal initialization for Read layer')
            nn.init.xavier_normal_(self.read_layer.fc1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.read_layer.fc2.weight)
        elif read_init == 'uniform':
            print('using Xavier Uniform initialization for Read layer')
            nn.init.xavier_uniform_(self.read_layer.fc1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.read_layer.fc2.weight)
        else:
            pass

    def set_mean_std(self, mean: float, std: float, device):
        """
        for norm and anti-norm strategy
        """
        self.mean = torch.tensor(mean, device=device)
        self.std = torch.tensor(std, device=device)
