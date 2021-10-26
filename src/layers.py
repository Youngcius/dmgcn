import torch
from torch import nn
import dgl
import torch.nn.functional as F


class NodeEmbedding(nn.Module):
    """
    The nodes with same atomic number share the same initial embedding
    """

    def __init__(self, dim_dict=20, dim_embedded=128, pre_train=None, dim_node_app=0):
        super(NodeEmbedding, self).__init__()
        if pre_train is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_train, padding_idx=0)
        else:
            self.embedding = nn.Embedding(dim_dict, dim_embedded, padding_idx=0)
        self.dim_node_app = dim_node_app

    def forward(self, g, name='h'):
        embedded = self.embedding(g.ndata['Z'])  # [num_nodes, 127]
        if self.dim_node_app == 0:
            g.ndata[name] = embedded
        else:
            if g.ndata['feat'].dim() == 1:
                g.ndata['feat'] = g.ndata['feat'].view(-1, 1)
            g.ndata[name] = torch.cat((embedded, g.ndata['feat']), dim=1)  # dim=(embedded.dim() - 1))


class EdgeEmbedding(nn.Module):
    """
    Convert the edge to embedding.
    The edge links same pair of atoms share the same initial embedding.
    """

    def __init__(self, dim_dict=400, dim_embedded=128, pre_train=None, dim_edge_app=0):
        super(EdgeEmbedding, self).__init__()
        # self._dim = dim
        # self._edge_num = edge_num
        if pre_train is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_train, padding_idx=0)
        else:
            # self.embedding = nn.Embedding(edge_num, dim, padding_idx=0)
            self.embedding = nn.Embedding(dim_dict, dim_embedded, padding_idx=0)
        self.dim_edge_app = dim_edge_app

    def forward(self, g, name="h"):
        embedded = self.embedding(g.edata["type"])  # 边嵌入
        if self.dim_edge_app == 0:
            g.edata[name] = embedded
        else:
            if g.edata['feat'].dim() == 1:
                g.edata['feat'] = g.edata['feat'].view(-1, 1)
            g.edata[name] = torch.cat((embedded, g.edata['feat']), dim=1)


class RBFLayer(nn.Module):
    """
    Radial basis functions Layer.
    e(d) = exp(- gamma * ||d - mu_k||^2)
    default settings:
        gamma = 10
        0 <= mu_k <= 30 for k=1~300
    """

    def __init__(self, low, high, n_centers=150, dim=1):
        super(RBFLayer, self).__init__()
        self.centers = nn.Parameter(torch.linspace(low, high, n_centers), requires_grad=False)
        self.gap = self.centers[1] - self.centers[0]  # could be regarded as FWHM
        # self._fan_out = self.dim * self.n_center
        self.fan_out = dim * n_centers

    def dis2rbf(self, edges, duoyu=None):
        # dist = edges.data["dist"]
        # radial = dist - self.centers
        # coef = -1 / self._gap
        # sigma = self.gap / 2.3548
        # coef = -1 / 2 / sigma ** 2
        coef = -1 / self.gap
        rbf = torch.exp(coef * ((edges.data['dist'].view(-1, 1) - self.centers.view(1, -1)) ** 2))
        return {"rbf": rbf}

    def forward(self, g):
        """Convert distance scalar to rbf vector"""
        g.apply_edges(self.dis2rbf)
        return g.edata["rbf"]


class ReadLayer(nn.Module):
    """
    Readout layer of DTNN model
    """

    def __init__(self, dim_in, dim_hidden):
        super(ReadLayer, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, 1)

    def forward(self, g: dgl.DGLGraph):
        g.apply_nodes(self.linear_trans)  # 不同图的节点数目不一样，单节点特征的维度一样
        # return dgl.sum_nodes(g, 'E')

    def linear_trans(self, nodes):
        # h1 = torch.tanh(self.fc1(nodes.data['h']))
        h1 = torch.relu(self.fc1(nodes.data['h']))  # ReLU效果更好, tanh难收敛，死难收敛
        h2 = self.fc2(h1)
        return {'h': h2}


class DMGCNLayer(nn.Module):
    """
    Convolutional layer of DMGCN model
    """

    def __init__(self, dim_node, dim_edge, norm=None):
        super(DMGCNLayer, self).__init__()
        self.fc_node_1 = nn.Linear(dim_node, 128)
        self.fc_node_2 = nn.Linear(128, 128, bias=False)
        self.fc_edge_1 = nn.Linear(dim_edge, 128)
        self.fc_edge_2 = nn.Linear(128, 128, bias=False)

        self.fc_combine = nn.Linear(128, dim_node, bias=False)  # option 1: element-wise multiplication
        # self.fc_combine = nn.Linear(128 * 2, dim_node, bias=False)  # option 2: tensor concatenate

        self.fc_update_edge = nn.Linear(dim_node, dim_edge, bias=False)  # option 1: element-wise multiplication
        # self.fc_update_edge = nn.Linear(dim_node * 2, dim_edge, bias=False)  # option 2: tensor concatenate

    def forward(self, g: dgl.DGLGraph):
        """
        :param g: DGLGraph instance
        """
        # g.ndata['h'] : [N, dim_node]
        # 1) message passing & reducing
        # g.update_all(self.msg_func, self.reduce_func)
        # 2) edges updating (enhanced)
        # g.apply_edges(self.update_edges)

        # 2021-10-14 修改：先更新边再更新顶点
        g.apply_edges(self.update_edges)
        g.update_all(self.msg_func, self.reduce_func)
        return g.ndata['h']

    def update_edges(self, edges):
        # option 1: element-wise multiplication
        h = 0.8 * edges.data['h'] + (1 - 0.8) * self.fc_update_edge(edges.src['h'] * edges.dst['h'])
        # option 2: tensor concatenate
        # h = 0.8 * edges.data['h'] + (1 - 0.8) * self.fc_update_edge(torch.cat([edges.src['h'], edges.dst['h']], dim=1))
        return {'h': h}

    def msg_func(self, edges):
        # print('edges.src[h]', edges.src['h'].size())
        # print('edges.data[feat]',edges.data['feat'].size())

        # M 一批边的个数
        # option 1: element-wise multiplication
        m1 = self.fc_node_2(F.relu(self.fc_node_1(edges.src['h'])))  # [M,dim_node] --> [M,128]
        m2 = self.fc_edge_2(F.relu(self.fc_edge_1(edges.data['h'])))  # [M,dim_edge] --> [M,128]
        m = torch.tanh(self.fc_combine(m1 * m2))  # [M,128]

        # option 2: tensor concatenate
        # m1 = F.relu(self.fc_node_1(edges.src['h']))  # [M,dim_node] --> [M,128]
        # m2 = F.relu(self.fc_edge_1(edges.data['h']))
        # m = torch.tanh(self.fc_combine(torch.cat([m1, m2], dim=1)))

        # print('m1',m1.size())
        # print('m2',m2.size())
        # m = torch.tanh(self.fc_combine(m1 * m2))

        # print('m',m.size())
        return {'m': m}

    def reduce_func(self, nodes):
        # size of nodes.mailbox['m']: [N, ]
        return {'h': nodes.mailbox['m'].sum(dim=1) + nodes.data['h']}

# # TODO:
# 1. element-wise multification --> tensor concatenate
# 2. factors for messages
# 3. attention
