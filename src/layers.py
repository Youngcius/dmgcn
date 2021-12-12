import torch
from torch import nn
import dgl


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
    Radial basis functions embedding layer.
    ---
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


# TODO:MD17

# class SBFLayer(nn.Module):
#     """
#     Sphere Bessel basis functions embedding layer.
#     ---
#
#     """
#
#     def __init__(self, n_centers=50, dim=1):
#         super(SBFLayer, self).__init__()
#         self.
#         pass
#
#     def dis2rbf(self, edges, duoyu=None):
#         # dist = edges.data["dist"]
#         # radial = dist - self.centers
#         # coef = -1 / self._gap
#         # sigma = self.gap / 2.3548
#         # coef = -1 / 2 / sigma ** 2
#         coef = -1 / self.gap
#         rbf = torch.exp(coef * ((edges.data['dist'].view(-1, 1) - self.centers.view(1, -1)) ** 2))
#         return {"rbf": rbf}
#
#     def ang2sbf(self, angles):
#         # 从g.edata['dist']数据出发计算angle
#         coef = ...
#
#     def forward(self, g):
#         # g.ndata['loc']: [N, 3]
#
#         g.edata['dist']
#         angles =
#
#         return angles


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


class TopoConvLayer(nn.Module):
    """
    Topological convolutional layer
    """

    def __init__(self, dim_node, dim_edge, n_centers_dist, dim_hidden=128, norm=True):
        super(TopoConvLayer, self).__init__()
        self.norm = norm
        self.fc_edge = nn.Linear(dim_edge, dim_hidden)
        self.fc_node = nn.Linear(dim_node, dim_hidden)
        self.fc_dist = nn.Linear(n_centers_dist, dim_hidden)
        self.fc_combine = nn.Linear(dim_hidden * 2, dim_node, bias=False)

    def foward(self, g: dgl.DGLGraph):
        """
        Message passing and node feature updating
        :param g: DGLGraph instance
        """
        g.update_all(self.msg_func, self.reduce_func)
        return g.ndata['h']

    def msg_func(self, edges):
        """
        :param edges: a batch of edges
        :return: message {'m': m}
        """
        # M 一批边的个数
        m1 = self.fc_node(edges.src['h']) * self.fc_dist(edges.data['dist'])
        m2 = self.fc_edge(edges.data['h'])
        m = torch.cat([m1, m2], dim=1)
        m = torch.tanh(self.fc_combine(m))

        edges.data['m'] = m  # dynamical saving messages on edges
        return {'m': m}

    def reduce_func(self, nodes):
        """
        :param nodes: nodes batch with the same in-degree
        :param norm: normalizing the final node features
        :return: nodes features, {'h': h}
        """
        # size of nodes.mailbox['m']: [b, indegree, m.size(1)]
        # b 是该图中入度为 indegree 节点的个数
        h = nodes.mailbox['m'].sum(dim=1)
        if self.norm:
            return {'h': h / h.norm(dim=1)}
        else:
            return {'h': h}


class SpatConvLayer(nn.Module):
    """
    Spatial convolutional layer
    """

    def __init__(self, dim_node, dim_edge, n_centers_dist, n_centers_angle, dim_hidden=128, norm=True):
        super(SpatConvLayer, self).__init__()
        self.norm = norm
        self.fc_msg = nn.Linear(dim_node, dim_hidden)  # dim_msg == dim_node
        self.fc_edge = nn.Linear(dim_edge, dim_hidden)
        self.fc_dist = nn.Linear(n_centers_dist, dim_hidden)
        self.fc_angle = nn.Linear(n_centers_angle, dim_hidden)
        self.fc_combine = nn.Linear(dim_hidden * 2, dim_node, bias=False)

        # angles embedding layer: RBF
        self.centers = nn.Parameter(torch.linspace(-1, 1, n_centers_angle), requires_grad=False)
        self.angles_rbf = lambda cos_a: torch.exp(-((cos_a.view(-1, 1) - self.centers.view(1, -1)) ** 2) / 2)

    def forward(self, g: dgl.DGLGraph):
        """
        Message passing and node feature updating
        :param g: DGLGraph instance
        """
        self.g_nx = dgl.to_networkx(g)
        g.update_all(self.msg_func, self.reduce_func)
        return g.ndata['h']

    def fc_msg_interact(self, m, d, a, e):
        """
        Interactive layer to realize directed message passing
        :param m: (topological) convolutional message, [-1 ,dim_node]
        :param d: distance of edges, [-1, n_centers_dist]
        :param a: angles, [-1, n_centers_angle]
        :param e: edges features, [-1, dim_edge]
        :return: new message data, [-1, dim_node]
        """
        return torch.tanh(
            self.fc_combine(
                torch.cat([
                    self.fc_msg(m) * self.fc_dist(d) * self.fc_angle(a),
                    self.fc_edge(e)
                ], dim=1)
            )
        )

    def msg_func(self, edges):
        """
        :param edges: a batch of edges
        :return: message {'m': m}
        """

        m = edges.data['m']

        # TODO: 列表序列化加速
        # for v, u in self.g_nx.edges():
        idx_edges = list(self.g_nx.edges())
        for i, (v, u) in enumerate(idx_edges):
            # directed edge: v --> u
            pre_v = self.g_nx.predecessors(v)
            if u in pre_v:
                pre_v.remove(u)
            if len(pre_v) != 0:
                # integrate angle information
                vu = idx_edges.index((v, u))
                wv = [idx_edges.index((w, v)) for w in pre_v]
                # edges.data['vec'][wv] : [-1, 3]
                print('edges.data[vec][wv]', edges.data['vec'][wv].shape)
                cos_alpha = torch.sum(edges.data['vec'][wv] * edges.data['vec'][vu], dim=1)  # size: [num_w,]
                print('size of cos theta', cos_alpha)
                m_tmp = self.fc_msg_interact(m[wv], edges.data['dist'][wv], self.angles_rbf(cos_alpha),
                                             edges.data['h'][wv]).sum(0)
                m[i] = 0.8 * m[i] + (1 - 0.8) * m_tmp

        edges.data['m'] = m  # dynamical saving messages on edges
        return {'m': m}

    def reduce_func(self, nodes):
        """
        :param nodes: nodes batch with the same in-degree
        :param norm: normalizing the final node features
        :return: nodes features, {'h': h}
        """
        h = nodes.mailbox['m'].sum(dim=1)
        if self.norm:
            return {'h': h / h.norm(dim=1)}
        else:
            return {'h': h}


class DMGCNLayer(nn.Module):
    """
    Joint convolutional layer, including topo-conv layer and spat-conv layer
    """

    def __init__(self, dim_node, dim_edge, n_centers_dist, n_centers_angle, norm=True):
        super(DMGCNLayer, self).__init__()
        self.topo_conv = TopoConvLayer(dim_node, dim_edge, n_centers_dist, norm=norm)
        self.spat_conv = SpatConvLayer(dim_node, dim_edge, n_centers_dist, n_centers_angle, norm=norm)

    def forward(self, g: dgl.DGLGraph):
        """
        Message passing and node feature updating
        :param g: DGLGraph instance
        """
        # 1) topological convolution
        g.apply_edges(self.update_edges)
        self.topo_conv(g)

        # 2) spatial convolution
        g.apply_edges(self.update_edges)
        self.spat_conv(g)

        return g.ndata['h']

    def update_edges(self, edges):
        h = 0.8 * edges.data['h'] + (1 - 0.8) * self.fc_update_edge(edges.src['h'] * edges.dst['h'])
        return {'h': h}
#
# class DMGCNLayer(nn.Module):
#     """
#     Convolutional layer of DMGCN model
#     """
#
#     def __init__(self, dim_node, dim_edge, n_centers_dist,n_centers_angle,norm=None):
#         super(DMGCNLayer, self).__init__()
#         self.fc_node_1 = nn.Linear(dim_node, 128)
#         self.fc_node_2 = nn.Linear(128, 128, bias=False)
#         self.fc_edge_1 = nn.Linear(dim_edge, 128)
#         self.fc_edge_2 = nn.Linear(128, 128, bias=False)
#
#         self.fc_combine = nn.Linear(128, dim_node, bias=False)  # option 1: element-wise multiplication
#         # self.fc_combine = nn.Linear(128 * 2, dim_node, bias=False)  # option 2: tensor concatenate
#
#         self.fc_update_edge = nn.Linear(dim_node, dim_edge, bias=False)  # option 1: element-wise multiplication
#         # self.fc_update_edge = nn.Linear(dim_node * 2, dim_edge, bias=False)  # option 2: tensor concatenate
#
#     def forward(self, g: dgl.DGLGraph):
#         """
#         Message passing and node feature updating
#         :param g: DGLGraph instance
#         """
#         # g.ndata['h'] : [N, dim_node]
#         # 1) message passing & reducing
#         # g.update_all(self.msg_func, self.reduce_func)
#         # 2) edges updating (enhanced)
#         # g.apply_edges(self.update_edges)
#
#         # 2021-10-14 修改：先更新边再更新顶点
#         g.apply_edges(self.update_edges)
#         g.update_all(self.msg_func, self.reduce_func)
#         return g.ndata['h']
#
#     def update_edges(self, edges):
#         # option 1: element-wise multiplication
#         h = 0.8 * edges.data['h'] + (1 - 0.8) * self.fc_update_edge(edges.src['h'] * edges.dst['h'])
#         # option 2: tensor concatenate
#         # h = 0.8 * edges.data['h'] + (1 - 0.8) * self.fc_update_edge(torch.cat([edges.src['h'], edges.dst['h']], dim=1))
#         return {'h': h}
#
#     def msg_func(self, edges):
#         # print('edges.src[h]', edges.src['h'].size())
#         # print('edges.data[feat]',edges.data['feat'].size())
#
#         # M 一批边的个数
#         # option 1: element-wise multiplication
#         m1 = self.fc_node_2(F.relu(self.fc_node_1(edges.src['h'])))  # [M,dim_node] --> [M,128] TODO 似乎应该取消这种对称性
#         m2 = self.fc_edge_2(F.relu(self.fc_edge_1(edges.data['h'])))  # [M,dim_edge] --> [M,128]
#         m = torch.tanh(self.fc_combine(m1 * m2))  # [M,128]
#
#         # option 2: tensor concatenate
#         # m1 = F.relu(self.fc_node_1(edges.src['h']))  # [M,dim_node] --> [M,128]
#         # m2 = F.relu(self.fc_edge_1(edges.data['h']))
#         # m = torch.tanh(self.fc_combine(torch.cat([m1, m2], dim=1)))
#
#         # print('m1',m1.size())
#         # print('m2',m2.size())
#         # m = torch.tanh(self.fc_combine(m1 * m2))
#
#         # print('m',m.size())
#         return {'m': m}
#
#     def reduce_func(self, nodes):
#         # size of nodes.mailbox['m']: [b, indegree, m.size(1)]
#         # b 是该图中入度为 indegree 节点的个数
#         return {'h': nodes.mailbox['m'].sum(dim=1) + nodes.data['h']}


# 存储mailbox：根据node indices得到链接的边的indx
#
# class DMGCNLayer(nn.Module):
#     """
#     Convolutional layer of DMGCN model
#     """
#
#     def __init__(self, dim_node, dim_edge, norm=None):
#         super(DMGCNLayer, self).__init__()
#         self.fc_node_1 = nn.Linear(dim_node, 128)
#         self.fc_node_2 = nn.Linear(128, 128, bias=False)
#         self.fc_edge_1 = nn.Linear(dim_edge, 128)
#         self.fc_edge_2 = nn.Linear(128, 128, bias=False)
#
#         self.fc_combine = nn.Linear(128, dim_node, bias=False)  # option 1: element-wise multiplication
#         # self.fc_combine = nn.Linear(128 * 2, dim_node, bias=False)  # option 2: tensor concatenate
#
#         self.fc_update_edge = nn.Linear(dim_node, dim_edge, bias=False)  # option 1: element-wise multiplication
#         # self.fc_update_edge = nn.Linear(dim_node * 2, dim_edge, bias=False)  # option 2: tensor concatenate
#
#     def forward(self, g: dgl.DGLGraph):
#         """
#         :param g: DGLGraph instance
#         """
#         # g.ndata['h'] : [N, dim_node]
#         # 1) message passing & reducing
#         # g.update_all(self.msg_func, self.reduce_func)
#         # 2) edges updating (enhanced)
#         # g.apply_edges(self.update_edges)
#
#         # 2021-10-14 修改：先更新边再更新顶点
#         g.apply_edges(self.update_edges)
#         g.update_all(self.msg_func, self.reduce_func)
#         return g.ndata['h']
#
#     def update_edges(self, edges):
#         # option 1: element-wise multiplication
#         h = 0.8 * edges.data['h'] + (1 - 0.8) * self.fc_update_edge(edges.src['h'] * edges.dst['h'])
#         # option 2: tensor concatenate
#         # h = 0.8 * edges.data['h'] + (1 - 0.8) * self.fc_update_edge(torch.cat([edges.src['h'], edges.dst['h']], dim=1))
#         return {'h': h}
#
#     def msg_func(self, edges):
#         # print('edges.src[h]', edges.src['h'].size())
#         # print('edges.data[feat]',edges.data['feat'].size())
#
#         # M 一批边的个数
#         # option 1: element-wise multiplication
#         m1 = self.fc_node_2(F.relu(self.fc_node_1(edges.src['h'])))  # [M,dim_node] --> [M,128]
#         m2 = self.fc_edge_2(F.relu(self.fc_edge_1(edges.data['h'])))  # [M,dim_edge] --> [M,128]
#         m = torch.tanh(self.fc_combine(m1 * m2))  # [M,128]
#
#         # option 2: tensor concatenate
#         # m1 = F.relu(self.fc_node_1(edges.src['h']))  # [M,dim_node] --> [M,128]
#         # m2 = F.relu(self.fc_edge_1(edges.data['h']))
#         # m = torch.tanh(self.fc_combine(torch.cat([m1, m2], dim=1)))
#
#         # print('m1',m1.size())
#         # print('m2',m2.size())
#         # m = torch.tanh(self.fc_combine(m1 * m2))
#
#         # print('m',m.size())
#         return {'m': m}
#
#     def reduce_func(self, nodes):
#         # size of nodes.mailbox['m']: [N, ]
#         return {'h': nodes.mailbox['m'].sum(dim=1) + nodes.data['h']}

# # TODO:
# 1. element-wise multification --> tensor concatenate
# 2. factors for messages
# 3. attention
