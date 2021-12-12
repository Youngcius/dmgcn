import re
import os
import shutil
import tempfile
import torch
import numpy as np
import scipy.stats as sps
import dgl
from dgl.data import DGLDataset
from ase.io.extxyz import read_xyz
from ase.units import Debye, Bohr, Hartree
from rdkit import Chem
from dgl.data.utils import save_info, load_info


class QM9Dataset(DGLDataset):
    """
    Our built-in QM9Dataset class extending from the DGLDataset class
    """
    # 原始文本第二行为"scalar"类型的属性，共17个，后15个有意义
    # 能量参数的单位都是 Hartree
    # ----------------------------
    # 这 3 个属性不用做训练和预测
    A = "rotational_constant_A"  # 转动频率：GHz
    B = "rotational_constant_B"
    C = "rotational_constant_C"
    # ----------------------------
    # 这是用于训练和预测的 12 个重要属性
    mu = "dipole_moment"
    alpha = "isotropic_polarizability"  # 各向同性极化率
    homo = "homo"  # 最高占据分子轨道
    lumo = "lumo"  # 最低未占据分子轨道（lumo > homo）
    gap = "gap"  # lumo - homo
    r2 = "electronic_spatial_extent"  # 电子空间距离度量: a0^2
    zpve = "zpve"  # 零点振动能
    U0 = "energy_U0"  # OK 内能
    U = "energy_U"  # RT 内能
    H = "enthalpy_H"  # RT 焓
    G = "free_energy"  # RT 吉布斯函数
    Cv = "heat_capacity"  # RT 热容

    reference = {zpve: 0, U0: 1, U: 2, H: 3, G: 4, Cv: 5}
    prop_names = [A, B, C, mu, alpha, homo,
                  lumo, gap, r2, zpve, U0, U, H, G, Cv]  # 15 labels
    units = [1.0, 1.0, 1.0, Debye, Bohr ** 3, Hartree, Hartree, Hartree,
             Bohr ** 2, Hartree, Hartree, Hartree, Hartree, Hartree, 1.0]
    energy_indices = np.array(units) == Hartree  # energy unit: Hartree --> eV

    def __init__(self, raw_dir='../data/', type='non'):
        """
        :param raw_dir: directory where the QM9 dataset file is located
        :param type: 'com' means 'complete', 'non' means 'non-complete'
        """
        # super(QM9Dataset, self).__init__(name='qm9', url=url, raw_dir=raw_dir,
        #  save_dir=save_dir, fore_reload=fore_reload, verbose=verbose)
        if type not in ['com', 'non']:
            raise TypeError('{} is not in the supported type set [complete, non-complete]'.format(type))
        self.type = type
        self.dgl_graph_fname = 'dgl_graph_{}.bin'.format(type)
        self.info_dict_fname = 'info.pkl'
        super(QM9Dataset, self).__init__(name='qm9', raw_dir=raw_dir)  # 至此 self.save_path 为 '../data/qm9'

    def process(self):
        """
        process raw data to graphs, labels, splitting masks
        当没有缓存dgl_graph.bin和info.bin的时候执行这个函数
        :return: self.graphs, self.labels
        """
        # 假定原始数据已经存在self.raw_dir
        # 该函数在类的实例化时自动执行
        # if self.has_cache():
        #     self.load()
        # else:
        self.dict = self._load_data()
        self.graphs, self.labels = self._load_graphs()

    def _load_data(self):
        """
        从 data目录下的dsgdb9nsd.xyz文件夹中读取数据
        :return: self.dict
        """
        Smiles = []
        InChI = []
        num_atoms = []  # 每个分子包含的原子个数
        properties = []  # 暂时用列表（数组）存储属性值，每个元素为长度15的列表
        charges = []  # element length： n
        harmonics = []  # 自由度：3n-6
        all_atoms = []  # 每个元素是一个 atoms object（ASE 库所支持）
        data_path = os.path.join(self.raw_dir, 'dsgdb9nsd.xyz')
        fnames = [os.path.join(data_path, fname)
                  for fname in os.listdir(data_path)]
        fnames = sorted(fnames, key=lambda x: (int(re.sub("\D", "", x)), x))
        # fnames = fnames[:1000]
        tmpdir = tempfile.mkdtemp("gdb9")  # 临时目录为利用 read_xyz 读取xyz文件
        N = len(fnames)  # 133885
        for i, fname in enumerate(fnames):
            if (i + 1) % 10000 == 0:
                print("Parsed: {:6d} / {}".format(i + 1, N))

            tmp = os.path.join(tmpdir, "tmp.xyz")
            with open(fname, "r") as f:
                lines = f.readlines()
                lines = [line.replace("*^", "e") for line in lines]
                # atom number in every molecule
                num_atoms.append(int(lines[0]))
                harmonics.append(list(map(float, lines[-3].split())))
                Smiles.append(lines[-2].split())
                InChI.append(lines[-1].split())
                # l = lines[1].split()[2:]
                prop_values = np.array(list(map(float, lines[1].split()[2:])))
                prop_values[QM9Dataset.energy_indices] *= Hartree  # convert unit to 'eV'
                # prop_values[[5,6,7] + [9,10,11,12,13]]
                properties.append(prop_values.tolist())  # 后15个属性值
                # 坐标&电荷数据
                coord = np.array([])
                charge = []
                for i in range(2, 2 + int(lines[0])):
                    charge.append(float(lines[i].split()[-1]))
                    # coord = np.append(coord, list(map(float, l[1:-1])))
                charges.append(charge)
                # corrdinates.append(coord)
                with open(tmp, "wt") as fout:
                    for line in lines:
                        # fout.write(line.replace("*^", "e"))
                        fout.write(line)

            with open(tmp, "r") as f:
                ats = list(read_xyz(f, 0))[0]  # atoms object (ASE)
            all_atoms.append(ats)

        shutil.rmtree(tmpdir)  # 删除 gdb9 临时目录
        print('parsing 已经完成!')
        return {'fnames': fnames, 'Smiles': Smiles, 'InChI': InChI, 'properties': properties,
                'charges': charges, 'all_atoms': all_atoms, 'harmonics': harmonics, 'num_atoms': num_atoms}

    def _load_graphs(self):
        labels = torch.tensor(self.dict['properties']).float()  # size: [n, 15]
        # 暂时不需要 self.dict['harmonics'][i] 作为标签
        graphs = []
        num_graphs = len(labels)

        for i in range(num_graphs):
            # f.write(str(i) + '\n')
            if (i + 1) % 10000 == 0:
                print('Graphs created: {:6d} / 133885'.format(i + 1))
            ats = self.dict['all_atoms'][i]
            # mol = Chem.AddHs(Chem.MolFromInchi(self.dict['InChI'][i][0]))
            mol = Chem.AddHs(Chem.MolFromSmiles(self.dict['Smiles'][i][0]))

            bonds_info = []
            bonds_info += [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
            bonds_info += [(bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()) for bond in mol.GetBonds()]
            g = dgl.graph(bonds_info)  # 不完全无向图

            # 构造节点初始特征
            # g.ndata['loc'] = torch.tensor(ats.positions).float()  # positions
            pos = torch.tensor(ats.positions).float()
            g.ndata['feat'] = torch.tensor(self.dict['charges'][i]).float()  # charges
            g.ndata['Z'] = torch.tensor(ats.numbers).long()  # Z, long means int64
            # 构造 边 初始特征
            g.edata['dist'] = torch.empty(g.num_edges())  # bonds distances
            g.edata['type'] = torch.empty(g.num_edges()).long()  # bonds types, long means int64
            for i, (u, v) in enumerate(zip(g.edges()[0], g.edges()[1])):
                # (u, v) 的特征,即第 i 条边, size: torch.Size([])
                Z_u, Z_v = g.ndata['Z'][u].item(), g.ndata['Z'][v].item()
                # g.edata['dist'][i] = torch.norm(g.ndata['X'][u] - g.ndata['X'][v])
                g.edata['dist'][i] = np.linalg.norm(ats.positions[u] - ats.positions[v])  # 完全图时，当 u=v 时，dist=0
                g.edata['type'][i] = Z_u * Z_v + (np.abs(Z_u - Z_v) - 1) ** 2 / 4  # 自动转换为long类型
                vec = pos[v] - pos[v]
                g.edata['vec'][i] = vec / torch.norm(vec)

            graphs.append(g)

        return graphs, labels

    def get_prop_stat(self) -> dict:
        """
        Returns:
            nobs : int or ndarray of ints
               Number of observations (length of data along `axis`).
               When 'omit' is chosen as nan_policy, each column is counted separately.
            minmax: tuple of ndarrays or floats
               Minimum and maximum value of data array.
            mean : ndarray or float
               Arithmetic mean of data along axis.
            variance : ndarray or float
               Unbiased variance of the data along axis, denominator is number of
               observations minus one.
            skewness : ndarray or float
               Skewness, based on moment calculations with denominator equal to
               the number of observations, i.e. no degrees of freedom correction.
            kurtosis : ndarray or float
               Kurtosis (Fisher).  The kurtosis is normalized so that it is
               zero for the normal distribution.  No degrees of freedom are used.
        """
        return sps.describe(self.dict['properties'])._asdict()

    def get_dist_min_max_with_idx(self, idx) -> tuple:
        dist = self.graphs[idx].edata['dist']
        return dist.min().item(), dist.max().item()

    def save(self):
        """
        default execuion: self.save_path = os.path.join(self.raw_dir, name)
        """
        # if self.has_cache():
        #     return
        # if not os.path.exists(self.save_path):
        #     os.mkdir(self.save_path)
        # 保存图和标签到 self.save_path
        graph_path = os.path.join(self.save_path, self.dgl_graph_fname)
        dgl.save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # 在Python字典里保存其他信息
        info_path = os.path.join(self.save_path, self.info_dict_fname)
        save_info(info_path, self.dict)
        print('....saved....')

    def load(self):
        """
        load data from dictionary `self.save_path`
        """
        graph_path = os.path.join(self.save_path, self.dgl_graph_fname)
        self.graphs, label_dict = dgl.load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = os.path.join(self.save_path, self.info_dict_fname)
        self.dict = load_info(info_path)
        print('....loaded....')

    def has_cache(self):
        # 检查在 `self.save_path` 里是否有处理过的数据文件
        graph_path = os.path.join(self.save_path, self.dgl_graph_fname)
        info_path = os.path.join(self.save_path, self.info_dict_fname)
        return os.path.exists(graph_path) and os.path.exists(info_path)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    @property
    def num_labels(self):
        return 15
