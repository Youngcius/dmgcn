import torch
import os
import datetime
import time
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dgl.data import split_dataset
import random
import argparse
import matplotlib.pyplot as plt
from process import train, test, train_and_evaluate
from qm9 import QM9Dataset
from utils import batcher, count_model_parameter
# from mgcn import MGCNModel
# from dtnn import EnhancedDTNN
from dmgcn import  DMGCN
from sklearn import metrics


def plot_train_eval_result(loss, mae, omit, title, fname):
    epochs = len(running_loss)
    epoch_steps = np.arange(1, epochs + 1)[omit:]
    running_loss_arr = np.array(loss)[omit:]
    running_mae_arr = np.array(mae)[omit:]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)  # loss
    plt.plot(epoch_steps, running_loss_arr[:, 0], label='train loss', color='k')
    plt.plot(epoch_steps, running_loss_arr[:, 1], label='evaluate loss', color='r')
    plt.legend()
    plt.xticks(epoch_steps)
    plt.title('Loss')
    plt.subplot(1, 2, 2)  # MAE
    plt.plot(epoch_steps, running_mae_arr[:, 0], label='train MAE', color='k')
    plt.plot(epoch_steps, running_mae_arr[:, 1], label='evaluate MAE', color='r')
    plt.legend()
    plt.xticks(epoch_steps)
    plt.title('MAE')
    plt.suptitle(title)
    plt.savefig(fname, dpi=400)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx_label', type=int, default=12,
                        help='index choice from 12 labels (default value (12) is the enthalpy H)')
    parser.add_argument('-dn', '--dim_node', type=int, default=128, help='dim of nodes features after embedded')
    parser.add_argument('-de', '--dim_edge', type=int, default=128, help='dim of edges features after type embedded')
    parser.add_argument('-lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=512, help='batch size of samples')
    parser.add_argument('-n', '--n_conv', type=int, default=3, help='number of convolution layers')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='epochs for training')
    parser.add_argument('-M', '--model', type=str, default='dtnn', help='select a model: dtnn, mgcn or schnet')
    parser.add_argument('-mom', '--momentum', type=float, default=0.95, help='momentum parameters for SGD optimizer')
    parser.add_argument('-opt', type=str, default='adam', help='optimizer type')
    parser.add_argument('-cuda', type=int, default=0, help='select a GPU to train model')
    parser.add_argument('-type', type=str, default='non', help='which modeling type, optional: [non, com]')
    parser.add_argument('-o', '--out_dir', type=str, default='output', help='output directory')
    parser.add_argument('-norm', type=bool, default=False, help='whether conduct normalization and de-normalization')
    parser.add_argument('-mr', '--multi_read', type=bool, default=False,
                        help='whether read results of each convolutional layer')
    parser.add_argument('-dist', type=bool, default=True, help='whether use distance embedding')
    parser.add_argument('-ei', type=str, default=None, help='initialization way for embedding layer parameters')
    parser.add_argument('-ci', type=str, default='normal', help='initialization way for convolutional layer parameters')
    parser.add_argument('-ri', type=str, default='uniform', help='initialization way for readout layer parameters')

    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    plt.style.use('seaborn')

    args = parser.parse_args()
    graph_type = args.type
    idx_label = args.idx_label
    epochs = args.epochs
    opt = args.opt
    lr = args.lr
    batch_size = args.batch_size
    n_conv = args.n_conv
    dim_node = args.dim_node
    dim_edge = args.dim_edge
    model_type = args.model
    momentum = args.momentum
    out_dir = args.out_dir
    norm = args.norm
    multi_read = args.multi_read
    embed_init = args.ei
    conv_init = args.ci
    read_init = args.ri
    dist = args.dist

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.cuda))
    else:
        device = torch.device('cpu')
    print('using device:', device)
    ############
    qm9 = QM9Dataset(type=graph_type)
    # train_eval_test_spit_list = [0.7, 0.15, 0.15]
    train_eval_test_spit_list = [0.9, 0.05, 0.05]  # 总数据量134 k

    trainset, evalset, testset = split_dataset(qm9, train_eval_test_spit_list, shuffle=True, random_state=123)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=batcher())
    eval_loader = DataLoader(evalset, batch_size=batch_size, shuffle=True, collate_fn=batcher())
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, collate_fn=batcher())

    N = len(qm9)
    dist_extreme = np.array([qm9.get_dist_min_max_with_idx(i) for i in range(N)])
    max_num_atoms = np.max(qm9.dict['num_atoms'])
    dist_min, dist_max = dist_extreme[:, 0].min(), dist_extreme[:, 1].max()
    prop_desc = qm9.get_prop_stat()

    # model selecting
    # if model_type == 'dtnn':
    #     model = EnhancedDTNN(dim_node=dim_node, dim_edge=dim_edge, cutoff_low=np.floor(dist_min),
    #                          cutoff_high=np.ceil(dist_max), n_conv=n_conv, norm=norm, multi_read=multi_read,
    #                          dim_node_app=1, embed_init=embed_init, conv_init=conv_init,read_init=read_init,
    #                          dist=dist)
    # elif model_type == 'mgcn':
    #     model = MGCNModel(dim=dim_node, output_dim=1, edge_dim=dim_edge, cutoff=np.ceil(dist_max),
    #                       width=np.ceil(dist_max) / 30, n_conv=n_conv, norm=norm)
    # else:
    #     raise TypeError('Please input a correct model name (dtnn, mgcn or schnet)!')

    model = DMGCN(dim_node, dim_edge, cutoff_low=np.floor(dist_min), cutoff_high=np.ceil(dist_max), n_conv=n_conv,
                  norm=norm,multi_read=multi_read, dim_node_app=1, embed_init=embed_init, conv_init=conv_init,
                  read_init=read_init)




    # convert to a gpu-supported model (or still in cpu)
    model.to(device)
    if not os.path.exists('../{}/'.format(out_dir)):
        os.mkdir('../{}/'.format(out_dir))
    model_dir = '../{}/{}/'.format(out_dir, graph_type)
    res_dir = '../{}/{}-{}-{}/'.format(out_dir, str(datetime.date.today()), model_type, idx_label)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    model_name = 'model_{}_conv_{}_label_{}.pkl'.format(model_type, n_conv, idx_label)
    if os.path.exists(os.path.join(model_dir, model_name)):
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
        print('已从已训练的模型中加载参数！')

    # 标签归一化
    # if norm:
    #     labels = qm9.labels[:,idx_label]
    #     mean = labels.mean().item()
    #     std = labels.std().item()
    #     model.set_mean_std(mean, std, device=device)

    fname_record = 'model_{}_conv_{}_label_{}_{}.txt'.format(model_type, n_conv, str(datetime.date.today()),
                                                             round(time.time()))
    fname_record = os.path.join(res_dir, fname_record)
    para_record = [
        'model_type: {}\n'.format(model_type),
        'number of parameters: {}\n'.format(count_model_parameter(model)),
        'batch_size: {}\n'.format(batch_size),
        'n_conv: {}\n'.format(n_conv),
        'learngin rate: {}\n'.format(lr),
        'device: {}\n'.format(device)
    ]
    # 写入参数信息到文件
    with open(fname_record, 'w') as f:
        f.write('=' * 10 + str(datetime.datetime.now()) + '=' * 10 + '\n')
        f.writelines(para_record)
        f.write('-' * 20 + '\n')

    # 输出参数信息到终端
    print(fname_record)
    with open(fname_record, 'r') as f:
        print(f.read())

    ###################
    # training & evaluating
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        raise TypeError('please input a supported optimizer type')

    running_loss, running_mae, epoch_best, mae_best, best_model_wts = train_and_evaluate(model, optimizer, train_loader,
                                                                                         eval_loader, epochs, idx_label,
                                                                                         device=device,
                                                                                         log_file_name=fname_record)

    torch.save(best_model_wts, os.path.join(model_dir, model_name))
    print('Best model has been saved! Best model obtained at epoch {} with eval MAE {:.4f}'.format(epoch_best, mae_best))

    ######################
    # running loss & mae plot
    title = 'model: {}, label: {}, {}'.format(model_type, idx_label, qm9.prop_names[idx_label])
    fig_fname = 'model_{}_label_{}_epoch_{}_{}_{}'.format(model_type, idx_label, epochs,
                                                          str(datetime.date.today()), round(time.time()))
    fig_fname = os.path.join(res_dir, fig_fname)
    for omit in [0, int(epochs / 2)]:
        plot_train_eval_result(running_mae, running_mae, omit, title=title, fname=fig_fname + 'omit_{}'.format(omit))
        plot_train_eval_result(running_mae, running_mae, omit, title=title, fname=fig_fname + 'omit_{}'.format(omit))

    ####################################
    # testing
    print('=============== testing ===============')
    predict_specific_idx = []
    labels_specific_idx = []
    with open(fname_record, 'a') as f:
        with torch.no_grad():
            for i, (graphs, labels) in enumerate(test_loader):
                graphs = graphs.to(device)
                labels = labels.to(device)
                predict_specific_idx += model(graphs).tolist()
                labels_specific_idx += labels[:, idx_label].tolist()

        mse = F.mse_loss(torch.tensor(predict_specific_idx), torch.tensor(labels_specific_idx)).item()
        mae = F.l1_loss(torch.tensor(predict_specific_idx), torch.tensor(labels_specific_idx)).item()
        r2 = metrics.r2_score(labels_specific_idx, predict_specific_idx)
        f.write('MSE on test set: {:.4f}\n'.format(mse))
        f.write('MAE on test set: {:.4f}\n'.format(mae))
        f.write('R2 score: {:.4f}\n'.format(r2))
        print('MSE on test set: {:.4f}'.format(mse))
        print('MAE on test set: {:.4f}'.format(mae))
        print('R2 score: {:.4f}'.format(r2))

    # 测试集效果图
    plt.figure(figsize=(8, 4))
    plt.hist(labels_specific_idx, label='Target', bins=30, alpha=0.7)
    plt.hist(predict_specific_idx, label='Predict', bins=30, alpha=0.7)
    plt.legend()
    plt.title('model: {}, MSE: {:.4f}, MAE: {:.4f}'.format(model_type, mse, mae))
    plt.xlabel('Predict: {} {}'.format(idx_label, qm9.prop_names[idx_label]))
    plt.ylabel('Count')
    fig_fname = 'result_model_{}_label_{}_epoch_{}_{}_{}.png'.format(model_type, idx_label, epochs,
                                                                     str(datetime.date.today()), round(time.time()))
    fig_fname = os.path.join(res_dir, fig_fname)
    plt.savefig(fig_fname, dpi=400)
