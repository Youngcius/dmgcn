import torch
import datetime
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import time


def train(graph_model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, idx_label=0,
          device=torch.device('cpu')):
    """
    :param graph_model: Graph Neural Network instance
    :param dataloader: DataLoader instance
    :param optimizer: optimizer for updating parameters of the GNN model
    :param idx_label: index of 12 kinds of labels to determine which label to be trained and predicted
    :param device: default is CPU, requires to be set to use GPU
    :return: MSE loss of each test epoch, MAE loss of each test epoch
    """
    loss_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    loss_train = []
    mae_train = []
    for i, (graphs, labels) in enumerate(dataloader):
        labels = labels.to(device)
        graphs = graphs.to(device)
        if labels.dim() != 1:
            labels = labels[:, idx_label]
        predict = graph_model(graphs)
        loss = loss_func(predict, labels)
        mae = mae_func(predict.detach(), labels.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
        mae_train.append(mae.item())
        if (i + 1) % 100 == 0:
            print('\tbatch: {}, loss: {:.4f}, MAE: {:.4f}'.format(i + 1, loss.item(), mae.item()))
    loss_train = np.mean(loss_train)
    mae_train = np.mean(mae_train)
    print('Training:\t loss: {:.4f}, MAE: {:.4f}'.format(loss_train, mae_train))
    return loss_train, mae_train


def test(graph_model: nn.Module, dataloader: DataLoader, idx_label=0, device=torch.device('cpu')):
    """
    :param graph_model: Graph Neural Network instance
    :param dataloader: DataLoader instance
    :param idx_label: index of 12 kinds of labels to determine which label to be trained and predicted
    :param device: default is CPU, requires to be set to use GPU
    :return: MSE loss of each test epoch, MAE loss of each test epoch
    """
    loss_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    loss_test = []
    mae_test = []
    with torch.no_grad():
        for i, (graphs, labels) in enumerate(dataloader):
            graphs = graphs.to(device)
            labels = labels.to(device)
            if labels.dim() != 1:
                labels = labels[:, idx_label]
            predict = graph_model(graphs)
            loss = loss_func(predict.detach(), labels.detach())
            mae = mae_func(predict.detach(), labels.detach())
            loss_test.append(loss.item())
            mae_test.append(mae.item())
        loss_test = np.mean(loss_test)
        mae_test = np.mean(mae_test)
        print('Evaluating:\t loss: {:.4f}, MAE: {:.4f}'.format(loss_test, mae_test))

    return loss_test, mae_test


def train_and_evaluate(graph_model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader,
                       eval_loader: DataLoader, epochs=5, idx_label=0, device=torch.device('cpu'), log_file_name=None):
    """
    training and evaluating process
    :param graph_model: Graph Neural Network instance
    :param optimizer: optimizer for updating parameters of the GNN model in training process
    :param train_loader: DataLoader instance for training
    :param eval_loader: DataLoader instance for evaluating
    :param epochs: number of training epochs
    :param idx_label: index of 12 kinds of labels to determine which label to be trained and predicted
    :param device: default is CPU, requires to be set to use GPU
    :param log_file_name: logging file name
    :return: running_loss, running_mae, epoch_best, mae_best, best_model_wts
    """
    # training & evaluating
    best_model_wts = graph_model.state_dict()
    epoch_best = 0
    mae_best = np.inf
    running_loss = []
    running_mae = []

    if log_file_name is None:
        log_file_name = 'train_eval_record_{}_{}_{}.txt'.format(idx_label, str(datetime.date.today()),
                                                                round(time.time()))
    f = open(log_file_name, 'a')
    print('=============== training & evaluating ===============')
    for e in range(epochs):
        beg = time.time()
        print('---' * 10)
        print('Epoch {}'.format(e + 1))

        loss_train, mae_train = train(graph_model, train_loader, optimizer, idx_label=idx_label, device=device)
        loss_eval, mae_eval = test(graph_model, eval_loader, idx_label=idx_label, device=device)
        running_loss.append([loss_train, loss_eval])
        running_mae.append([mae_train, mae_eval])
        if mae_eval < mae_best:
            mae_best = mae_eval
            epoch_best = e
            best_model_wts = graph_model.state_dict()

        # 每个 epoch 结果记录到文件
        rec = 'train MSE: {:.4f}, evaluate MSE: {:.4f}\n'.format(running_loss[e][0], running_loss[e][1])
        f.write(rec)
        print(rec)

        rec = 'train MAE: {:.4f}, evaluate MAE: {:.4f}\n'.format(running_mae[e][0], running_mae[e][1])
        f.write(rec)
        print(rec)

        end = time.time()
        rec = '--- time consumption (s): {}\n'.format(round(end - beg))
        f.write(rec)
        print(rec)

    f.close()

    # save running loss & MAE
    np.savetxt(log_file_name[:-4] + '_running_loss.txt', running_loss)
    np.savetxt(log_file_name[:-4] + '_running_mae.txt', running_mae)

    epoch_best += 1
    return running_loss, running_mae, epoch_best, mae_best, best_model_wts
