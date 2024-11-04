import os
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import torch
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import matplotlib.pyplot as plt
from scipy import sparse
import pickle
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import dense_to_sparse
import warnings
warnings.filterwarnings('ignore')

class Dataset_ship(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.args = args
        self.root_path = self.args.root_path
        self.data_path = self.args.data_path
        self.rate = self.args.data_rate
        self.seq_len = self.args.seq_len
        self.label_len = self.args.label_len
        self.pred_len = self.args.pred_len

        self.device = self.acquire_device()
        self.read_data()


    def read_data(self):
        print('Loading_data')
        path = os.path.join(self.root_path, self.data_path)
        f = open(path, 'rb')
        self.data_raw = pickle.load(f)
        f.close()
        print('Loading_finish: {} batch'.format(int(len(self.data_raw))))
        self.border1 = int(len(self.data_raw) * self.rate[0] / 10)
        self.border2 = int(len(self.data_raw) * (self.rate[0] + self.rate[1])/ 10)

    def pointer(self, flag):
        if flag == 'train':
            self.data = self.data_raw[:self.border1]
        elif flag == 'val':
            self.data = self.data_raw[self.border1: self.border2]
        elif flag == 'test':
            self.data = self.data_raw[self.border2:]
        elif flag == 'pred':
            self.data = [self.data_raw[-2]]

    def get_length(self, flag):
        if flag == 'train':
            return self.border1
        elif flag == 'test' or flag == 'val':
            return self.border2 - self.border1

    def batch_operation(self, batch_data):

        E_index = []
        E_attr =[]
        for a, adj in enumerate(batch_data[1:]):
            adj = adj[:self.seq_len, :, :]
            max_value = np.max(adj)
            min_value = np.min(adj)
            A = (adj-min_value)/(max_value-min_value)
            if a ==2:
                padding = np.zeros_like(A)
                A_above = np.concatenate((padding,padding),axis=-1)
                A_below = np.concatenate((A, padding),axis=-1)
                A = np.concatenate((A_above,A_below),axis=1)
            A = torch.tensor(A).to(self.device)
            edge_index, edge_attr = dense_to_sparse(A)
            E_index.append(edge_index)
            E_attr.append(edge_attr)

        batch = self.standard(batch_data[0][:self.seq_len + self.pred_len,:,2:]) # B2
        batch_x = torch.tensor(batch[:self.seq_len, :, :]).to(self.device)
        batch_y = torch.tensor(batch[self.seq_len:, :, :]).to(self.device)


        return batch_x, batch_y, E_index, E_attr

    def standard (self, batch):
        batch = batch.astype("float32")
        mean = batch.mean(axis=(0, 1), keepdims=True)
        std = batch.std(axis=(0, 1), keepdims=True)
        return (batch - mean) / std

    def acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
        return device

    def __getitem__(self, idx):
        return self.batch_operation(self.data[idx])

    def __len__(self):
        return len(self.data)