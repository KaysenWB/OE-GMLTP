from data.data_loader import  Dataset_ship
from exp.exp_basic import Exp_Basic
from models.model import GMLTP
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from torch_geometric.loader import DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Gmltp(Exp_Basic):
    def __init__(self, args):
        super(Exp_Gmltp, self).__init__(args)
        self.dataset = Dataset_ship(args)

    def _build_model(self):
        model = GMLTP(
            self.args.dec_in,
            self.args.c_out,
            self.args.pred_len,
            self.args.gnn_feats,
            self.args.gnn_layer,
            self.args.factor,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.dropout,
            self.args.attn,
            self.args.activation,
            self.args.distil,
            self.args.mix,
            self.device,
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        self.dataset.pointer(flag=flag)
        custom_collate_fn = lambda batch: batch
        shuffle = True if flag == 'train' else False
        Loader = DataLoader(dataset=self.dataset, shuffle=shuffle, collate_fn=custom_collate_fn)
        print(flag, len(self.dataset))
        return Loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()
        #criterion = nn.MSELoss()
        return criterion

    def vali(self,  vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu()[:,:,:2])
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        cost_time_average = []
        for epoch in range(self.args.train_epochs):
            train_loader = self._get_data(flag='train')
            train_steps = len(train_loader)

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, E_index, E_attr) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                pred, true = self._process_one_batch(batch_x, batch_y, E_index, E_attr)
                loss = criterion(pred, true[:,:,:2])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            cost_time = time.time() - epoch_time
            cost_time_average.append(cost_time)
            print("Epoch: {} cost time: {}".format(epoch + 1, cost_time))
            train_loss = np.average(train_loss)
            vali_loader = self._get_data(flag='val')
            vali_loss = self.vali(vali_loader, criterion)
            test_loader = self._get_data(flag='test')
            test_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(
            torch.load(best_model_path, map_location={'cuda:{}'.format(self.args.gpu): 'cuda:0'}))
        self.cost_time_average = np.mean(cost_time_average)

        return self.model

    def test(self, setting):
        test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(
            torch.load(best_model_path, map_location={'cuda:{}'.format(self.args.gpu): 'cuda:0'}))

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, E_index, E_attr) in enumerate(test_loader):
            pred, true = self._process_one_batch(batch_x, batch_y, E_index, E_attr)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.vstack(preds)
        trues = np.vstack(trues)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, ade, fde = metric(preds, trues[:,:,:2])
        print('mae:{}, mse:{}, ade{}, fde{}'.format(mae, mse, ade, fde))
        # print('average_time: {}'.format(self.cost_time_average))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, ade, fde]))

        return

    def predict(self, setting, load=False):
        pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(
                torch.load(best_model_path, map_location={'cuda:{}'.format(self.args.gpu): 'cuda:0'}))

        self.model.eval()

        preds = []
        trues = []
        pre_tra = []

        for i, (batch_x, batch_y, E_index, E_attr) in enumerate(pred_loader):

            pred, true = self._process_one_batch(batch_x, batch_y, E_index, E_attr)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            pre_tra.append(batch_y.squeeze().detach().cpu().numpy())

        # result save
        preds = np.array(preds)
        trues = np.array(trues)
        pre_tra = np.array(pre_tra)

        preds = preds.reshape(-1, preds.shape[-1])
        trues = trues.reshape(-1, preds.shape[-1])
        pre_tra = pre_tra.reshape(-1, pre_tra.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'pred_prediction.npy', preds)
        np.save(folder_path + 'true_prediction.npy', trues)
        np.save(folder_path + 'pre_tra.npy', pre_tra)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")

        return

    def _process_one_batch(self, batch_x, batch_y, E_index, E_attr):

        batch_x = batch_x.squeeze().permute(1, 0, 2)
        batch_y = batch_y.squeeze().permute(1, 0, 2)
        E_index = [i.squeeze() for i in E_index]
        E_attr = [i.squeeze() for i in E_attr]

        padding = torch.zeros_like(batch_y).to(self.device)
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], padding[:,:self.args.pred_len,:]], dim=1)

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, dec_inp, E_index, E_attr)
        else:
            outputs = self.model(batch_x, dec_inp, E_index, E_attr)

        batch_y = batch_y[:, -self.args.pred_len:, :]
        return outputs, batch_y
