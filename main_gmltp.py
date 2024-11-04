import argparse
import os
import torch
from exp.exp_gmltp import Exp_Gmltp

parser = argparse.ArgumentParser(description='[GMLTP] Graph-driven Multi-vessel Long-term Trajectories Prediction')

parser.add_argument('--model', type=str, required=False, default='GMLTP')
parser.add_argument('--data', type=str, required=False, default='AIS_processed_mgsc', help='data')
parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='AIS_processed_mgsc.cpkl', help='data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--data_rate',type=list, default=[6,2,2], help='ratio of train, val, test')

parser.add_argument('--seq_len', type=int, default=48, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=24, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=2, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--do_predict', default=False, help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)

parser.add_argument('--itr', type=int, default=6, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=12, help='train epochs')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--use_multi_gpu', default=False, help='use multiple gpus')
parser.add_argument('--devices', type=str, default='3,1,2,0',help='device ids of multile gpus')

parser.add_argument('--gnn_feats',type=int, default=128)
parser.add_argument('--gnn_layer',type=int, default=2)


args = parser.parse_args()


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]


print('Args in experiment:')
print(args)

Exp = Exp_Gmltp


for ii in range(args.itr):
    # setting record of experiments
    #ii=ii+6
    if ii == 0:
        args.seq_len, args.label_len, args.pred_len = 24, 12, 6
    elif ii == 1:
        args.seq_len, args.label_len, args.pred_len = 48, 24, 12
    elif ii == 2:
        args.seq_len, args.label_len, args.pred_len = 96, 48, 24
    elif ii == 3:
        args.seq_len, args.label_len, args.pred_len = 96, 48, 48
    elif ii == 4:
        args.seq_len, args.label_len, args.pred_len = 96, 48, 72
    elif ii == 5:
        args.seq_len, args.label_len, args.pred_len = 96, 48, 96
    elif ii == 6:
        args.seq_len, args.label_len, args.pred_len = 144, 72, 48

    setting = '{}_{}_sl{}_ll{}_pl{}'.format(args.model, args.data, args.seq_len,args.label_len, args.pred_len)

    exp = Exp(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()

# 192 - 10   144 for 48
# 96 - 20    72 for 24
# 64 - 30    48 for 16
# 32 - 60    24 for 8
# 16 - 120   12 for 4