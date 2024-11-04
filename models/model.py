import torch
import torch.nn as nn
from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from torch_geometric.nn.models import GCN as GCNModel



class GMLTP(nn.Module):
    def __init__(self, dec_in, c_out,  out_len, gnn_feats, gnn_layer,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', activation='gelu',distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(GMLTP, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.device = device

        # MGSC
        self.gnn_pos = GCNModel(in_channels=2, hidden_channels=gnn_feats, num_layers=gnn_layer)
        self.gnn_speed = GCNModel(in_channels=2, hidden_channels=gnn_feats, num_layers=gnn_layer)
        self.gnn_channel = GCNModel(in_channels=2, hidden_channels=gnn_feats, num_layers=gnn_layer)
        self.gnn_size = GCNModel(in_channels=1, hidden_channels=gnn_feats, num_layers=gnn_layer)
        # Encoding
        self.enc_embedding = DataEmbedding(gnn_feats * 4, d_model)
        self.dec_embedding = DataEmbedding(dec_in, d_model)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [ConvLayer(d_model) for l in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)


    def forward(self, x_enc, x_dec, E_index, E_attr,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        B, L, F = x_enc.shape
        x_enc = x_enc.contiguous().view(-1, F)

        enc_pos = self.gnn_pos(x_enc[:, :2],E_index[0], E_attr[0].type(torch.float32))
        enc_speed = self.gnn_speed(x_enc[:, 2:4], E_index[1], E_attr[1].type(torch.float32))
        enc_size = self.gnn_size(x_enc[:, 5].unsqueeze(1), E_index[3], E_attr[3].type(torch.float32))
        pos = torch.cat((x_enc[:,:2], x_enc[:, 5:]), dim=0)
        enc_channel = self.gnn_channel(pos, E_index[2], E_attr[2].type(torch.float32))[:B*L,:]

        enc = torch.cat((enc_pos,enc_speed,enc_channel,enc_size),dim=-1)
        x_enc = enc.view(B, L, -1)

        enc_out = self.enc_embedding(x_enc)
        dec_out = self.dec_embedding(x_dec)

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)


        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

