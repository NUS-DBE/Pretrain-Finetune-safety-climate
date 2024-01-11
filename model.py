import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
# 定义模型
class AccidentClassifier(nn.Module):
    def __init__(self, num_features):
        super(AccidentClassifier, self).__init__()
        self.layer_1 = nn.Linear(num_features, 64)
        self.layer_2 = nn.Linear(64, 128)
        self.bn1=nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer_3 = nn.Linear(128, 128)
        self.layer_out = nn.Linear(128, 16)
        self.layer_out1 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.sig=nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, inputs):
        x1 = self.relu(self.layer_1(inputs))
        x2 = self.relu(self.layer_2(x1))
        try:
            x2=self.bn1(x2)
        except:
            pass
        x3 = self.relu(self.layer_3(x2))
        x3=x3+x2
        x = self.bn2(x3)
        x = self.layer_out(x)
        x = self.layer_out1(x)
        # x=self.sig(x)
        # x=x.squeeze(1)
        return x
# class AccidentClassifier(nn.Module):
#     def __init__(self, num_features,hidden_dim, n_layers, n_heads, dropout):
#         super().__init__()
#         d_model = 128
#         self.pre = nn.Linear(num_features, d_model)
#         self.pre2 = nn.Linear(d_model, d_model)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,batch_first=True, nhead=n_heads)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
#         self.transformer=nn.Transformer(d_model=d_model,batch_first=True, nhead=n_heads)
#         self.fc = nn.Linear(d_model, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.output_layer = nn.Linear(hidden_dim, 1)
#         self.last=nn.Sigmoid()
#         self.relu=nn.ReLU()
#
#     def forward(self, x):
#         x=x.unsqueeze(-1)
#         # b,s,d=x.shape
#         # x=x.view(b*s,d)
#         x = self.relu(self.pre(x))
#         # x=self.relu(self.pre2(x))
#
#         x = self.transformer_encoder(x)
#         # x=self.transformer(x,x)
#         x = x[:,-1,:]#torch.mean(x, dim=1)  # 对序列维度求平均
#         x = self.relu(self.fc(x))
#         # x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.output_layer(x)
#         x=self.last(x)
#         return x