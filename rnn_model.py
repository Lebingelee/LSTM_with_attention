import torch.nn as nn
import torch
import torchvision
from torch.nn import init
import torch.nn.functional as F
from torch.nn import LayerNorm
# Defin RNN model


class RNN_with_product_id_embedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=2, product_id_num=10, attention_dim = 64):
        super(RNN_with_product_id_embedding, self).__init__()
        embedding_dim = 64
        self.embedding_dim = embedding_dim
        self.product_id_embedding = nn.Embedding(product_id_num, embedding_dim)
        embedded_input_dim = 64
        self.input_embedding = torchvision.ops.MLP(in_channels=input_dim, hidden_channels=[hidden_dim]*2 + [embedded_input_dim], dropout=0.0)
        # self.rnn = nn.RNN(embedded_input_dim+embedding_dim, hidden_dim, layers, batch_first=True, dropout=0.0)
        self.lstm = nn.LSTM(embedded_input_dim+embedding_dim, hidden_dim, layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attention_dim, batch_first=True)
        self.layer_norm = LayerNorm(hidden_dim)
        


        self.final_ac = nn.Softplus()
        self.dropout = nn.Dropout(0.2)
        self.initial_feature_to_hidden = nn.Linear(input_dim+embedding_dim, hidden_dim * layers)
        # self.initial_feature_to_hidden = torchvision.ops.MLP(in_channels=input_dim+embedding_dim,
        #                                                      hidden_channels=[hidden_dim * layers]*3)
 # Initialize parameters
        self._initialize_weights()

    def attention_layer(self, lstm_output):
        """
        自注意力层
        :param lstm_output: LSTM的输出 (batch_size, seq_len, hidden_dim)
        :return: 加权后的上下文向量和注意力权重
        """
        # 1. 投影到注意力空间 (batch_size, seq_len, attention_dim)
        attention_proj = torch.tanh(self.attention_w(lstm_output))
        
        # 2. 生成注意力分数 (batch_size, seq_len, 1)
        attention_scores = self.attention_u(attention_proj)
        
        # 3. Softmax 归一化 (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1).squeeze(-1)
        
        # 4. 注意力加权求和，得到上下文向量 (batch_size, hidden_dim)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        
        return context_vector, attention_weights
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        init.kaiming_normal_(param, mode='fan_in', nonlinearity='tanh')
                    elif 'weight_hh' in name:
                        init.orthogonal_(param)
                    elif 'bias' in name:
                        init.constant_(param, 0)
            elif isinstance(m, nn.Embedding):
                init.normal_(m.weight, mean=0, std=self.embedding_dim**-0.5)

                                                             
    def forward_back(self, x, product_id):
        product_id = product_id.long()
        product_id = self.product_id_embedding(product_id)
        product_id = product_id.unsqueeze(1).repeat(1, x.size(1), 1)
        x = self.input_embedding(x)
        x = torch.cat((x, product_id), dim=2)
        out, _ = self.rnn(x)
        return out
    
    def forward_all(self, x, product_id):
        return self.forward(x, product_id)
        product_id = product_id.long()
        product_id = self.product_id_embedding(product_id)
        product_id = product_id.unsqueeze(1).repeat(1, x.size(1), 1)
        x = self.input_embedding(x)
        x = torch.cat((x, product_id), dim=2)
        # h0 = self.initial_feature_to_hidden(x[:, 0, :]).reshape(self.rnn.num_layers, x.size(0), -1)
        # out, _ = self.rnn(x,h0)
        out, _ = self.rnn(x)
        out = self.fc(out)
        # return out
        x, y, z = out[:, :, 0], out[:, :, 1], out[:, :, 2]
        x, y = self.final_ac(x), self.final_ac(y)
        z = 0.5 * (x - y) + 0.5 * z
        out_ = torch.stack((x, y, z), dim=2)
        return out_
    
    def forward(self, x, product_id):
        product_id = product_id.long()
        product_id = self.product_id_embedding(product_id)
        product_id = product_id.unsqueeze(1).repeat(1, x.size(1), 1)
        x = self.input_embedding(x)
        x = torch.cat((x, product_id), dim=2)
        # h0 = self.initial_feature_to_hidden(x[:, 0, :]).reshape(self.rnn.num_layers, x.size(0), -1)
        # out, _ = self.rnn(x, h0)
        lstm_out, _ = self.lstm(x)

        attn_output, attn_weights = self.multihead_attention(
            query=lstm_out, key=lstm_out, value=lstm_out
        )
        attn_output = self.layer_norm(attn_output)
        attn_output = self.dropout(attn_output)
        out = self.fc(attn_output)
        # x = torch.cat((x, product_id), dim=2)
        #print(x,out)
        #print(out)
        # return out
        x, y, z = out[ :, :, 0], out[:,  :, 1], out[:, :, 2]
        x, y = self.final_ac(x), self.final_ac(y)
        z = 0.5 * (x - y) + 0.5 * z
        out_ = torch.stack((x, y, z), dim=2)
        return out_
    
    def forward_hidden(self, x, product_id, h0):
        assert False, "Do not use"
        product_id = product_id.long()
        product_id = self.product_id_embedding(product_id)
        product_id = product_id.unsqueeze(1).repeat(1, x.size(1), 1)
        # x = torch.cat((x, product_id), dim=2)
        # out, _ = self.rnn(x, h0)
        out, _ = self.rnn(x)
        out = self.fc(out)
        # return out
        # x, y, z = out[:, :, 0], out[:, :, 1], out[:, :, 2]
        # x, y = self.final_ac(x), self.final_ac(y)
        # out_ = torch.stack((x, y, z), dim=2)
        # return out_