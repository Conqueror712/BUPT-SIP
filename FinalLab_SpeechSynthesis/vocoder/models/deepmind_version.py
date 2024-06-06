import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.display import *
from utils.dsp import *


class WaveRNN(nn.Module) :
    def __init__(self, hidden_size=896, quantisation=256) :
        """
        初始化 WaveRNN 模型。

        参数:
        - hidden_size: 隐藏层的大小，决定了网络的复杂度。
        - quantisation: 量化级别，通常与音频处理中的量化比特数有关。
        """
        super(WaveRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.split_size = hidden_size // 2
        
        # 网络中的主要矩阵乘法操作
        self.R = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        
        # 输出全连接层
        self.O1 = nn.Linear(self.split_size, self.split_size)
        self.O2 = nn.Linear(self.split_size, quantisation)
        self.O3 = nn.Linear(self.split_size, self.split_size)
        self.O4 = nn.Linear(self.split_size, quantisation)
        
        # 输入全连接层
        self.I_coarse = nn.Linear(2, 3 * self.split_size, bias=False)
        self.I_fine = nn.Linear(3, 3 * self.split_size, bias=False)

        # 门控单元的偏置
        self.bias_u = nn.Parameter(torch.zeros(self.hidden_size))
        self.bias_r = nn.Parameter(torch.zeros(self.hidden_size))
        self.bias_e = nn.Parameter(torch.zeros(self.hidden_size))
        
        # 显示可训练参数数量
        self.num_params()

        
    def forward(self, prev_y, prev_hidden, current_coarse) :
        """
        前向传播定义。

        参数:
        - prev_y: 上一时间步的输出。
        - prev_hidden: 上一时间步的隐藏状态。
        - current_coarse: 当前粗糙的样本值。
        """
        # 主矩阵乘法
        R_hidden = self.R(prev_hidden)
        R_u, R_r, R_e, = torch.split(R_hidden, self.hidden_size, dim=1)
        
        # 输入的预处理 
        coarse_input_proj = self.I_coarse(prev_y)
        I_coarse_u, I_coarse_r, I_coarse_e = \
            torch.split(coarse_input_proj, self.split_size, dim=1)
        
        # 处理当前粗糙样本和先前输出
        fine_input = torch.cat([prev_y, current_coarse], dim=1)
        fine_input_proj = self.I_fine(fine_input)
        I_fine_u, I_fine_r, I_fine_e = \
            torch.split(fine_input_proj, self.split_size, dim=1)
        

        I_u = torch.cat([I_coarse_u, I_fine_u], dim=1)
        I_r = torch.cat([I_coarse_r, I_fine_r], dim=1)
        I_e = torch.cat([I_coarse_e, I_fine_e], dim=1)
        
        # 门控计算
        u = F.sigmoid(R_u + I_u + self.bias_u)
        r = F.sigmoid(R_r + I_r + self.bias_r)
        e = F.tanh(r * R_e + I_e + self.bias_e)
        hidden = u * prev_hidden + (1. - u) * e
        
        # 分割隐藏状态
        hidden_coarse, hidden_fine = torch.split(hidden, self.split_size, dim=1)
        
        # 输出计算 
        out_coarse = self.O2(F.relu(self.O1(hidden_coarse)))
        out_fine = self.O4(F.relu(self.O3(hidden_fine)))

        return out_coarse, out_fine, hidden
    
        
    def generate(self, seq_len):
        """
        生成音频样本序列。

        参数:
        - seq_len: 生成序列的长度。
        """
        with torch.no_grad():
            # 初始化输出列表和隐藏状态
            b_coarse_u, b_fine_u = torch.split(self.bias_u, self.split_size)
            b_coarse_r, b_fine_r = torch.split(self.bias_r, self.split_size)
            b_coarse_e, b_fine_e = torch.split(self.bias_e, self.split_size)

            c_outputs, f_outputs = [], []

 
            out_coarse = torch.LongTensor([0]).cuda()
            out_fine = torch.LongTensor([0]).cuda()

            # 隐藏层
            hidden = self.init_hidden()

            # 计时
            start = time.time()

            # 循环
            for i in range(seq_len) :

                # 拆分为两个隐藏状态
                hidden_coarse, hidden_fine = \
                    torch.split(hidden, self.split_size, dim=1)

                # 更新隐藏状态和输出
                out_coarse = out_coarse.unsqueeze(0).float() / 127.5 - 1.
                out_fine = out_fine.unsqueeze(0).float() / 127.5 - 1.
                prev_outputs = torch.cat([out_coarse, out_fine], dim=1)

                # 输入
                coarse_input_proj = self.I_coarse(prev_outputs)
                I_coarse_u, I_coarse_r, I_coarse_e = \
                    torch.split(coarse_input_proj, self.split_size, dim=1)

                R_hidden = self.R(hidden)
                R_coarse_u , R_fine_u, \
                R_coarse_r, R_fine_r, \
                R_coarse_e, R_fine_e = torch.split(R_hidden, self.split_size, dim=1)

                # 计算粗糙信号
                u = F.sigmoid(R_coarse_u + I_coarse_u + b_coarse_u)
                r = F.sigmoid(R_coarse_r + I_coarse_r + b_coarse_r)
                e = F.tanh(r * R_coarse_e + I_coarse_e + b_coarse_e)
                hidden_coarse = u * hidden_coarse + (1. - u) * e

                # 计算粗糙输出
                out_coarse = self.O2(F.relu(self.O1(hidden_coarse)))
                posterior = F.softmax(out_coarse, dim=1)
                distrib = torch.distributions.Categorical(posterior)
                out_coarse = distrib.sample()
                c_outputs.append(out_coarse)

                # Project the [prev outputs and predicted coarse sample]
                coarse_pred = out_coarse.float() / 127.5 - 1.
                fine_input = torch.cat([prev_outputs, coarse_pred.unsqueeze(0)], dim=1)
                fine_input_proj = self.I_fine(fine_input)
                I_fine_u, I_fine_r, I_fine_e = \
                    torch.split(fine_input_proj, self.split_size, dim=1)

                # 计算精细门
                u = F.sigmoid(R_fine_u + I_fine_u + b_fine_u)
                r = F.sigmoid(R_fine_r + I_fine_r + b_fine_r)
                e = F.tanh(r * R_fine_e + I_fine_e + b_fine_e)
                hidden_fine = u * hidden_fine + (1. - u) * e

                # 计算精细输出
                out_fine = self.O4(F.relu(self.O3(hidden_fine)))
                posterior = F.softmax(out_fine, dim=1)
                distrib = torch.distributions.Categorical(posterior)
                out_fine = distrib.sample()
                f_outputs.append(out_fine)

                # 将隐藏状态重新组合在一起
                hidden = torch.cat([hidden_coarse, hidden_fine], dim=1)

                # 显示进度
                speed = (i + 1) / (time.time() - start)
                stream('Gen: %i/%i -- Speed: %i',  (i + 1, seq_len, speed))

            coarse = torch.stack(c_outputs).squeeze(1).cpu().data.numpy()
            fine = torch.stack(f_outputs).squeeze(1).cpu().data.numpy()        
            output = combine_signal(coarse, fine)
        
        return output, coarse, fine

    def init_hidden(self, batch_size=1) :
        """
        初始化隐藏状态。

        参数:
        - batch_size: 批大小，默认为1。
        """
        return torch.zeros(batch_size, self.hidden_size).cuda()
    
    def num_params(self) :
        """
        计算模型的可训练参数数量。
        """
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)
