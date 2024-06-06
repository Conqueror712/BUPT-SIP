import torch
import torch.nn as nn
import torch.nn.functional as F
from vocoder.distribution import sample_from_discretized_mix_logistic
from vocoder.display import *
from vocoder.audio import *


class ResBlock(nn.Module):
    def __init__(self, dims):
        """
        初始化残差块。
        参数：
        - dims: 残差块的维度（即通道数或特征数）。
        """
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        """
        前向传播逻辑。
        参数：
        - x: 输入特征。
        """
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        """
        初始化 MelResNet 模型。
        参数：
        - res_blocks: 残差块的数量。
        - in_dims: 输入维度。
        - compute_dims: 计算维度。
        - res_out_dims: 残差块输出维度。
        - pad: 卷积的填充大小。
        """
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=k_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        """
        前向传播逻辑。
        参数：
        - x: 输入特征。
        """
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers: x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):
    def __init__(self, x_scale, y_scale):
        """
        初始化 Stretch2d 模块。
        参数：
        - x_scale: 沿x轴的扩展比例。
        - y_scale: 沿y轴的扩展比例。
        """
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        """
        前向传播，进行特征图扩展。
        参数：
        - x: 输入特征图。
        """
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):
    def __init__(self, feat_dims, upsample_scales, compute_dims,
                 res_blocks, res_out_dims, pad):
        """
        初始化 UpsampleNetwork 网络。
        参数：
        - feat_dims: 特征维度。
        - upsample_scales: 上采样比例列表。
        - compute_dims: 计算维度。
        - res_blocks: 残差块数量。
        - res_out_dims: 残差块输出维度。
        - pad: 填充大小。
        """
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        """
        前向传播逻辑。
        参数：
        - m: 输入特征。
        """
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers: m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


class WaveRNN(nn.Module):
    def __init__(self, rnn_dims, fc_dims, bits, pad, upsample_factors,
                 feat_dims, compute_dims, res_out_dims, res_blocks,
                 hop_length, sample_rate, mode='RAW'):
    """
    参数:
    - rnn_dims: RNN 层的维度。
    - fc_dims: 全连接层的维度。
    - bits: 用于量化的位数，影响音频的输出解析度。
    - pad: 用于填充的大小，影响卷积操作。
    - upsample_factors: 上采样因子数组，决定特征上采样的倍数。
    - feat_dims: 特征维度。
    - compute_dims: 计算维度，用于中间层。
    - res_out_dims: 残差网络输出维度。
    - res_blocks: 残差块的数量。
    - hop_length: 音频处理中的跳跃长度，影响时间分辨率。
    - sample_rate: 采样率，定义音频的时间分辨率。
    - mode: 模型的运行模式，'RAW' 用于原始波形生成，'MOL' 用于混合逻辑输出。
    """
        super().__init__()
        self.mode = mode
        self.pad = pad
        if self.mode == 'RAW' :
            self.n_classes = 2 ** bits
        elif self.mode == 'MOL' :
            self.n_classes = 30
        else :
            RuntimeError("Unknown model mode value - ", self.mode)

        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims, res_blocks, res_out_dims, pad)
        self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)

        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        self.num_params()

    def forward(self, x, mels):
        """
        前向传播过程。
        参数:
        - x: 输入音频数据。
        - mels: 梅尔频谱特征，通常是音频的压缩表示形式。
        """
        self.step += 1
        bsize = x.size(0)
        if torch.cuda.is_available():
            h1 = torch.zeros(1, bsize, self.rnn_dims).cuda()
            h2 = torch.zeros(1, bsize, self.rnn_dims).cuda()
        else:
            h1 = torch.zeros(1, bsize, self.rnn_dims).cpu()
            h2 = torch.zeros(1, bsize, self.rnn_dims).cpu()
        mels, aux = self.upsample(mels)

        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))

        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def generate(self, mels, batched, target, overlap, mu_law, progress_callback=None):
        """
        生成音频数据。
        参数:
        - mels: 梅尔频谱特征。
        - batched: 是否批处理。
        - target: 目标长度。
        - overlap: 重叠区域长度。
        - mu_law: 是否使用μ律压缩。
        - progress_callback: 进度回调函数。
        """
        mu_law = mu_law if self.mode == 'RAW' else False
        progress_callback = progress_callback or self.gen_display

        self.eval()
        output = []
        start = time.time()
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():
            if torch.cuda.is_available():
                mels = mels.cuda()
            else:
                mels = mels.cpu()
            wave_len = (mels.size(-1) - 1) * self.hop_length
            mels = self.pad_tensor(mels.transpose(1, 2), pad=self.pad, side='both')
            mels, aux = self.upsample(mels.transpose(1, 2))

            if batched:
                mels = self.fold_with_overlap(mels, target, overlap)
                aux = self.fold_with_overlap(aux, target, overlap)

            b_size, seq_len, _ = mels.size()

            if torch.cuda.is_available():
                h1 = torch.zeros(b_size, self.rnn_dims).cuda()
                h2 = torch.zeros(b_size, self.rnn_dims).cuda()
                x = torch.zeros(b_size, 1).cuda()
            else:
                h1 = torch.zeros(b_size, self.rnn_dims).cpu()
                h2 = torch.zeros(b_size, self.rnn_dims).cpu()
                x = torch.zeros(b_size, 1).cpu()

            d = self.aux_dims
            aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]

            for i in range(seq_len):

                m_t = mels[:, i, :]

                a1_t, a2_t, a3_t, a4_t = (a[:, i, :] for a in aux_split)

                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))

                logits = self.fc3(x)

                if self.mode == 'MOL':
                    sample = sample_from_discretized_mix_logistic(logits.unsqueeze(0).transpose(1, 2))
                    output.append(sample.view(-1))
                    if torch.cuda.is_available():
                        # x = torch.FloatTensor([[sample]]).cuda()
                        x = sample.transpose(0, 1).cuda()
                    else:
                        x = sample.transpose(0, 1)

                elif self.mode == 'RAW' :
                    posterior = F.softmax(logits, dim=1)
                    distrib = torch.distributions.Categorical(posterior)

                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                    output.append(sample)
                    x = sample.unsqueeze(-1)
                else:
                    raise RuntimeError("Unknown model mode value - ", self.mode)

                if i % 100 == 0:
                    gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
                    progress_callback(i, seq_len, b_size, gen_rate)

        output = torch.stack(output).transpose(0, 1)
        output = output.cpu().numpy()
        output = output.astype(np.float64)
        
        if batched:
            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]

        if mu_law:
            output = decode_mu_law(output, self.n_classes, False)
        if hp.apply_preemphasis:
            output = de_emphasis(output)

        # Fade-out at the end to avoid signal cutting out suddenly
        fade_out = np.linspace(1, 0, 20 * self.hop_length)
        output = output[:wave_len]
        output[-20 * self.hop_length:] *= fade_out
        
        self.train()

        return output


    def gen_display(self, i, seq_len, b_size, gen_rate):
        """
        显示生成音频的进度信息。
        参数:
        - i: 当前处理的步数。
        - seq_len: 总步数。
        - b_size: 批量大小。
        - gen_rate: 生成速率，单位kHz。
        """
        pbar = progbar(i, seq_len)
        msg = f'| {pbar} {i*b_size}/{seq_len*b_size} | Batch Size: {b_size} | Gen Rate: {gen_rate:.1f}kHz | '
        stream(msg)

    def get_gru_cell(self, gru):
        """
        从 GRU 层中提取单个 GRU 单元。
        参数:
        - gru: GRU 层。
        返回:
        - GRU 单元。
        """
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def pad_tensor(self, x, pad, side='both'):
        """
        对给定的张量在时间维度上进行填充。
        参数:
        - x: 输入张量。
        - pad: 填充大小。
        - side: 填充的方向，可以是 'both', 'before', 或 'after'。
        返回:
        - 填充后的张量。
        """
        b, t, c = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        if torch.cuda.is_available():
            padded = torch.zeros(b, total, c).cuda()
        else:
            padded = torch.zeros(b, total, c).cpu()
        if side == 'before' or side == 'both':
            padded[:, pad:pad + t, :] = x
        elif side == 'after':
            padded[:, :t, :] = x
        return padded

    def fold_with_overlap(self, x, target, overlap):
    """
    将输入张量折叠成多个重叠段，以便于批量处理。
    参数:
    - x: 输入张量，通常是上采样后的条件特征。
    - target: 目标段长度。
    - overlap: 重叠大小。
    返回:
    - 折叠后的张量。
    """

        _, total_len, features = x.size()

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side='after')

        if torch.cuda.is_available():
            folded = torch.zeros(num_folds, target + 2 * overlap, features).cuda()
        else:
            folded = torch.zeros(num_folds, target + 2 * overlap, features).cpu()

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]

        return folded

    def xfade_and_unfold(self, y, target, overlap):

    """
    应用交叉淡入淡出效果并将其展开为一维数组。

    参数:
    - y: 批处理的音频样本序列，形状为 (num_folds, target + 2 * overlap)，类型为 np.float64。
    - target: 目标段长度。
    - overlap: 重叠步数，用于交叉淡入淡出和 RNN 热身。

    返回值:
    - 展开的一维音频样本数组，形状为 (total_len)，类型为 np.float64。
    """

        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros((silence_len), dtype=np.float64)

        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([fade_out, silence])

        # 将增益应用到重叠样本
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = np.zeros((total_len), dtype=np.float64)

        # 循环以累加所有样本
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded

    def get_step(self) :
        """
        获取当前的步数。

        返回值:
        - 当前的步数，类型为 int。
        """
        return self.step.data.item()

    def checkpoint(self, model_dir, optimizer) :
    """
    保存模型的检查点。

    参数:
    - model_dir: 模型保存的目录。
    - optimizer: 优化器对象。
    """
        k_steps = self.get_step() // 1000
        self.save(model_dir.joinpath("checkpoint_%dk_steps.pt" % k_steps), optimizer)

    def log(self, path, msg) :
    """
    记录日志信息。

    参数:
    - path: 日志文件的路径。
    - msg: 要记录的信息。
    """
        with open(path, 'a') as f:
            print(msg, file=f)

    def load(self, path, optimizer) :
    """
    加载模型的检查点。

    参数:
    - path: 检查点文件的路径。
    - optimizer: 优化器对象。
    """
        checkpoint = torch.load(path)
        if "optimizer_state" in checkpoint:
            self.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            # Backwards compatibility
            self.load_state_dict(checkpoint)

    def save(self, path, optimizer) :
    """
    保存模型的状态和优化器的状态。

    参数:
    - path: 保存文件的路径。
    - optimizer: 优化器对象。
    """
        torch.save({
            "model_state": self.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, path)

    def num_params(self, print_out=True):
    """
    计算并打印模型的可训练参数数量。

    参数:
    - print_out: 是否打印参数数量，默认为 True。
    """
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out :
            print('Trainable Parameters: %.3fM' % parameters)
