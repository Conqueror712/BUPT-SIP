import math
import numpy as np
import librosa
import vocoder.hparams as hp
from scipy.signal import lfilter
import soundfile as sf

# 将标签转换为浮点数
def label_2_float(x, bits) :
    return 2 * x / (2**bits - 1.) - 1.

# 将浮点数转换为标签
def float_2_label(x, bits) :
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)

# 加载音频文件
def load_wav(path) :
    return librosa.load(str(path), sr=hp.sample_rate)[0]

# 保存音频文件
def save_wav(x, path) :
    sf.write(path, x.astype(np.float32), hp.sample_rate)

# 将信号分为粗略和精细部分
def split_signal(x) :
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine

# 合并粗略和精细部分为一个信号
def combine_signal(coarse, fine) :
    return coarse * 256 + fine - 2**15

# 将浮点数编码为16位整数
def encode_16bits(x) :
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)


mel_basis = None

# 将线性谱转换为梅尔谱
def linear_to_mel(spectrogram):
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.dot(mel_basis, spectrogram)

# 构建梅尔滤波器组
def build_mel_basis():
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)

# 归一化频谱
def normalize(S):
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)

# 反归一化频谱
def denormalize(S):
    return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db

# 将幅度谱转换为分贝谱
def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

# 将分贝谱转换为幅度谱
def db_to_amp(x):
    return np.power(10.0, x * 0.05)

# 计算音频的频谱图
def spectrogram(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - hp.ref_level_db
    return normalize(S)

# 计算音频的梅尔频谱图
def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)

# 计算短时傅里叶变换
def stft(y):
    return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)

# 对信号应用预加重滤波
def pre_emphasis(x):
    return lfilter([1, -hp.preemphasis], [1], x)

# 对信号应用去加重滤波
def de_emphasis(x):
    return lfilter([1], [1, -hp.preemphasis], x)

# 使用μ律压缩编码信号
def encode_mu_law(x, mu) :
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)

# 解码μ律压缩信号
def decode_mu_law(y, mu, from_labels=True) :
    if from_labels: 
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x

