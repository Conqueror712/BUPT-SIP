from __future__ import division
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter
# 绘图显示中文字体
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def enframe(x, win, inc=None):
    nx = len(x)
    if isinstance(win, list) or isinstance(win, np.ndarray):
        nwin = len(win)
        nlen = nwin  # 帧长=窗长
    elif isinstance(win, int):
        nwin = 1
        nlen = win  # 设置为帧长
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc
    frameout = np.zeros((nf, nlen))
    indf = np.multiply(inc, np.array([i for i in range(nf)]))
    for i in range(nf):
        frameout[i,:] = x[indf[i]:indf[i] + nlen]
    if isinstance(win, list) or isinstance(win, np.ndarray):
        frameout = np.multiply(frameout, np.array(win))
    return frameout

def FrameTimeC(frameNum, frameLen, inc, fs):
    ll = np.array([i for i in range(frameNum)])
    return ((ll - 1) * inc + frameLen / 2) / fs

def dct(x):
    N = len(x)
    X = np.zeros(N)
    ts = np.array([i for i in range(N)])
    C = np.ones(N)
    C[0] = np.sqrt(2) / 2
    for k in range(N):
        X[k] = np.sqrt(2 / N) * np.sum(C[k] * np.multiply(x, np.cos((2 * ts + 1) * k * np.pi / 2 / N)))
    return X


def idct(X):
    N = len(X)
    x = np.zeros(N)
    ts = np.array([i for i in range(N)])
    C = np.ones(N)
    C[0] = np.sqrt(2) / 2
    for n in range(N):
        x[n] = np.sqrt(2 / N) * np.sum(np.multiply(np.multiply(C[ts], X[ts]), np.cos((2 * n + 1) * np.pi * ts / 2 / N)))
    return x


def melbankm(p, NFFT, fs, fl, fh, w=None):
    """
    计算Mel滤波器组
    :param p: 滤波器个数
    :param NFFT: 一帧FFT后的数据长度
    :param fs: 采样率
    :param fl: 最低频率
    :param fh: 最高频率
    :param w: 窗(没有加窗，无效参数)
    :return:
    """
    bl = 1125 * np.log(1 + fl / 700)  # 把 Hz 变成 Mel
    bh = 1125 * np.log(1 + fh / 700)
    B = bh - bl  # Mel带宽
    y = np.linspace(0, B, p + 2)  # 将梅尔刻度等间隔
    Fb = 700 * (np.exp(y / 1125) - 1)  # 把 Mel 变成Hz
    W2 = int(NFFT / 2 + 1)
    df = fs / NFFT
    freq = [int(i * df) for i in range(W2)]  # 采样频率值
    bank = np.zeros((p, W2))
    for k in range(1, p + 1):
        f0, f1, f2 = Fb[k], Fb[k - 1], Fb[k + 1]
        n1 = np.floor(f1 / df)
        n2 = np.floor(f2 / df)
        n0 = np.floor(f0 / df)
        for i in range(1, W2):
            if n1 <= i <= n0:
                bank[k - 1, i] = (i - n1) / (n0 - n1)
            elif n0 < i <= n2:
                bank[k - 1, i] = (n2 - i) / (n2 - n0)
            elif i > n2:
                break
    return bank


def findSegment(express):
    """
    分割成語音段
    :param express:
    :return:
    """
    if express[0] == 0:
        voiceIndex = np.where(express)
    else:
        voiceIndex = express
    d_voice = np.where(np.diff(voiceIndex) > 1)[0]
    voiceseg = {}
    if len(d_voice) > 0:
        for i in range(len(d_voice) + 1):
            seg = {}
            if i == 0:
                st = voiceIndex[0]
                en = voiceIndex[d_voice[i]]
            elif i == len(d_voice):
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[-1]
            else:
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[d_voice[i]]
            seg['start'] = st
            seg['end'] = en
            seg['duration'] = en - st + 1
            voiceseg[i] = seg
    return voiceseg


def pitch_vad(x, wnd, inc, T1, miniL=10):
    """
    使用能熵比检测基音，实际上就是语音分段
    :param x: 语音信号
    :param wnd: 窗函数或窗长
    :param inc: 帧移
    :param T1: 端点检测阈值
    :param miniL: 语音段最小帧数
    :return:
    """
    y = enframe(x, wnd, inc)    # 语音分帧
    fn = y.shape[0]
    if isinstance(wnd, int):    # 判断是wnd是窗函数还是窗长，并以此设定窗长
        wlen = wnd
    else:
        wlen = len(wnd)

    Sp = np.abs(np.fft.fft(y, axis=1))  # FFT变换
    Sp = Sp[:, :wlen // 2 + 1]  # 取正频率部分
    Esum = np.sum(np.multiply(Sp, Sp), axis=1)  # 能量总和
    prob = Sp / np.sum(Sp, axis=1, keepdims=True)   # 概率密度
    H = -np.sum(np.multiply(prob, np.log10(prob + 1e-16)), axis=1)  # 短时熵谱
    H = np.where(H < 0.1, np.max(H), H)
    Ef = np.sqrt(1 + np.abs(Esum / H))  # 能熵比
    Ef = Ef / np.max(Ef)    # 归一化

    zseg = findSegment(np.where(Ef > T1)[0])    # 找到能熵比大于门限的语音帧
    zsl = len(zseg.keys()) # 计算帧长
    SF = np.zeros(fn)
    # 标记有声段、噪声段
    for k in range(zsl):
        if zseg[k]['duration'] < miniL:
            zseg.pop(k)
        else:
            SF[zseg[k]['start']:zseg[k]['end']] = 1
    return zseg, len(zseg.keys()), SF, Ef

