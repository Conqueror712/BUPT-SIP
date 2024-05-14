import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.io import wavfile
from IPython.display import Audio, display
from librosa import lpc


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
        frameout[i, :] = x[indf[i]:indf[i] + nlen]
    if isinstance(win, list) or isinstance(win, np.ndarray):
        frameout = np.multiply(frameout, np.array(win))
    return frameout


def Filpframe_OverlapS(x, win, inc):
    """
    基于重叠存储法的信号还原函数
    :param x: 分帧数据
    :param win: 窗
    :param inc: 帧移
    :return:
    """
    nf, slen = x.shape
    nx = (nf - 1) * inc + slen
    frameout = np.zeros(nx)
    x = x / win
    for i in range(nf):
        frameout[slen + (i - 1) * inc:slen + i * inc] += x[i, slen - inc:]
    return frameout
    
    
def lpc_coeff(s, p):
    """
    :param s: 一帧数据
    :param p: 线性预测的阶数
    :return:
    """
    n = len(s)
    # 计算自相关函数
    Rp = np.zeros(p)
    for i in range(p):
        Rp[i] = np.sum(np.multiply(s[i + 1:n], s[:n - i - 1]))
    Rp0 = np.matmul(s, s.T)
    Ep = np.zeros((p, 1))
    k = np.zeros((p, 1))
    a = np.zeros((p, p))
    # 处理i=0的情况
    Ep0 = Rp0
    k[0] = Rp[0] / Rp0
    a[0, 0] = k[0]
    Ep[0] = (1 - k[0] * k[0]) * Ep0
    # i=1开始，递归计算
    if p > 1:
        for i in range(1, p):
            k[i] = (Rp[i] - np.sum(np.multiply(a[:i, i - 1], Rp[i - 1::-1]))) / Ep[i - 1]
            a[i, i] = k[i]
            Ep[i] = (1 - k[i] * k[i]) * Ep[i - 1]
            for j in range(i - 1, -1, -1):
                a[j, i] = a[j, i - 1] - k[i] * a[i - j - 1, i - 1]
    ar = np.zeros(p + 1)
    ar[0] = 1
    ar[1:] = -a[:, p - 1]
    G = np.sqrt(Ep[p - 1])
    return ar, G

input_path = "data/C7_1_y.wav" # 或者是自己录制的 demo.wav
display(Audio(input_path))

fs, data = wavfile.read(input_path)
data = data / (2 ** (16 - 1))

data -= np.mean(data)
data /= np.max(np.abs(data))
N = len(data)
time = [i / fs for i in range(N)]  # 设置时间
p = 10 # 线性预测的阶数 
wlen, inc = 320, 80 # 设置窗长、帧移
msoverlap = wlen - inc
y = enframe(data, wlen, inc)
fn = y.shape[0]
Acoef = np.zeros((y.shape[0], p + 1))
resid = np.zeros(y.shape)
synFrame = np.zeros(y.shape)

# 求每帧的LPC系数与预测误差
for i in range(fn):
    a = lpc(y[i, :], order=p) # 计算lpc的参数，调用函数
    Acoef[i, :] = a # 预测系数
    resid[i, :] = lfilter(a, [1], y[i, :])

# 语音合成
for i in range(fn):
    synFrame[i, :] = lfilter([1], a, resid[i, :]) # 使用 lfilter 进行合成

outspeech = Filpframe_OverlapS(synFrame, np.hamming(wlen), inc)

display(Audio(outspeech, rate=fs))


plt.subplot(2, 1, 1)
plt.plot(data / np.max(np.abs(data)), 'k')
plt.title('Original Signal')
plt.subplot(2, 1, 2)
plt.title('Reconstruct Signal-LPC and error')
plt.subplots_adjust(hspace=0.5)
plt.plot(outspeech / np.max(np.abs(outspeech)), 'c')

plt.show()