import numpy as np
import scipy.io.wavfile as wav
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import IPython
from scipy.signal import lfilter
from voice.Experiment_1_materials_student.speechlib import enframe, pitch_vad, FrameTimeC, melbankm
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np


def Nmfcc(x, fs, p, frameSize, inc, nfft=512, n_dct=12):
    """
    计算mfcc系数
    :param x: 输入信号
    :param fs: 采样率
    :param p: Mel滤波器组的个数
    :param frameSize: 分帧的每帧长度
    :param inc: 帧移
    :return: mfcc系数
    """
    # 预加重处理
    # x_preemphasized = # TODO 需要补充：输入x, 调用lfilter方法进行预加重
    preemph = 0.97
    #[1.0, -preemph]滤波器的分子系数，[1.0]滤波器的分母系数，x这是原始音频信号
    x_preemphasized = lfilter([1.0, -preemph], [1.0], x)   #平衡频谱，提升高频部分。这里使用了一阶高通滤波器。


    # 分帧
    # frames = # TODO 需要补充：输入frames, 调用enframe方法进行分帧
    #enframe 函数将预加重后的信号分成若干帧。frameSize 是每一帧的长度，inc 是帧与帧之间的间隔。
    frames = enframe(x_preemphasized, frameSize, inc)  #

    # 加窗
    # frames = # TODO 需要补充：输入frames, 调用np.hanning和np.multiply方法进行加窗
    #以减少帧边界的不连续性。
    #创建了一个大小与音频帧相同的汉宁窗。汉宁窗是一种平滑窗口，其形状类似于半个余弦波。在帧的中间，窗值最大，在两端逐渐减小到零。
    window = np.hanning(frameSize)
    #np.multiply 是 Numpy 中用于执行逐元素乘法的函数。这行代码将每个音频帧与汉宁窗逐元素相乘
    frames = np.multiply(frames, window)

    # 计算FFT
    # fft_result = # TODO 需要补充：输入frames, 调用np.fft.rfft方法进行FFT
    #对每帧应用快速傅里叶变换（FFT）。np.fft.rfft 用于计算实数输入的FFT。
    fft_result = np.fft.rfft(frames, n=nfft)

    # 计算能量谱
    power_spectrum = np.abs(fft_result) ** 2

    # 计算通过Mel滤波器的能量
    bank = melbankm(p, nfft, fs, 0, 0.5 * fs, 0)
    ss = np.matmul(power_spectrum, bank.T)

    # 计算DCT倒谱
    M = bank.shape[0]  # 滤波器个数
    m = np.array([i for i in range(M)])
    mfcc = np.zeros((ss.shape[0], n_dct))  # 初始化mfcc系数
    for n in range(n_dct):
        # mfcc[:, n] = # TODO 需要补充：输入M,m,ss, 调用np计算mfcc系数
        #mfcc[:, n]：表示计算得到的第n个MFCC系数，np.pi 是圆周率π。
# n 是当前计算的MFCC系数的序号。
# M 是Mel滤波器的数量。
# m 是一个从0到M-1的整数数组。
# m + 0.5 和 np.pi * n / M 的乘积是DCT的关键部分，用于在频率域内进行转换
        mfcc[:, n] = np.sum(ss * np.cos(np.pi * n / M * (m + 0.5)), axis=1)
    return mfcc


(framerate, wave_data) = wavfile.read("demo.wav")

# 参数设置
wlen = 256
inc = 128
num = 8
nfft = 256
n_dct = 24

# 归一化处理
x = wave_data / max(np.abs(wave_data))
time = np.arange(0, len(wave_data)) / framerate

# 绘制原始波形
plt.figure(1)
plt.subplot(411)
plt.plot(time, x, 'b')
plt.title("(a) Waveform")
plt.ylabel("Amplitude")
plt.xlabel("Time/s")

# 计算MFCC特征
ccc1 = librosa.feature.mfcc(y=x,
                            n_fft=wlen,
                            sr=framerate,
                            n_mfcc=24,
                            fmax=4000,
                            dct_type=2,
                            hop_length=inc,
                            win_length=wlen)
ccc2 = np.transpose(ccc1)

# 进行NMFCC计算
ccc1 = Nmfcc(x, framerate, num, wlen, inc, nfft, n_dct)
fn = ccc1.shape[0]
cn = ccc1.shape[1]
frameTime = FrameTimeC(fn, wlen, inc, framerate)

# # 计算语谱图
D = librosa.stft(x, n_fft=wlen, hop_length=inc)
D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
# 绘制语谱图
plt.subplot(412)
librosa.display.specshow(D_db, sr=framerate, hop_length=inc, x_axis='time', y_axis='linear')
plt.title("(b) Spectrogram")
plt.colorbar(format='%+2.0f dB')

# 绘制MFCC系数
plt.subplot(413)
plt.plot(frameTime, ccc1[:, 0:int(cn/2)])
plt.title("(c) MFCC Coefficients")
plt.ylabel("Amplitude")
plt.xlabel("Time/s")




# 绘制MFCC特征图
plt.subplot(414)
plt.imshow(ccc1, cmap='hot', interpolation='nearest', aspect='auto')
plt.xlabel('Frame')
plt.ylabel('MFCC Coefficient')
plt.title('(d) MFCC Features')
plt.colorbar(label='Magnitude')

# 调整子图间距
plt.subplots_adjust(hspace=2)

os.makedirs('figs', exist_ok=True) # 创建文件夹
plt.savefig('figs/mfcc.png') # 保存图片
plt.show()