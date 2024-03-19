import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from speechlib import enframe, FrameTimeC, melbankm, idct, dct

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
    x_preemphasized = lfilter([1, -0.9375], [1], x)
    # TODO 需要补充：输入x, 调用lfilter方法进行预加重

    # 分帧
    frames = enframe(x_preemphasized, frameSize, inc)
    # TODO 需要补充：输入frames, 调用enframe方法进行分帧

    # 加窗
    frames = np.multiply(frames, np.hanning(frameSize))
    # TODO 需要补充：输入frames, 调用np.hanning和np.multiply方法进行加窗

    # 计算FFT
    fft_result = np.fft.rfft(frames)
    # TODO 需要补充：输入frames, 调用np.fft.rfft方法进行FFT

    # 计算能量谱
    power_spectrum = np.abs(fft_result) ** 2
    
    # 计算通过Mel滤波器的能量
    bank = melbankm(p, nfft, fs, 0, 0.5 * fs, 0)
    ss = np.matmul(power_spectrum, bank.T)
    
    # 计算DCT倒谱
    M = bank.shape[0] # 滤波器个数
    m = np.array([i for i in range(M)])
    mfcc = np.zeros((ss.shape[0], n_dct)) # 初始化mfcc系数
    for n in range(n_dct):
        mfcc[:, n] = np.sqrt(2 / M) * np.sum(np.multiply(np.log(ss), np.cos((2 * m - 1) * n * np.pi / 2 / M)), axis=1)
        # TODO 需要补充：输入M,m,ss, 调用np计算mfcc系数
    return mfcc

# 读取WAV文件
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