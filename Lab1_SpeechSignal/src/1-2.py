import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from speechlib import enframe, FrameTimeC, pitch_vad, findSegment, melbankm, idct, dct


def pitch_Corr(x, wnd, inc, T1, fs, miniL=10):
    """
    自相关法基音周期检测函数
    :param x: 语音信号
    :param wnd: 窗函数或窗长
    :param inc: 帧移
    :param T1: 门限
    :param fs: 采样率
    :param miniL: 语音段的最小帧数
    :return voiceseg, vsl, SF, Ef, period: 语音段的起始点和终止点，语音段数，语音段标记，非语音段标记，基音周期
    """
    y = enframe(x, wnd, inc)    # 调用enframe方法进行分帧
    fn = y.shape[0]
    if isinstance(wnd, int):
        wlen = wnd
    else:
        wlen = len(wnd)
    voiceseg, vsl, SF, Ef = pitch_vad(x, wnd, inc, T1, miniL)  # 语音分段
    lmin = fs // 500  # 基音周期的最小值
    lmax = fs // 60  # 基音周期的最大值
    period = np.zeros(fn)
    for i in range(vsl):    # 在所有语音段中
        ixb = voiceseg[i]['start'] # 语音段的起始帧
        ixd = voiceseg[i]['duration'] # 语音段的帧数
        for k in range(ixd):
            # TODO 需要补充：输入y, 调用np.correlate计算短时自相关,并找到最大值,返回自相关函数ru
            # TODO 需要补充：输入ru找到范围内短时自相关最大值的标号,将其作为基音周期的估值,并存入period
            ru = np.correlate(y[ixb + k, :], y[ixb + k, :], mode='full')   # 计算短时自相关
            # max_index = np.argmax(ru[len(ru)//2:]) + len(ru)//2      # 找到最大值
            # period[ixb + k] = max_index - len(ru)//2                # 存入period
            ru = ru[wlen:]
            tloc = np.argmax(ru[lmin:lmax])
            period[ixb + k] = tloc + lmin
    return voiceseg, vsl, SF, Ef, period


# 读取WAV文件
(fs, data) = wavfile.read('demo_3.wav')

# 去除直流偏移
data = data - np.mean(data)

# 幅值归一化
data = data / np.max(data)

# 分析参数
wlen = 320  # 分析窗口长度
inc = 80    # 连续窗口间隔
N = len(data)
time = [i / fs for i in range(N)]  # 时间向量
T1 = 0.05  # 用于基音校正的阈值

# 进行基音校正并检测有声段
voiceseg, vsl, SF, Ef, period = pitch_Corr(data, wlen, inc, T1, fs)

# 计算帧数
fn = len(SF)

# 计算帧时间位置
frameTime = FrameTimeC(fn, wlen, inc, fs)

# 设置子图布局
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.5)  # 调整垂直间距

# 绘制波形图
axs[0].plot(time, data)
axs[0].set_title('Waveform of speech')
axs[0].set_ylabel('Amplitude')
axs[0].set_xlabel('Time/s')

# 绘制自相关基音周期检测图
axs[1].plot(frameTime, period)
axs[1].set_title('Pitch Detection by Autocorrelation')
axs[1].set_ylabel('Period')
axs[1].set_xlabel('Time/s')

# 标记有声段
for i in range(vsl):
    nx1 = voiceseg[i]['start']
    nx2 = voiceseg[i]['end']

    # 在波形图上标记有声段
    axs[0].axvline(frameTime[nx1], np.min(data), np.max(data), color='blue', linestyle='--')
    axs[0].axvline(frameTime[nx2], np.min(data), np.max(data), color='red', linestyle='-')

    # 在自相关基音周期检测图上标记有声段
    axs[1].axvline(frameTime[nx1], np.min(period), np.max(period), color='blue', linestyle='--')
    axs[1].axvline(frameTime[nx2], np.min(period), np.max(period), color='red', linestyle='-')

axs[0].legend(['Waveform', 'Start', 'End'])
axs[1].legend(['Pitch', 'Start', 'End'])

os.makedirs('figs', exist_ok=True) # 创建文件夹
plt.savefig('figs/pitch.png') # 保存图片
plt.show()