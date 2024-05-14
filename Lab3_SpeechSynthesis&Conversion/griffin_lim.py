import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

n_fft, hop_length, win_length = 720, 160, 720 # 尝试改变参数

def _griffin_lim(S, gl_iters):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))  # 随机生成一个与频谱图S形状相同的复数数组
    S_complex = np.abs(S).astype(np.complex128)  # 计算频谱图S的绝对值，然后将其转换为双精度复数类型 
    y = _istft(S_complex * angles)  # 使用istft转换会音频
    for i in range(gl_iters):
        angles = np.exp(1j * np.angle(_stft(y)))  # 对当前的音频信号y进行短时傅里叶变换（_stft(y)），然后取其相位
        y = _istft(S_complex * angles)  # 使用istft进行音频转换
    return y

def _stft(y):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)

input_path = "data/C7_2_y.wav" # 或者是自己录制的 demo.wav
# display(Audio(input_path))

AudioData, sr = librosa.load(input_path, sr=16000, mono=True)  # 使用文件采样率
D = librosa.stft(AudioData, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

gl_iters = 50  # 设置 griffin_lim 算法重复迭代的次数
ReAudio = _griffin_lim(D, gl_iters)

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(AudioData)
plt.xlabel('t/s')
plt.title('original signal')
plt.subplot(2, 1, 2)
plt.plot(ReAudio)
plt.xlabel('t/s')
plt.title('reconstructed signal')
plt.subplots_adjust(hspace=0.5)
plt.show()

# display(Audio(ReAudio, rate=16000))