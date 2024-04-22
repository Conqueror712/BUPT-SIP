# 调库实现mfcc特征提取：extract_features(),准确率0.98

# 手动实现mfcc特征提取:extract_mfcc(),准确率0.92

# 调库实现lpcc特征提取lpcc(),准确率0.02（存在问题）

import json
import matplotlib.pyplot as plt
import joblib
import scipy
import scipy.fftpack
from sklearn.mixture import GaussianMixture
import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import librosa
import numpy as np
from scipy.signal import lfilter
from scipy.fftpack import fft, ifft



def load_data_from_json(data_json):
    """从JSON文件中加载数据"""
    with open(data_json, 'r') as f:
        data = json.load(f)
    return data


def load_audio_data(filepath):
    """加载音频文件"""
    audio, sr = librosa.load(filepath)
    # print(sr)
    return audio


def extract_mfcc(path):
    # 加载音频文件
    sample_rate, signal = scipy.io.wavfile.read(path)

    #预加重，提升高频部分，使信号的频谱变得平坦
    pre_emphasis = 0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    #分帧，由于语音信号的非平稳特性 和 短时平稳特性，将语音信号分分帧。
    frame_size = 0.025
    #帧移
    frame_stride = 0.01
    overlap = 0.015
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # 从秒转换为采样点
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    # 确保我们至少有1帧
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    # 填充信号，确保所有帧的采样数相等，而不从原始信号中截断任何采样
    pad_signal = numpy.append(emphasized_signal, z)
    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]

    #加窗，每个帧乘以一个窗函数
    frames *= numpy.hamming(frame_length)

    #FFT变换转换为频域上的能量分布
    NFFT=512
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))

    #功率谱，对语音信号的频谱取模平方，得到语音信号的谱线能量
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    #滤波器组，将功率谱通过一组Mel刻度来提取频带
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # 求最高hz频率对应的mel频率
    # 40个滤波器组，为此需要42个点，需要low_freq_mel和high_freq_mel之间线性间隔40个点
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 在mel频率上均分成42个点
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # 将mel频率再转到hz频率
    # bin = sample_rate/2 / NFFT/2=sample_rate/NFFT    # 每个频点的频率数
    # bins = hz_points/bin=hz_points*NFFT/ sample_rate    # hz_points对应第几个fft频点
    bins = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bins[m - 1])  # 左
        f_m = int(bins[m])  # 中
        f_m_plus = int(bins[m + 1])  # 右

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # 数值稳定性
    filter_banks = 20 * np.log10(filter_banks)  # dB
    filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)

    #mfcc，应用离散余弦变换（DCT）对滤波器组系数去相关处理
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # 保持在2-13
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
    return mfcc.T

def extract_features(audio):
    """提取音频特征（MFCC特征）"""
    #y:音频时间序列,sr:y的采样率,n_mfcc:要返回的MFCC数量
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=20)
    return mfccs


def lpcc(path, order):
    sample_rate, signal = scipy.io.wavfile.read(path)
    # Pre-emphasis
    pre_emphasis = 0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    # 分帧
    frame_size = 0.025
    frame_stride = 0.01
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z)
    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(
        numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]

    # 加窗
    frames *= numpy.hamming(frame_length)

    lpcc = np.zeros((num_frames, order))
    for i in range(num_frames):
        autocorr = lfilter([1], np.array([1, -pre_emphasis]), frames[i])
        r = autocorr[:order + 1]

        a = np.zeros(order + 1)
        k = np.zeros(order)
        a[0] = 1
        E = r[0]
        for m in range(1, order + 1):
            k[m - 1] = -np.sum(a[1:m] * r[m - 1:0:-1]) / E
            a[m] = k[m - 1]
            for j in range(1, m):
                a[j] += k[m - 1] * a[m - j]
            E *= 1 - k[m - 1] ** 2

        # 异常处理
        logE = np.log(E)
        if np.isnan(logE) or np.isinf(logE):
            # 当 logE 为 NaN 或无穷大时，将 lpcc 设置为全零向量
            lpcc[i] = np.zeros(order)
        else:
            lpcc[i] = fft(logE + ifft(np.log(np.abs(fft(a, 512)))[:order].real)).real

    return lpcc

def train_gmm_per_speaker(features_dict, num_components):
    """为每位说话人单独训练GMM模型
       无监督学习算法，用于模拟特征数据的分布。每个 GMM 存储了特定说话人的声音特征"""
    gmms = {}
    for speaker_id, features in features_dict.items():
        #diag 指每个分量有各自不同对角协方差矩阵（非对角为零，对角不为零）
        gmm = GaussianMixture(n_components=num_components, covariance_type='diag', random_state=42)
        gmm.fit(features)
        #训练好的 GMM 模型存储在 gmms 字典中，键是说话人的标识符，值是对应的 GMM 模型。
        gmms[speaker_id] = gmm
    return gmms


def test_gmm_per_speaker(gmms, test_features_dict):
    """测试各说话人的GMM模型，返回得分
       使用训练好的 GMM 针对测试数据集的特征进行打分。
       每个说话人模型返回一个得分，表示该模型如何适配测试特征
    """
    speaker_scores = {}
    for speaker_id, features in test_features_dict.items():
        gmm = gmms[speaker_id]
        #使用对应的 GMM 模型来计算测试声音特征的得分,score_samples 计算每个样本的加权对数概率，这些值可以用作样本与模型的拟合度量。
        scores = gmm.score_samples(features)
        speaker_scores[speaker_id] = scores
    return speaker_scores


def calculate_accuracy(test_data, gmms, test_features_dict):
    """计算准确率，使用总得分来确定最可能的说话人"""
    correct_predictions = 0
    total_predictions = 0

    for item in test_data:
        speaker_id = item['speaker_id']
        features = test_features_dict[speaker_id]

        # 计算每个模型的得分总和
        individual_scores = {sid: gmm.score_samples(features).sum() for sid, gmm in gmms.items()}

        # 找出得分最高的说话人ID
        predicted_speaker = max(individual_scores, key=individual_scores.get)

        if predicted_speaker == speaker_id:
            correct_predictions += 1

        total_predictions += 1

    return correct_predictions / total_predictions if total_predictions > 0 else 0


def visualize(audio, mfccs):
    """可视化音频波形和MFCC特征"""
    plt.figure(figsize=(10, 4))
    plt.plot(audio)
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    train_data = load_data_from_json('train_info.json')
    test_data = load_data_from_json('test_info.json')

    train_features_dict = {}
    test_features_dict = {}

    # 处理训练数据
    for item in train_data:
        speaker_id = item['speaker_id']
        if speaker_id not in train_features_dict:
            train_features_dict[speaker_id] = []

        # 调库实现mfcc特征提取：
        # audio = load_audio_data(item['filepath'])
        # features = extract_features(audio)

        # 手动实现mfcc特征提取
        # features = extract_mfcc(item['filepath'])

        #调库实现lpcc特征提取
        features = lpcc(item['filepath'],12)
        #将提取的特征添加到对应说话人的特征列表中。
        train_features_dict[speaker_id].append(features)



    for speaker_id in train_features_dict:
        #将每个说话人的特征列表堆叠成一个二维数组，其中每一行是一个样本的特征向量
        train_features_dict[speaker_id] = np.vstack(train_features_dict[speaker_id])

    # 处理测试数据
    for item in test_data:
        speaker_id = item['speaker_id']
        if speaker_id not in test_features_dict:
            test_features_dict[speaker_id] = []

        #调库实现mfcc特征提取：
        # audio = load_audio_data(item['filepath'])
        # features = extract_features(audio)

        #手动实现mfcc特征提取
        # features = extract_mfcc(item['filepath'])

        #调库实现lpcc特征提取
        features = lpcc(item['filepath'],12)

        test_features_dict[speaker_id].append(features)

    for speaker_id in test_features_dict:
        test_features_dict[speaker_id] = np.vstack(test_features_dict[speaker_id])

    # 训练GMM模型
    num_components = 8
    gmms = train_gmm_per_speaker(train_features_dict, num_components)

    # 计算测试准确率
    test_accuracy = calculate_accuracy(test_data, gmms, test_features_dict)
    print("Test Accuracy:", test_accuracy)

    # 可视化示例音频
    sample_audio = load_audio_data(train_data[0]['filepath'])

    # 调库实现mfcc特征提取：
    # sample_features = extract_features(sample_audio)

    # 手动实现mfcc特征提取
    # sample_features = extract_mfcc(train_data[0]['filepath'])

    # 调库实现lpcc特征提取
    sample_features = lpcc(train_data[0]['filepath'],12)

    visualize(sample_audio, sample_features)

    # 保存模型
    joblib.dump(gmms, 'gmms_model.pkl')

main()
