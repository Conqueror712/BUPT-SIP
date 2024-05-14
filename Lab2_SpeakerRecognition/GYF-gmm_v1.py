import os
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

def load_data_from_json(data_json):
    with open(data_json, 'r') as f:
        data = json.load(f)
    return data


def load_audio_data(filepath):
    audio, _ = librosa.load(filepath, sr=None)
    return np.array(audio)  # 这里我们转换为 NumPy 数组类型，以便后续处理


def extract_features(audio):    # 提取音频特征（MFCC特征）
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    return mfccs.T  # 转置以满足 sklearn 的输入要求（查文档得知）


def train_gmm(features, num_components):
    gmm = GaussianMixture(n_components=num_components, covariance_type='diag')
    gmm.fit(features)
    return gmm
    

def test_gmm(gmm, features):
    predictions = gmm.predict(features)
    return predictions


def calculate_accuracy(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy


def visualize(audio, mfccs):
    # 可视化音频波形
    plt.figure(figsize=(10, 4))
    plt.plot(audio)
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

    # 可视化MFCC特征
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


def main():
    train_data = load_data_from_json('train_info.json')
    test_data = load_data_from_json('test_info.json')

    train_features = []
    train_labels = []
    for item in train_data:
        audio = load_audio_data(item['filepath'])
        features = extract_features(audio)
        train_features.append(features)
        train_labels.extend([item['speaker_id']] * features.shape[0])

    test_features = []
    test_labels = []
    for item in test_data:
        audio = load_audio_data(item['filepath'])
        features = extract_features(audio)
        test_features.append(features)
        test_labels.extend([item['speaker_id']] * features.shape[0])

    # 将特征展平，为了后续训练 GMM 模型
    train_features = np.vstack(train_features)
    test_features = np.vstack(test_features)

    # 训练 GMM 模型
    num_components = 8
    gmm = train_gmm(train_features, num_components)

    # 测试 GMM 模型
    train_predictions = test_gmm(gmm, train_features)
    test_predictions = test_gmm(gmm, test_features)

    sample_audio = load_audio_data(train_data[0]['filepath'])
    sample_features = extract_features(sample_audio)
    visualize(sample_audio, sample_features)

    joblib.dump(gmm, 'gmm_model.pkl')

main()