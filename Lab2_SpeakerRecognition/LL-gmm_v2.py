import os
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score


def load_data_from_json(data_json):
    """从JSON文件中加载数据"""
    with open(data_json, 'r') as f:
        data = json.load(f)
    return data


def load_audio_data(filepath):
    """加载音频文件"""
    audio, _ = librosa.load(filepath, sr=None)
    return np.array(audio)


def extract_features(audio):
    """提取音频特征（MFCC特征）"""
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=20)
    return mfccs.T


def train_gmm_per_speaker(features_dict, num_components):
    """为每位说话人单独训练GMM模型
       为每个说话人的特征数据集训练一个独立的高斯混合模型（GMM）。
       无监督学习算法，用于模拟特征数据的分布。每个 GMM 存储了特定说话人的声音特征"""
    gmms = {}
    for speaker_id, features in features_dict.items():
        gmm = GaussianMixture(n_components=num_components, covariance_type='diag', random_state=42)
        gmm.fit(features)
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
        audio = load_audio_data(item['filepath'])
        features = extract_features(audio)
        train_features_dict[speaker_id].append(features)

    for speaker_id in train_features_dict:
        train_features_dict[speaker_id] = np.vstack(train_features_dict[speaker_id])

    # 处理测试数据
    for item in test_data:
        speaker_id = item['speaker_id']
        if speaker_id not in test_features_dict:
            test_features_dict[speaker_id] = []
        audio = load_audio_data(item['filepath'])
        features = extract_features(audio)
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
    sample_features = extract_features(sample_audio)
    visualize(sample_audio, sample_features)

    # 保存模型
    joblib.dump(gmms, 'gmms_model.pkl')


main()
