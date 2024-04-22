import numpy as np
import os
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from python_speech_features import mfcc
import matplotlib.pyplot as plt

TrainDir = "Dataset/TRAIN"
TestDir = "Dataset/TEST"

def load_dataset(directory):
    dataset = []
    labels = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                dataset.append(filepath)
                labels.append(os.path.basename(os.path.dirname(filepath)))  # 提取文件夹名作为标签
    return dataset, labels

def extract_features(dataset):
    features = []
    for filepath in dataset:
        y, sr = librosa.load(filepath, sr=None)
        mfccs = mfcc(signal=y, samplerate=sr, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, preemph=0.97)
        mfccs_mean = np.mean(mfccs, axis=0)
        features.append(mfccs_mean)
    return features

# 加载数据集
train_dataset, train_labels = load_dataset(TrainDir)
test_dataset, test_labels = load_dataset(TestDir)

# 提取特征
train_features = extract_features(train_dataset)
test_features = extract_features(test_dataset)

# 数据规范化
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# 训练Logistic回归模型
model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
model.fit(train_features_scaled, train_labels)

# 预测
test_predictions = model.predict(test_features_scaled)

# 计算准确率
accuracy = accuracy_score(test_labels, test_predictions)
print(f"Logistic Regression Accuracy: {accuracy:.2f}")

# 绘制混淆矩阵热图
cm = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=False, cmap='Blues')  
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.xticks([])
plt.yticks([])
plt.show()
