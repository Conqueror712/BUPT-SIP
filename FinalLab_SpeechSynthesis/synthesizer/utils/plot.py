import numpy as np

# 将给定的文本标题按单词数量分行
def split_title_line(title_text, max_words=5):
	seq = title_text.split()
	return "\n".join([" ".join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])


def plot_alignment(alignment, path, title=None, split_title=False, max_len=None):
    """
    绘制解码器的对齐矩阵并保存为图像文件。

    参数:
        alignment (ndarray): 对齐矩阵，通常表示注意力权重。
        path (str): 图像保存路径。
        title (str, 可选): 图像标题。
        split_title (bool, 可选): 是否将标题分行显示。
        max_len (int, 可选): 最大长度限制对齐矩阵的列。

    依赖:
        matplotlib: 用于创建和保存图像。
    """
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	if max_len is not None:
		alignment = alignment[:, :max_len]

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111)

	im = ax.imshow(
		alignment,
		aspect="auto",
		origin="lower",
		interpolation="none")
	fig.colorbar(im, ax=ax)
	xlabel = "Decoder timestep"

	if split_title:
		title = split_title_line(title)

	plt.xlabel(xlabel)
	plt.title(title)
	plt.ylabel("Encoder timestep")
	plt.tight_layout()
	plt.savefig(path, format="png")
	plt.close()


def plot_spectrogram(pred_spectrogram, path, title=None, split_title=False, target_spectrogram=None, max_len=None, auto_aspect=False):
    """
    绘制预测的声谱图，可选择同时绘制目标声谱图。

    参数:
        pred_spectrogram (ndarray): 预测的声谱图数据。
        path (str): 图像保存路径。
        title (str, 可选): 图像标题。
        split_title (bool, 可选): 是否将标题分行显示。
        target_spectrogram (ndarray, 可选): 目标声谱图数据，用于比较。
        max_len (int, 可选): 最大长度限制声谱图的列。
        auto_aspect (bool, 可选): 是否自动调整声谱图的长宽比。

    依赖:
        matplotlib: 用于创建和保存图像。
    """
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	if max_len is not None:
		target_spectrogram = target_spectrogram[:max_len]
		pred_spectrogram = pred_spectrogram[:max_len]

	if split_title:
		title = split_title_line(title)

	fig = plt.figure(figsize=(10, 8))
	# Set common labels
	fig.text(0.5, 0.18, title, horizontalalignment="center", fontsize=16)

	#target spectrogram subplot
	if target_spectrogram is not None:
		ax1 = fig.add_subplot(311)
		ax2 = fig.add_subplot(312)

		if auto_aspect:
			im = ax1.imshow(np.rot90(target_spectrogram), aspect="auto", interpolation="none")
		else:
			im = ax1.imshow(np.rot90(target_spectrogram), interpolation="none")
		ax1.set_title("Target Mel-Spectrogram")
		fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
		ax2.set_title("Predicted Mel-Spectrogram")
	else:
		ax2 = fig.add_subplot(211)

	if auto_aspect:
		im = ax2.imshow(np.rot90(pred_spectrogram), aspect="auto", interpolation="none")
	else:
		im = ax2.imshow(np.rot90(pred_spectrogram), interpolation="none")
	fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)

	plt.tight_layout()
	plt.savefig(path, format="png")
	plt.close()
