from encoder.params_data import *
from encoder.model import SpeakerEncoder
from encoder.audio import preprocess_wav   # We want to expose this function from here
from matplotlib import cm
from encoder import audio
from pathlib import Path
import numpy as np
import torch

_model = None # type: SpeakerEncoder
_device = None # type: torch.device


def load_model(weights_fpath: Path, device=None):
    """
    加载内存中的模型。如果不显式调用此函数，将在第一次调用 embed_frames() 时使用默认权重文件自动运行。

    参数:
        weights_fpath (Path): 保存模型权重的文件路径。
        device (torch.device 或 str, 可选): 模型将在此设备上加载并运行，输出总在 cpu 上。如果为 None，则默认使用 GPU（如果可用），否则使用 CPU。
    """
    # TODO: I think the slow loading of the encoder might have something to do with the device it
    #   was saved on. Worth investigating.
    global _model, _device
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        _device = torch.device(device)
    _model = SpeakerEncoder(_device, torch.device("cpu"))
    checkpoint = torch.load(weights_fpath, _device)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath.name, checkpoint["step"]))


def is_loaded():
    return _model is not None


def embed_frames_batch(frames_batch):
    """
    计算一批 Mel 频谱图的嵌入。

    参数:
        frames_batch (numpy.ndarray): 形状为 (batch_size, n_frames, n_channels) 的一批 Mel 频谱图。

    返回:
        numpy.ndarray: 形状为 (batch_size, model_embedding_size) 的嵌入数组。
    """
    if _model is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")

    frames = torch.from_numpy(frames_batch).to(_device)
    embed = _model.forward(frames).detach().cpu().numpy()
    return embed


def compute_partial_slices(n_samples, partial_utterance_n_frames=partials_n_frames,
                           min_pad_coverage=0.75, overlap=0.5):
     """
    计算如何切分波形及其对应的 Mel 频谱图，以获得每个包含指定帧数的部分语音。返回的范围可能会超出波形长度，
    建议将波形补零至 wave_slices[-1].stop。

    参数:
        n_samples (int): 波形中的样本数。
        partial_utterance_n_frames (int): 每个部分语音的 Mel 频谱图帧数。
        min_pad_coverage (float): 当最后一个部分语音帧数不足时，如果至少有 min_pad_coverage 比例的帧，
                                  则认为存在该部分语音，否则将其丢弃。
        overlap (float): 部分语音的重叠比例，0 表示完全不重叠。

    返回:
        tuple: 包含波形切片和 Mel 频谱图切片的列表，可用这些切片分别索引波形和 Mel 频谱图获取部分语音。
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1

    samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices


def embed_utterance(wav, using_partials=True, return_partials=False, **kwargs):
 """
    计算单个语音的嵌入。

    参数:
        wav (numpy.ndarray): 预处理后的语音波形。
        using_partials (bool): 如果为 True，则将语音分割成部分语音并计算其嵌入的归一化平均值。
        return_partials (bool): 如果为 True，则返回部分嵌入和对应的波形切片。
        kwargs: compute_partial_splits() 的额外参数。

    返回:
        numpy.ndarray: 形状为 (model_embedding_size,) 的嵌入数组。如果 return_partials 为 True，
                       同时返回形状为 (n_partials, model_embedding_size) 的部分嵌入数组和波形切片列表。
    """
    # TODO: handle multiple wavs to benefit from batching on GPU
    # Process the entire utterance if not using partials
    if not using_partials:
        frames = audio.wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed

    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

    # Split the utterance into partials
    frames = audio.wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(frames_batch)

    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed


def embed_speaker(wavs, **kwargs):
    raise NotImplemented()


def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    if shape is None:
        height = int(np.sqrt(len(embed)))
        shape = (height, -1)
    embed = embed.reshape(shape)

    cmap = cm.get_cmap()
    mappable = ax.imshow(embed, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(*color_range)

    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title)
