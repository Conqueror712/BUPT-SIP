from scipy.ndimage.morphology import binary_dilation
from encoder.params_data import *
from pathlib import Path
from typing import Optional, Union
from warnings import warn
import numpy as np
import librosa
import struct

try:
    import webrtcvad
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad=None

int16_max = (2 ** 15) - 1


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None,
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = True):
    """
    对波形文件进行预处理，适用于语音编码器训练，调整波形的采样率，可选地标准化音量和修剪静音。

    参数:
        fpath_or_wav (Union[str, Path, np.ndarray]): 音频文件的路径或者作为numpy数组的波形。
        source_sr (Optional[int]): 如果波形已加载，其输入采样率。对于文件，将自动检测此参数用于重采样。
        normalize (Optional[bool]): 若为True，则标准化波形的音量。
        trim_silence (Optional[bool]): 若为True，则使用声音活动检测器修剪长时间静音。

    返回:
        np.ndarray: 预处理后的波形，以numpy数组形式。
    """

    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    

    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)
        
    if normalize:
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    if webrtcvad and trim_silence:
        wav = trim_long_silences(wav)
    
    return wav


def wav_to_mel_spectrogram(wav):
    """
    将预处理后的波形转换为Mel频谱图。

    参数:
        wav (np.ndarray): 预处理后的音频波形，为numpy数组。

    返回:
        np.ndarray: 从输入波形导出的Mel频谱图，转置后供编码器使用。
    """
    frames = librosa.feature.melspectrogram(
        wav,
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T


def trim_long_silences(wav):
    """
    使用语音活动检测器修剪波形中的长时间静音。

    参数:
        wav (np.ndarray): 原始音频波形，为numpy数组。

    返回:
        np.ndarray: 修剪静音后的波形，可能比输入更短。
    """
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    """
    将波形的音量标准化到目标分贝满标尺（dBFS）级别。

    参数:
        wav (np.ndarray): 音频波形，为numpy数组。
        target_dBFS (float): 目标音量的dBFS。
        increase_only (bool): 如果为True，仅增加音量。
        decrease_only (bool): 如果为True，仅减少音量。

    返回:
        np.ndarray: 音量已标准化的波形。
    """
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))
