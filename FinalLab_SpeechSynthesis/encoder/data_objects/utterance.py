import numpy as np


class Utterance:
    def __init__(self, frames_fpath, wave_fpath):
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath

    #从文件中加载并返回语音帧数据
    def get_frames(self):
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames):
        """
        从完整的语音片段中随机裁剪出一个指定帧数的部分语音片段。
        
        参数:
            n_frames (int): 需要的部分语音片段的帧数。
        
        返回值:
            tuple: 第一个元素为包含指定帧数的部分语音片段的NumPy数组，第二个元素为一个元组，
                   指明部分语音片段在完整语音片段中的起始和结束帧的索引。
        """
        frames = self.get_frames()
        if frames.shape[0] == n_frames:
            start = 0
        else:
            start = np.random.randint(0, frames.shape[0] - n_frames)
        end = start + n_frames
        return frames[start:end], (start, end)
