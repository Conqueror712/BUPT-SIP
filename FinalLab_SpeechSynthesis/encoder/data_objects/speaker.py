from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.utterance import Utterance
from pathlib import Path

# Contains the set of utterances of a single speaker
class Speaker:
    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None
        
    def _load_utterances(self):
        """
        从磁盘加载所有语音片段，并创建一个RandomCycler实例用于随机抽取。
        
        该方法从根目录的'_sources.txt'文件中读取语音文件信息，然后加载这些文件作为Utterance实例。
        """
        with self.root.joinpath("_sources.txt").open("r") as sources_file:
            sources = [l.split(",") for l in sources_file]
        sources = {frames_fname: wave_fpath for frames_fname, wave_fpath in sources}
        self.utterances = [Utterance(self.root.joinpath(f), w) for f, w in sources.items()]
        self.utterance_cycler = RandomCycler(self.utterances)
               
    def random_partial(self, count, n_frames):
        """
        从磁盘中抽取一批随机的、独特的部分语音片段。
        
        参数:
            count (int): 需要抽取的部分语音片段的数量。
            n_frames (int): 每个部分语音片段的帧数。

        返回值:
            list: 返回一个元组列表，每个元组包含一个Utterance实例、部分语音的帧和该部分语音在完整语音中的范围。
        """
        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)

        a = [(u,) + u.random_partial(n_frames) for u in utterances]

        return a
