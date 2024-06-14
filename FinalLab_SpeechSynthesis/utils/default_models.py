import urllib.request
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError

from tqdm import tqdm


default_models = {
    "encoder": ("https://drive.google.com/uc?export=download&id=1q8mEGwCkFy23KZsinbuvdKAQLqNKbYf1", 17090379),
    "synthesizer": ("https://drive.google.com/u/0/uc?id=1EqFMIbvxffxtjiVrtykroF6_mUh-5Z3s&export=download&confirm=t", 370554559),
    "vocoder": ("https://drive.google.com/uc?export=download&id=1cf2NO6FtI0jDuy8AV3Xgn6leO6dHjIgu", 53845290),
}

#根据下载的字节数更新进度条的状态
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, target: Path, bar_pos=0):
    """
    从指定的 URL 下载文件到本地路径，并显示进度条。如果在下载过程中遇到 HTTP 错误，则下载会提前结束。

    参数：
    - url: 要下载的文件的 URL。
    - target: 本地存储文件的路径对象。
    - bar_pos: 进度条在多进度条列表中的位置，默认为0。

    返回值：
    - 无返回值。如果下载过程中出现 HTTP 错误，函数将提前返回。
    """
    target.parent.mkdir(exist_ok=True, parents=True)

    desc = f"Downloading {target.name}"
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc, position=bar_pos, leave=False) as t:
        try:
            urllib.request.urlretrieve(url, filename=target, reporthook=t.update_to)
        except HTTPError:
            return


def ensure_default_models(models_dir: Path):
    """
    确保指定目录中包含所有预设的模型文件。如果模型文件不存在或大小不正确，将会从指定的URL重新下载。

    参数：
    - models_dir: 存放模型文件的根目录的路径对象。

    返回值：
    - 无返回值。所有文件应当在函数结束时正确下载并存放于指定位置。
    """
    jobs = []
    for model_name, (url, size) in default_models.items():
        target_path = models_dir / "default" / f"{model_name}.pt"
        if target_path.exists():
            if target_path.stat().st_size != size:
                print(f"File {target_path} is not of expected size, redownloading...")
            else:
                continue

        thread = Thread(target=download, args=(url, target_path, len(jobs)))
        thread.start()
        jobs.append((thread, target_path, size))

    #运行并添加线程
    for thread, target_path, size in jobs:
        thread.join()

        assert target_path.exists() and target_path.stat().st_size == size, \
            f"Download for {target_path.name} failed. You may download models manually instead.\n" \
            f"https://drive.google.com/drive/folders/1fU6umc5uQAVR2udZdHX-lDgXYzTyqG_j"
