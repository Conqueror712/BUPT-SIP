from vocoder.models.fatchord_version import WaveRNN
from vocoder import hparams as hp
import torch


_model = None   # type: WaveRNN

def load_model(weights_fpath, verbose=True):
    """
    加载 WaveRNN 模型及其权重。

    参数:
    - weights_fpath: 权重文件路径。
    - verbose: 是否输出详细信息，默认为 True。
    """
    global _model, _device
    
    if verbose:
        print("Building Wave-RNN")
    _model = WaveRNN(
        rnn_dims=hp.voc_rnn_dims,
        fc_dims=hp.voc_fc_dims,
        bits=hp.bits,
        pad=hp.voc_pad,
        upsample_factors=hp.voc_upsample_factors,
        feat_dims=hp.num_mels,
        compute_dims=hp.voc_compute_dims,
        res_out_dims=hp.voc_res_out_dims,
        res_blocks=hp.voc_res_blocks,
        hop_length=hp.hop_length,
        sample_rate=hp.sample_rate,
        mode=hp.voc_mode
    )

    if torch.cuda.is_available():
        _model = _model.cuda()
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
    
    if verbose:
        print("Loading model weights at %s" % weights_fpath)
    checkpoint = torch.load(weights_fpath, _device)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()

# 检查模型是否已经加载。
def is_loaded():
    return _model is not None


def infer_waveform(mel, normalize=True,  batched=True, target=8000, overlap=800, 
                   progress_callback=None):
    """
    从梅尔频谱图推断波形，格式必须与合成器的输出一致。

    参数:
    - mel: 梅尔频谱图。
    - normalize: 是否归一化梅尔频谱图，默认为 True。
    - batched: 是否批处理，默认为 True。
    - target: 目标长度，默认为 8000。
    - overlap: 重叠长度，默认为 800。
    - progress_callback: 进度回调函数，默认为 None。

    返回值:
    - 生成的波形。
    """
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")
    
    if normalize:
        mel = mel / hp.mel_max_abs_value
    mel = torch.from_numpy(mel[None, ...])
    wav = _model.generate(mel, batched, target, overlap, hp.mu_law, progress_callback)
    return wav
