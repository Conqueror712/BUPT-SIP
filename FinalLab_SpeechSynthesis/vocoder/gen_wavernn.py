from vocoder.models.fatchord_version import  WaveRNN
from vocoder.audio import *


def gen_testset(model: WaveRNN, test_set, samples, batched, target, overlap, save_path):

    """
    使用 WaveRNN 模型生成测试集音频样本并保存。

    参数:
    - model: WaveRNN 模型实例。
    - test_set: 测试集数据。
    - samples: 要生成的样本数量。
    - batched: 是否使用批处理。
    - target: 目标长度。
    - overlap: 重叠步数。
    - save_path: 保存生成音频文件的路径。

    """
    k = model.get_step() // 1000

    for i, (m, x) in enumerate(test_set, 1):
        if i > samples: 
            break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        bits = 16 if hp.voc_mode == 'MOL' else hp.bits

        if hp.mu_law and hp.voc_mode != 'MOL' :
            x = decode_mu_law(x, 2**bits, from_labels=True)
        else :
            x = label_2_float(x, bits)

        save_wav(x, save_path.joinpath("%dk_steps_%d_target.wav" % (k, i)))
        
        batch_str = "gen_batched_target%d_overlap%d" % (target, overlap) if batched else \
            "gen_not_batched"
        save_str = save_path.joinpath("%dk_steps_%d_%s.wav" % (k, i, batch_str))

        wav = model.generate(m, batched, target, overlap, hp.mu_law)
        save_wav(wav, save_str)

