from synthesizer.hparams import hparams as _syn_hp


# 音频设置------------------------------------------------------------------------
# 匹配语音合成器的值
sample_rate = _syn_hp.sample_rate
n_fft = _syn_hp.n_fft
num_mels = _syn_hp.num_mels
hop_length = _syn_hp.hop_size
win_length = _syn_hp.win_size
fmin = _syn_hp.fmin
min_level_db = _syn_hp.min_level_db
ref_level_db = _syn_hp.ref_level_db
mel_max_abs_value = _syn_hp.max_abs_value
preemphasis = _syn_hp.preemphasis
apply_preemphasis = _syn_hp.preemphasize

bits = 9                            # 信号的比特深度
mu_law = True                       # 如果使用原始比特在 hp.voc_mode 中，推荐抑制噪声
                                


# WAVERNN / 语音编码器  --------------------------------------------------------------------------------
voc_mode = 'RAW'                    # 可以是 'RAW' (在原始比特上使用 softmax) 或 'MOL' (从混合逻辑分布中采样)
voc_upsample_factors = (5, 5, 8)    # 正确分解 hop_length
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_batch_size = 100
voc_lr = 1e-4
voc_gen_at_checkpoint = 5           # 每个检查点生成的样本数量
voc_pad = 2                         # 这将填充输入，以便 resnet 可以看到比输入长度更宽的范围
voc_seq_len = hop_length * 5        # 必须是 hop_length 的倍数

# Generating / Synthesizing
voc_gen_batched = True              # 非常快（实时+）的单句批处理生成
voc_target = 8000                   # 每个批处理条目生成的目标样本数
voc_overlap = 400                   # 批处理之间交叉淡入淡出的样本数
