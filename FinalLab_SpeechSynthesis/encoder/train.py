from pathlib import Path

import torch

from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.model import SpeakerEncoder
from encoder.params_model import *
from encoder.visualizations import Visualizations
from utils.profiler import Profiler


def sync(device: torch.device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
          backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool):
    """
    训练语音编码器模型。

    参数:
        run_id (str): 训练运行的标识符。
        clean_data_root (Path): 清洗后数据的根目录路径。
        models_dir (Path): 保存模型的目录路径。
        umap_every (int): 每多少步计算并保存UMAP投影。
        save_every (int): 每多少步保存模型状态。
        backup_every (int): 每多少步备份模型。
        vis_every (int): 每多少步更新可视化。
        force_restart (bool): 是否强制从头开始训练，忽略任何现有的模型。
        visdom_server (str): Visdom服务器的地址。
        no_visdom (bool): 是否禁用Visdom可视化。

    功能:
        创建数据集和数据加载器，设置训练设备，初始化模型和优化器，执行训练循环，
        包括前向传播、损失计算、反向传播和参数更新，以及可视化和保存模型状态。
    """
    # 创建数据集和数据加载器
    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=4,
    )

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cpu")

    # 创建模型和优化器
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1

    # 配置模型文件路径
    model_dir = models_dir / run_id
    model_dir.mkdir(exist_ok=True, parents=True)
    state_fpath = model_dir / "encoder.pt"

    # 加载任何现有模型
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")
    model.train()

    # 初始化可视化环境
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    vis.log_dataset(dataset)
    vis.log_params()
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_implementation({"Device": device_name})

    # 训练循环
    profiler = Profiler(summarize_every=10, disabled=False)
    for step, speaker_batch in enumerate(loader, init_step):
        profiler.tick("Blocking, waiting for batch (threaded)")

        # 前向传播
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        profiler.tick("Data to %s" % device)
        embeds = model(inputs)
        sync(device)
        profiler.tick("Forward pass")
        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        sync(loss_device)
        profiler.tick("Loss")

        # 反向传播和优化
        model.zero_grad()
        loss.backward()
        profiler.tick("Backward pass")
        model.do_gradient_ops()
        optimizer.step()
        profiler.tick("Parameter update")

        # 更新可视化
        # learning_rate = optimizer.param_groups[0]["lr"]
        vis.update(loss.item(), eer, step)

        # UMAP投影和模型保存
        if umap_every != 0 and step % umap_every == 0:
            print("Drawing and saving projections (step %d)" % step)
            projection_fpath = model_dir / f"umap_{step:06d}.png"
            embeds = embeds.detach().cpu().numpy()
            vis.draw_projections(embeds, utterances_per_speaker, step, projection_fpath)
            vis.save()

        if save_every != 0 and step % save_every == 0:
            print("Saving the model (step %d)" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)

        if backup_every != 0 and step % backup_every == 0:
            print("Making a backup (step %d)" % step)
            backup_fpath = model_dir / f"encoder_{step:06d}.bak"
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, backup_fpath)

        profiler.tick("Extras (visualizations, saving)")
