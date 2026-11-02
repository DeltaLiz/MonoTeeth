"""DDP training. Launch with: torchrun --standalone --nproc_per_node=<gpus> -m src.train [overrides]"""

import json
import os
import shutil
import sys
from itertools import islice
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset.monoteeth import build_loader
from .model.network import MonoTeethNet, build_network
from .util.misc import Averager, log, seed_everything, setup_logging, to_device


def train_epoch(model: DDP, network: MonoTeethNet, loader: DataLoader, device: torch.device,
                optimizer: torch.optim.Optimizer, cfg: DictConfig, epoch: int, step: int,
                writer: SummaryWriter | None) -> tuple[dict[str, float], int]:

    model.train()
    loader.sampler.set_epoch(epoch)
    num_steps = len(loader) if cfg.train.max_steps is None else min(len(loader), cfg.train.max_steps)
    avg = Averager()

    for i, batch in enumerate(islice(loader, num_steps), start=1):
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.amp):
            pred = model(batch["rgb"])

        losses = network.compute_loss(pred, batch, cfg.loss)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if cfg.optim.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(network.trainable_parameters(), cfg.optim.grad_clip)

        optimizer.step()
        step += 1

        # Accumulate epoch averages; rank 0 logs every log_every steps.
        avg.add(losses, batch["rgb"].shape[0])
        if writer is not None and (i % cfg.train.log_every == 0 or i == num_steps):
            log.info(f"epoch={epoch + 1} step={i}/{num_steps} loss={losses['loss'].item():.5f}")
            for k, v in losses.items():
                writer.add_scalar(f"train_step/{k}", v.item(), step)

    # Epoch means averaged over all ranks.
    return avg.result(reduce=True), step


@torch.no_grad()
def validate(network: MonoTeethNet, loader: DataLoader, device: torch.device, cfg: DictConfig, epoch: int,
             run_dir: Path, writer: SummaryWriter) -> dict[str, float]:

    network.eval()
    vis_dir = run_dir / "vis" / f"epoch_{epoch + 1:04d}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    avg, num_vis = Averager(), 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.amp):
            pred = network(batch["rgb"])

        avg.add({**network.compute_loss(pred, batch, cfg.loss), 
                 **network.compute_metrics(pred, batch)},
                 batch["rgb"].shape[0])

        for i in range(min(cfg.train.num_vis - num_vis, batch["rgb"].shape[0])):
            image = network.visualize(pred, batch, i)
            Image.fromarray(image).save(vis_dir / f"{batch['subject_id'][i]}_crop{int(batch['crop_index'][i])}.png")
            writer.add_image(f"vis/{num_vis}", image, epoch + 1, dataformats="HWC")
            num_vis += 1

    return avg.result()


def save_checkpoint(path: Path, network: MonoTeethNet, optimizer: torch.optim.Optimizer, epoch: int, step: int,
                    best: float) -> None:
    # Trainable weights, optimizer and progress; the config lives in run_dir/config.yaml.
    state = {
        "model": network.checkpoint_state(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "best": best,
    }
    torch.save(state, path)


@hydra.main(version_base="1.3", config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    # DDP init: one process per GPU, per-rank seed.
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    seed_everything(cfg.train.seed + rank)

    # Use rank 0's run_dir on all ranks (${now:...} may resolve differently).
    run_dir = [cfg.run_dir]
    dist.broadcast_object_list(run_dir, src=0)
    cfg.run_dir = run_dir[0]

    # Rank 0: refuse to overwrite an existing run, then create ckpt dir, config.yaml and TensorBoard writer.
    run_dir = Path(cfg.run_dir)
    ckpt_dir = run_dir / "ckpt"
    writer = None
    if rank == 0:
        if cfg.resume is None and (ckpt_dir / "last.pt").exists():
            raise FileExistsError(f"{run_dir} already has a run; set resume or a new name.")

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
        writer = SummaryWriter(str(run_dir / "tensorboard"))

    # Log to stdout and run_dir/train.log on every rank.
    setup_logging(run_dir / "train.log", rank)

    # Data: sharded train loader on all ranks, full validation loader on rank 0 only.
    train_loader = build_loader(cfg.dataset, "train", cfg.train.batch_size, cfg.train.num_workers, distributed=True)
    val_loader = build_loader(cfg.dataset, "test", cfg.train.batch_size, cfg.train.num_workers) if rank == 0 else None
    # Network, DDP wrapper and optimizer over trainable (non-encoder) parameters.
    network = build_network(cfg.model).to(device)
    network.decoder.act_ckpt = cfg.train.act_ckpt
    model = DDP(network, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(network.trainable_parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    # Optionally resume: weights always, optimizer and progress when present; everything else is the current config.
    start_epoch, step, best = 0, 0, float("inf")
    if cfg.resume is not None:
        ckpt = torch.load(cfg.resume, map_location="cpu", weights_only=True)
        network.load_checkpoint_state(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", start_epoch)
        step = ckpt.get("global_step", step)
        best = ckpt.get("best", best)
    
    # Run summary.
    if rank == 0:
        num_trainable = sum(p.numel() for p in network.trainable_parameters())
        log.info(f"run_dir={run_dir} world_size={world_size} train={len(train_loader.dataset)} "
                 f"val={len(val_loader.dataset)} batch/gpu={cfg.train.batch_size} trainable={num_trainable / 1e6:.2f}M "
                 f"start_epoch={start_epoch + 1}")

    # Epoch loop: all ranks train; rank 0 validates, logs and saves.
    for epoch in range(start_epoch, cfg.train.epochs):
        train_metrics, step = train_epoch(model, network, train_loader, device, optimizer, cfg, epoch, step, writer)
        if rank == 0:
            # Record epoch-averaged train metrics.
            record = {"epoch": epoch + 1, "step": step, "train": train_metrics}
            for k, v in train_metrics.items():
                writer.add_scalar(f"train/{k}", v, epoch + 1)

            # Validate every val_every epochs and at the end; track the best metric.
            improved = False
            if (epoch + 1) % cfg.train.val_every == 0 or epoch + 1 == cfg.train.epochs:
                record["val"] = validate(network, val_loader, device, cfg, epoch, run_dir, writer)
                for k, v in record["val"].items():
                    writer.add_scalar(f"val/{k}", v, epoch + 1)
                improved = record["val"][cfg.train.best_metric] < best
                best = min(best, record["val"][cfg.train.best_metric])

            # Flush TensorBoard and append this epoch to metrics.jsonl.
            writer.flush()
            with (run_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")

            # Checkpoints: last.pt every epoch, best.pt on improvement, epoch_XXXX.pt every ckpt_every.
            save_checkpoint(ckpt_dir / "last.pt", network, optimizer, epoch + 1, step, best)
            if improved:
                shutil.copyfile(ckpt_dir / "last.pt", ckpt_dir / "best.pt")

            if (epoch + 1) % cfg.train.ckpt_every == 0:
                shutil.copyfile(ckpt_dir / "last.pt", ckpt_dir / f"epoch_{epoch + 1:04d}.pt")

            # One-line epoch summary.
            val_text = f" val_{cfg.train.best_metric}={record['val'][cfg.train.best_metric]:.5f}" if "val" in record else ""
            log.info(f"epoch={epoch + 1} train_loss={train_metrics['loss']:.5f}{val_text}")

        # Keep ranks in sync while rank 0 logs and saves.
        dist.barrier()

    # Teardown.
    if writer is not None:
        writer.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
