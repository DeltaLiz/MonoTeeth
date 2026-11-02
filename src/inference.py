import argparse
import json
from pathlib import Path

import numpy as np
import skimage.measure
import torch
import trimesh
from omegaconf import OmegaConf
from PIL import Image

from .model.network import MonoTeethNet, build_network
from .third_party.dmm.networks.dmm_net import DMMNet
from .util.constants import JAWS


def load_network(config_path: Path, ckpt_path: Path, sam3_ckpt: Path, device: torch.device) -> MonoTeethNet:
    """Build the network from a training config.yaml and load the trained weights of a checkpoint.

    The frozen encoder is loaded from sam3_ckpt instead of the path recorded in the config; the decoder's SAM3
    initialization is skipped since the checkpoint overwrites it.
    """
    cfg = OmegaConf.load(config_path)
    cfg.model.encoder.ckpt = str(sam3_ckpt)
    cfg.model.decoder.init_ckpt = None
    network = build_network(cfg.model)
    network.load_checkpoint_state(torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"])
    return network.to(device).eval()


def load_image(path: Path) -> torch.Tensor:
    """RGB crop -> [3, H, W] float in [0, 1], as MonoTeethDataset feeds the network."""
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"))
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)


def load_dmm(dmm_dir: Path, jaw: str, device: torch.device) -> DMMNet:
    """Pretrained DMM of one jaw from <dmm_dir>/<jaw>/{args.json, model.pth}."""
    specs = json.loads((dmm_dir / jaw / "args.json").read_text())
    dmm = DMMNet(**specs)
    dmm.load_state_dict(torch.load(dmm_dir / jaw / "model.pth", map_location="cpu", weights_only=True))
    return dmm.to(device).eval()


@torch.inference_mode()
def latents_to_mesh(dmm: DMMNet, latents: torch.Tensor, part_ids: tuple[int, ...], res: int = 128,
                    bound: float = 0.75, chunk: int = 2 ** 18) -> tuple[np.ndarray, np.ndarray]:
    """Individual-SDF meshing: marching cubes on each part's own SDF, then the part meshes are concatenated."""
    device = latents.device
    axis = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).view(1, -1, 3)
    spacing = 2 * bound / (res - 1)
    vertices, faces, num_vertices = [], [], 0
    for part_id, latent in zip(part_ids, latents):
        component = dmm.components[str(part_id)]
        sdf = torch.cat([component.infer(latent[None], grid[:, i:i + chunk])[0]
                         for i in range(0, grid.shape[1], chunk)], dim=1)
        part_vertices, part_faces, _, _ = skimage.measure.marching_cubes(
            sdf.view(res, res, res).cpu().numpy(), level=0.0, spacing=(spacing,) * 3, method="lorensen")
        vertices.append(part_vertices - bound)
        faces.append(part_faces + num_vertices)
        num_vertices += len(part_vertices)
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-image inference: cropped mouth image -> upper.ply / lower.ply")
    parser.add_argument("--config", type=Path, required=True, help="run_dir/config.yaml written by src.train")
    parser.add_argument("--ckpt", type=Path, required=True, help="checkpoint, e.g. run_dir/ckpt/best.pt")
    parser.add_argument("--image", type=Path, required=True, help="mouth crop framed like the training crops")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sam3-ckpt", type=Path, default=Path("pretrained/sam3/sam3.pt"))
    parser.add_argument("--dmm-dir", type=Path, default=Path("pretrained/dmm"))
    parser.add_argument("--mesh-res", type=int, default=128, help="marching-cubes grid resolution per axis")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    network = load_network(args.config, args.ckpt, args.sam3_ckpt, device)
    with torch.inference_mode():
        latents = network(load_image(args.image)[None].to(device))["latent"][0].float()  # [30, D] in PARTS order
    del network

    args.out_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    for jaw, part_ids in JAWS:
        dmm = load_dmm(args.dmm_dir, jaw, device)
        vertices, faces = latents_to_mesh(dmm, latents[offset:offset + len(part_ids)], part_ids, res=args.mesh_res)
        trimesh.Trimesh(vertices, faces, process=False).export(args.out_dir / f"{jaw}.ply")
        print(f"{args.out_dir / f'{jaw}.ply'}: {len(vertices)} vertices, {len(faces)} faces", flush=True)
        offset += len(part_ids)
        del dmm


if __name__ == "__main__":
    main()
