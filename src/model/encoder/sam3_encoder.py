import torch
import torch.nn.functional as F
from torch import nn

from src.third_party.sam3.model.necks import Sam3DualViTDetNeck
from src.third_party.sam3.model.position_encoding import PositionEmbeddingSine
from src.third_party.sam3.model.vitdet import ViT

CKPT_PREFIX = "detector.backbone.vision_backbone."


class Sam3Encoder(nn.Module):
    def __init__(self, ckpt: str, image_size: int = 1008) -> None:
        super().__init__()
        self.image_size = image_size
        vit = ViT(
            img_size=image_size, pretrain_img_size=336, patch_size=14, embed_dim=1024, depth=32,
            num_heads=16, mlp_ratio=4.625, norm_layer="LayerNorm", drop_path_rate=0.1, qkv_bias=True,
            use_abs_pos=True, tile_abs_pos=True, global_att_blocks=(7, 15, 23, 31), rel_pos_blocks=(),
            use_rope=True, use_interp_rope=True, window_size=24, pretrain_use_cls_token=True,
            retain_cls_token=False, ln_pre=True, ln_post=False, return_interm_layers=False,
            bias_patch_embed=False,
        )
        self.neck = Sam3DualViTDetNeck(
            trunk=vit, position_encoding=PositionEmbeddingSine(num_pos_feats=256, normalize=True),
            d_model=256, scale_factors=[4.0, 2.0, 1.0], add_sam2_neck=False,
        )
        state = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
        state = state.get("model", state)
        own = self.neck.state_dict()
        self.neck.load_state_dict({key: state[CKPT_PREFIX + key] for key in own}, strict=True)

        # always frozen
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)  # frozen: drop_path and dropout stay off

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> dict:
        """
        Args:
            rgb: [B, 3, H, W] in [0, 1], any H, W (stretched to image_size x image_size).

        Returns:
            fpn: [B, 256, 288, 288], [B, 256, 144, 144], [B, 256, 72, 72].
            tokens, tokens_pos: [B, 72*72, 256] coarsest level and its sine positional encoding.
        """
        x = F.interpolate(rgb, size=(self.image_size, self.image_size), mode="bilinear",
                          align_corners=False, antialias=True)
        x = (x - 0.5) / 0.5
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            fpn, pos, _, _ = self.neck(x)
        fpn = [f.float() for f in fpn]
        return {
            "fpn": fpn,
            "tokens": fpn[-1].flatten(2).transpose(1, 2),
            "tokens_pos": pos[-1].float().flatten(2).transpose(1, 2),
        }
