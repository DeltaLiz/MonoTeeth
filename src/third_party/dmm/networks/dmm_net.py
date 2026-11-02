

import torch
from torch import nn
from ..utils.math import compute_gradient, compute_jacobian, screw_axis_to_rt
from ..utils.loss import overall_Eikonal_loss
from .gum_net import GumNet
from .tooth_net import ToothNet


class DMMNet(nn.Module):
    def __init__(self, avg_centers=None, dental_type='upper', mlp_output_dim=5, **kwargs):
        super().__init__()
        # Define DMM-Net
        if dental_type == 'upper':
            self.label_list = [0, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]
        else:
            self.label_list = [0, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47]

        if avg_centers is not None:
            self.avg_centers = avg_centers[..., :3]

        self.components = torch.nn.ModuleDict()

        for label_id in self.label_list:
            if label_id == 0:
                self.components[str(label_id)] = GumNet(mlp_output_dim=5, **kwargs)
            else:
                if avg_centers is not None:
                    self.components[str(label_id)] = ToothNet(label_id=label_id, avg_center=self.avg_centers[label_id], mlp_output_dim=8, **kwargs)
                else:
                    self.components[str(label_id)] = ToothNet(label_id=label_id, mlp_output_dim=8, **kwargs)


    # for training
    def forward(self, latent_codes, scene_ids, surface_points, gt_labels, gt_normals, spatial_points, centers):
        # Get coordinates
        coords = torch.cat([surface_points, spatial_points], dim=-2).requires_grad_(True)

        # Collect all component indices that are involved in this batch, useful for acceleration
        batch_label_list = torch.unique(gt_labels).int().tolist()
        # # -1 means off-the-surface
        # if -1 in batch_label_list:
        #     batch_label_list.remove(-1)

        sdf_list = [None] * len(batch_label_list)
        blend_weight_list = [None] * len(batch_label_list)
        losses_list = [None] * len(batch_label_list)

        for i, label_id in enumerate(batch_label_list):
            embedding = latent_codes(label_id, scene_ids)

            pred_sdf, blending_weights, losses = self.components[str(label_id)](embedding, coords, gt_labels, gt_normals, centers[:, label_id, :])
            sdf_list[i] = pred_sdf
            blend_weight_list[i] = blending_weights
            losses_list[i] = losses

        # Collect segmentation indicators and normalize them
        predict_weights = torch.stack(blend_weight_list, dim=-1)
        predict_weights = torch.nn.functional.normalize(predict_weights, dim=-1, p=1)

        # Blend component-wise SDF values for a final SDF value
        sdf_final = torch.concat(sdf_list, dim=-1)
        sdf_final = torch.sum(sdf_final * predict_weights, dim=-1, keepdim=True)

        overall_losses = overall_Eikonal_loss(sdf_final, coords, gt_normals)

        overall_loss = 0.
        loss_record = dict()

        for sep_losses in losses_list:
            for loss_name, loss in sep_losses.items():
                loss_record.setdefault(loss_name, 0.)
                loss_record[loss_name] += loss.mean().item()
                overall_loss += loss.mean()

        for loss_name, loss in overall_losses.items():
            single_loss = loss.mean()
            loss_record["overall_" + loss_name] = single_loss.item()
            overall_loss += single_loss * 10.

        return sdf_final, overall_loss, loss_record


    def fit(self, latent_codes, label_list, surface_points, gt_normals, spatial_points):
        # Get coordinates
        coords = torch.cat([surface_points, spatial_points], dim=-2).requires_grad_(True)

        sdf_list = [None] * len(label_list)
        blend_weight_list = [None] * len(label_list)

        overall_loss = 0.

        for i, label_id in enumerate(label_list):
            embedding = latent_codes(label_id, torch.tensor([0], device=coords.device))
            # Compute the regularization loss on the latent code
            overall_loss += torch.mean(embedding ** 2) * 2e5

            pred_sdf, blending_weights = self.components[str(label_id)].infer(embedding, coords)
            sdf_list[i] = pred_sdf
            blend_weight_list[i] = blending_weights

        # Collect segmentation indicators and normalize them
        predict_weights = torch.stack(blend_weight_list, dim=-1)
        predict_weights = torch.nn.functional.normalize(predict_weights, dim=-1, p=1)

        # Blend component-wise SDF values for a final SDF value
        sdf_final = torch.concat(sdf_list, dim=-1)
        sdf_final = torch.sum(sdf_final * predict_weights, dim=-1, keepdim=True)

        overall_losses = overall_Eikonal_loss(sdf_final, coords, gt_normals)

        for loss_name, loss in overall_losses.items():
            single_loss = loss.mean()
            overall_loss += single_loss

        return sdf_final, overall_loss


    def infer(self, latent_codes, scene_id, label_list, coords, rgb=None):
        if len(coords.shape) == 2:
            coords = coords.unsqueeze(0)

        # Get coordinates
        coords = coords.requires_grad_(True)

        sdf_list = [None] * len(label_list)
        blend_weight_list = [None] * len(label_list)
        if rgb is not None:
            color_list = [None] * len(label_list)

        for i, label_id in enumerate(label_list):
            embedding = latent_codes(label_id, torch.tensor([scene_id], device=coords.device))

            pred_sdf, blending_weights = self.components[str(label_id)].infer(embedding, coords)
            sdf_list[i] = pred_sdf
            blend_weight_list[i] = blending_weights

            if rgb is not None:
                color_list[i] = rgb[label_id][None, :].expand(coords.shape[-2], -1)

        # Collect segmentation indicators and normalize them
        predict_weights = torch.stack(blend_weight_list, dim=-1)
        predict_weights = torch.nn.functional.normalize(predict_weights, dim=-1, p=1)

        # Blend component-wise SDF values for a final SDF value
        sdf_final = torch.concat(sdf_list, dim=-1)
        sdf_final = torch.sum(sdf_final * predict_weights, dim=-1, keepdim=True)

        if rgb is not None:
            color_final = torch.stack(color_list, dim=-1).to(coords.device)
            color_final = torch.sum(color_final * predict_weights[..., None, :], dim=-1).squeeze()
            return sdf_final, color_final

        return sdf_final
