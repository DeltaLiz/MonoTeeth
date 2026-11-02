import torch
from torch import nn
from ..utils.math import compute_gradient, compute_jacobian
from ..utils.loss import component_loss
from .deform_net import DeformNet
from .mlp import MLP


class GumNet(nn.Module):
    def __init__(self, mlp_output_dim=8, **kwargs):
        super().__init__()

        self.deform_net = DeformNet(latent_dim=kwargs['gum_latent_dim'], mlp_output_dim=mlp_output_dim, **kwargs)
        self.ref_net = MLP(latent_dim=kwargs['ref_latent_dim'], **kwargs)

        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, latent_codes, input_coords, gt_labels, gt_normals, centers=None):
        losses = dict()

        # Compute the regularization loss on the latent code
        losses['embedding'] = torch.mean(latent_codes ** 2) * 1e6

        # Deformed by the deform-net
        deform_model_output = self.deform_net(input_coords, latent_codes)

        displacement = deform_model_output[..., :3]
        ref_coords = input_coords + displacement

        # Blend weights and corrections for the overall SDF values
        blend_weights = deform_model_output[..., -1]
        correction = deform_model_output[..., -2:-1]

        losses['bce'] = self.bce(blend_weights[:, :gt_labels.shape[-1]], (gt_labels == 0).float()) * 100

        # Use sigmoid to make sure the blend weights are positive
        blend_weights = torch.sigmoid(blend_weights)

        # Forward the ref-net for SDF values
        pred_sdf = self.ref_net(ref_coords)
        component_sdf = pred_sdf + correction

        # Compute gradients for component-wise losses
        grad_template = compute_gradient(pred_sdf, ref_coords)
        grad_deform = compute_jacobian(displacement, input_coords)

        losses.update(component_loss(component_sdf, input_coords, correction, grad_deform, grad_template,
                                           gt_labels, gt_normals, 0))

        return component_sdf, blend_weights, losses

    def infer(self, latent_codes, input_coords):
        # Deformed by the deform-net
        deform_model_output = self.deform_net(input_coords, latent_codes)

        displacement = deform_model_output[..., :3]
        ref_coords = input_coords + displacement

        # Blend weights and corrections for the overall SDF values
        blend_weights = deform_model_output[..., -1]
        correction = deform_model_output[..., -2:-1]

        # Use sigmoid to make sure the blend weights are positive
        blend_weights = torch.sigmoid(blend_weights)

        # Forward the ref-net for SDF values
        pred_sdf = self.ref_net(ref_coords)
        component_sdf = pred_sdf + correction

        return component_sdf, blend_weights