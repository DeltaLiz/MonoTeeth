import torch
from torch import nn
from ..utils.math import compute_gradient, compute_jacobian, screw_axis_to_rt
from ..utils.loss import component_loss
from .deform_net import DeformNet
from .mlp import MLP

class ToothNet(nn.Module):
    def __init__(self, label_id=-1, avg_center=None, mlp_output_dim=8, **kwargs):
        super().__init__()

        self.deform_net = DeformNet(latent_dim=kwargs['tooth_latent_dim'], mlp_output_dim=mlp_output_dim, **kwargs)
        self.ref_net = MLP(latent_dim=kwargs['ref_latent_dim'], **kwargs)

        self.label_id = label_id
        self.avg_center = avg_center
        self.bce = torch.nn.BCEWithLogitsLoss()


    def forward(self, latent_codes, input_coords, gt_labels, gt_normals, centers=None):
        tooth_existence = (gt_labels == self.label_id).any(-1)

        losses = dict()

        # Compute the regularization loss on the latent code
        losses['embedding'] = torch.mean(latent_codes[tooth_existence] ** 2) * 1e6

        # Deformed by the deform-net
        deform_model_output = self.deform_net(input_coords, latent_codes)

        screw_axis = deform_model_output[..., :6]
        screw_axis = screw_axis.view(-1, screw_axis.shape[-1])
        R, T = screw_axis_to_rt(screw_axis)
        coords_flatten = input_coords.view(-1, input_coords.shape[-1])
        ref_coords = (torch.bmm(R, coords_flatten[..., None]) + T).squeeze().view(input_coords.shape)
        displacement = ref_coords - input_coords

        # Blend weights and corrections for the overall SDF values
        blend_weights = deform_model_output[..., -1]
        correction = deform_model_output[..., -2:-1]

        losses['bce'] = self.bce(blend_weights[:, :gt_labels.shape[-1]], (gt_labels == self.label_id).float()) * 100

        # Use sigmoid to make sure the blend weights are positive
        blend_weights = torch.sigmoid(blend_weights)

        # Forward the ref-net for SDF values
        pred_sdf = self.ref_net(ref_coords)
        component_sdf = pred_sdf + correction

        # Deform centers
        # TODO: SHOULD Be Incorrect here.
        deform_model_center_output = self.deform_net(centers.unsqueeze(-2), latent_codes)

        center_screw_axis = deform_model_center_output[..., :6].view(-1, 6)

        # Compute rotation and translation from the screw axis
        R, T = screw_axis_to_rt(center_screw_axis)

        # Transform the center from original position to a new position in the shared reference coordinate
        center_ref_coords = (torch.bmm(R, centers[..., None]) + T).squeeze(-1)

        #valid_center_mask = (gt_labels==self.label_id).any(-1)
        losses['center'] = torch.abs(center_ref_coords[tooth_existence] - self.avg_center).mean()

        # Compute gradients for component-wise losses
        grad_template = compute_gradient(pred_sdf, ref_coords)
        grad_deform = compute_jacobian(displacement, input_coords)

        losses.update(component_loss(component_sdf, input_coords, correction, grad_deform, grad_template,
                                           gt_labels, gt_normals, self.label_id))

        return component_sdf, blend_weights, losses


    def infer(self, latent_codes, input_coords):

        # Deformed by the deform-net
        deform_model_output = self.deform_net(input_coords, latent_codes)

        screw_axis = deform_model_output[..., :6]
        screw_axis = screw_axis.view(-1, screw_axis.shape[-1])
        R, T = screw_axis_to_rt(screw_axis)
        coords_flatten = input_coords.view(-1, input_coords.shape[-1])
        ref_coords = (torch.bmm(R, coords_flatten[..., None]) + T).squeeze().view(input_coords.shape)

        # Blend weights and corrections for the overall SDF values
        blend_weights = deform_model_output[..., -1]
        correction = deform_model_output[..., -2:-1]

        # Use sigmoid to make sure the blend weights are positive
        blend_weights = torch.sigmoid(blend_weights)

        # Forward the ref-net for SDF values
        pred_sdf = self.ref_net(ref_coords)
        component_sdf = pred_sdf + correction

        return component_sdf, blend_weights