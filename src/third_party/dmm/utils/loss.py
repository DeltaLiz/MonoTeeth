import torch
import torch.nn.functional as F
from .math import compute_gradient


def component_loss(pred_sdf, coords, correction, grad_deform, grad_template,
                                           gt_labels, gt_normals, label_id):

    gradient = compute_gradient(pred_sdf, coords)

    pred_surface_sdf = pred_sdf[:, :gt_normals.shape[-2], :]
    pred_spatial_sdf = pred_sdf[:, gt_normals.shape[-2]:, :]
    #
    # surface_coords = coords[:, :gt_normals.shape[-2], :]
    # spatial_coords = coords[:, gt_normals.shape[-2]:, :]

    # indicate if a point is on the surface of the specific component
    component_mask = (gt_labels == label_id).unsqueeze(-1)
    sdf_constraint = torch.abs(pred_surface_sdf[component_mask]).mean()#torch.where(selection_mask, pred_sdf, torch.zeros_like(pred_sdf))

    # to avoid flipping
    #surface_constraint_mask = (~component_mask) & (pred_surface_sdf < -3e-2)
    sdf_constraint += torch.relu(-pred_surface_sdf[~component_mask]).mean()

    # Eikonal loss
    inter_constraint = torch.exp(-1e2 * torch.abs(pred_surface_sdf[~component_mask])).mean()
    inter_constraint += torch.exp(-1e2 * torch.abs(pred_spatial_sdf)).mean()

    surf_gradient = gradient[:, :gt_normals.shape[-2], :]
    normal_constraint = 1 - F.cosine_similarity(surf_gradient[component_mask.squeeze(-1)], gt_normals[component_mask.squeeze(-1)], dim=-1)

    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

    # minimal correction prior
    sdf_correct_constraint = torch.abs(correction)

    # deformation smoothness prior
    grad_deform_constraint = grad_deform.norm(dim=-1)

    # normal consistency prior
    surf_grad_template = grad_template[:, :gt_normals.shape[-2], :]
    grad_temp_constraint = 1 - F.cosine_similarity(surf_grad_template[component_mask.squeeze(-1)], gt_normals[component_mask.squeeze(-1)], dim=-1)

    return {'sdf': sdf_constraint * 3e3,
            'inter': inter_constraint.mean() * 1e2,
            'normal_constraint': normal_constraint.mean() * 1e2,
            'grad_constraint': grad_constraint.mean() * 5e1,
            'sdf_correct_constraint': sdf_correct_constraint.mean() * 5e2,
            'grad_deform_constraint': grad_deform_constraint.mean() * 5,
            'grad_temp_constraint': grad_temp_constraint.mean() * 1.,
            }


def overall_Eikonal_loss(pred_sdf, xyz, gt_normals):

    gradient = compute_gradient(pred_sdf, xyz)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = pred_sdf[:, :gt_normals.shape[-2], :]#torch.where(is_on_surface != 0, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.exp(-1e2 * torch.abs(pred_sdf[:, gt_normals.shape[-2]:, :]))#torch.where(is_on_surface != 0, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = 1 - F.cosine_similarity(gradient[:, :gt_normals.shape[-2], :], gt_normals, dim=-1)#torch.where(is_on_surface != 0, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
    #                                 torch.zeros_like(gradient[..., :1]))
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1