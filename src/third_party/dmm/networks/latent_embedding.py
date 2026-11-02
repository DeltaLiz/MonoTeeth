import torch
from torch import nn

class LatentEmbedding(nn.Module):
    def __init__(self, label_list=None, num_scenes=-1, gum_latent_dim=10, tooth_latent_dim=10, std=0.002, **kwargs):
        super().__init__()

        # Initialize latent codes for teeth and gums
        self.latent_vecs = torch.nn.ModuleDict()
        for label_id in label_list:
            if label_id == 0:
                self.latent_vecs[str(label_id)] = torch.nn.Embedding(num_scenes, gum_latent_dim)
            else:
                self.latent_vecs[str(label_id)] = torch.nn.Embedding(num_scenes, tooth_latent_dim)

            torch.nn.init.normal_(self.latent_vecs[str(label_id)].weight, mean=0, std=std)


    def forward(self, label_id, scene_ids):
        return self.latent_vecs[str(label_id)](scene_ids)
