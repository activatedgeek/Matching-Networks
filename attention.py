import torch
import torch.nn as nn
import torch.nn.functional as f


class AttentionNet(nn.Module):
    """
    This network computes the softmax over cosine similarities between the support
    set and a target image. The inputs in the forward pass are the
    embeddings computed previously (referenced as f and g in the literature)
    """
    def __init__(self):
        super(AttentionNet, self).__init__()

        self.layer = nn.Softmax()

    def forward(self, support_sets, target_images):
        """
        Support sets will be [batch_size, k, 64] dimensional tensors and
        Target Images will be [batch_size, 64] dimensional tensors
        Similarities will be [batch_size, k, 1] dimensional tensors
        """

        # Compute the cosine similarity by dot product of two L2 normalized vectors
        support_sets_norm = f.normalize(support_sets, p=2, dim=2)
        target_images_norm = f.normalize(target_images, p=2, dim=1).unsqueeze(dim=1).permute(0, 2, 1)
        similarities = torch.bmm(support_sets_norm, target_images_norm).squeeze(dim=2)

        output = self.layer(similarities)
        return output
