import torch
import torch.nn as nn
import torch.nn.functional as f


class AttentionNet(nn.Module):
    """
    This network computes the attention - softmax over cosine similarities between the support
    set and a target image. The output is the label computed as the max pdf over
    all the labels in the support set (referenced as y_hat in the literature)

    The inputs in the forward pass are the embeddings computed previously
    (referenced as f and g in the literature)
    """
    def __init__(self):
        super(AttentionNet, self).__init__()

        self.layer = nn.Softmax()

    def forward(self, support_sets, support_set_labels, target_images):
        """
        Compute the forward pass
        :param support_sets: [batch_size, k, D] dimensional tensors
        :param support_set_labels: [batch_size, k, num_classes] dimensional tensors, one-hot encoded labels
        :param target_images: [batch_size, D] dimensional tensors
        :return: [batch_size, 1] dimensional tensors as the pdf for target images
        """

        # Compute the cosine similarity by dot product of two L2 normalized vectors
        support_sets_norm = f.normalize(support_sets, p=2, dim=2)
        target_images_norm = f.normalize(target_images, p=2, dim=1).unsqueeze(dim=1).permute(0, 2, 1)
        similarities = torch.bmm(support_sets_norm, target_images_norm).squeeze(dim=2)

        # Compute the softmax and distribution over all classes
        softmax_pdf = self.layer(similarities).unsqueeze(dim=1)
        prediction_pdf = torch.bmm(softmax_pdf, support_set_labels)

        _, predictions = torch.max(prediction_pdf, dim=2)
        return predictions
