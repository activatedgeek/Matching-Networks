import torch
import torch.nn as nn
from embedding import EmbedNet
from attention import AttentionNet


class MatchingNet(nn.Module):
    """
    This is the Matching Network which first creates embeddings for all images in the
    support sets and the target sets. Then it uses those embeddings to use the attentional
    classifier
    """
    def __init__(self, num_classes, input_channels=1):
        super(MatchingNet, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels

        self.embed = EmbedNet(self.input_channels)
        self.attend = AttentionNet()

    def forward(self, support_sets, support_set_labels, target_images):
        """
        :param support_sets: [batch_size, k, H, W] dimensional tensor, contains support image sets
        :param support_set_labels: [batch_size, k, 1] dimensional tensor, contains support image labels
        :param target_images: [batch_size, H, W]
        :return:
        """
        batch_size, k, h, w = support_sets.size()
        support_embeddings = self.embed(support_sets.view(batch_size*k, h, w)).view(batch_size, k, -1)
        target_embeddings = self.embed(target_images)

        support_set_one_hot_labels = torch.FloatTensor(batch_size, k, self.num_classes).zero_()
        support_set_one_hot_labels.scatter_(2, support_set_labels, 1)

        attention_classify = self.attend(support_embeddings, support_set_one_hot_labels, target_embeddings)
        return attention_classify
