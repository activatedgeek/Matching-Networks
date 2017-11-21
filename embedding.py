from functools import reduce
import torch.nn as nn


class EmbedNet(nn.Module):
    """
    This class implements the embedding CNN (referenced as f' in the literature)
    It takes in a 1x28x28 image as input and returns a 64x1 vector for each image.
    This happens via stacking 4 modules of:
        64 (3x3) Convolution Filters -> Batch Normalization -> ReLU -> Subsampling [-> Dropout]
    All the parameters are initialized via Glorot initialization
    """
    def __init__(self, input_channels, dropout_prob=0.1):
        super(EmbedNet, self).__init__()

        self.dropout_prob = dropout_prob

        self.module1 = self._create_module(input_channels, 64)
        self._init_weights(self.module1)

        self.module2 = self._create_module(64, 64)
        self._init_weights(self.module2)

        self.module3 = self._create_module(64, 64)
        self._init_weights(self.module3)

        self.module4 = self._create_module(64, 64)
        self._init_weights(self.module4)

    def forward(self, images):
        output = self.module1(images)
        output = self.module2(output)
        output = self.module3(output)
        output = self.module4(output)

        batch_size, *dims = output.size()
        flatten_size = reduce(lambda x,y: x*y, dims)

        output = output.view(batch_size, 1, flatten_size)
        return output

    def _create_module(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Dropout2d(self.dropout_prob)
        )

    def _init_weights(self, mod, nonlinearity='relu'):
        for m in mod:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight, gain=nn.init.calculate_gain(nonlinearity))
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
