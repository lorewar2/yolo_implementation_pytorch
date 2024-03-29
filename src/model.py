import torch
import torch.nn as nn

class Custom_yolo (nn.Module):
    def __init__(self):
        # config for VOC dataset
        in_channels = 3
        split_size = 7
        num_boxes = 2
        num_classes = 20
        super(Custom_yolo, self).__init__()
        # create conv layers according to the architecture
        self.darknet = self.create_conv_layers(cnn_architecture_config, in_channels)
        # create 2 fc layers
        self.fcs = self.create_fcs(split_size, num_boxes, num_classes)

    def forward(self, x):
        # go through the darknet cnn network and fully connected layer
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture, in_channels):
        layers = []
        # go through the architecture and make the darknet layers
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0]  # tuple
                conv2 = x[1]  # tuple
                num_repeats = x[2]  # integer
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]
        return nn.Sequential(*layers)

    def create_fcs(self, split_size, num_boxes, num_classes):
        # create the specified fcs layers
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)),
        )

cnn_architecture_config = [
    # tuple = (kernel size, number of filters of output, stride, padding)
    (7, 64, 2, 3),
    "M",  # max-pooling 2x2 stride = 2
    (3, 192, 1, 1),
    "M",  # max-pooling 2x2 stride = 2
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",  # max-pooling 2x2 stride = 2
    # [tuple, tuple, repeat times]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",  # max-pooling 2x2 stride = 2
    # [tuple, tuple, repeat times]
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    # single cnn block
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x