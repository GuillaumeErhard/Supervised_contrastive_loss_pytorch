'''
Paper: https://arxiv.org/abs/1512.03385
Original code taken from: https://github.com/akamaster/pytorch_resnet_cifar10

Note that this code has been modified to have a contrastive approach. By design there are two added function.
The first forward_contrastive to use the encoder and projecter part and the forward while the forward for inference is
the encoder then a linear classifier.
The second is freeze_projection to freze weight learned on the encoder
'''

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetContrastive(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, contrastive_dimension=64):
        super(ResNetContrastive, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.contrastive_hidden_layer = nn.Linear(64, contrastive_dimension)
        self.contrastive_output_layer = nn.Linear(contrastive_dimension, contrastive_dimension)

        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def freeze_projection(self):
        self.conv1.requires_grad_(False)
        self.bn1.requires_grad_(False)
        self.layer1.requires_grad_(False)
        self.layer2.requires_grad_(False)
        self.layer3.requires_grad_(False)

    def _forward_impl_encoder(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = F.normalize(out, dim=1)

        return out

    def forward_constrative(self, x):
        # Implement from the encoder E to the projection network P
        x = self._forward_impl_encoder(x)

        x = self.contrastive_hidden_layer(x)
        x = F.relu(x)
        x = self.contrastive_output_layer(x)

        # Normalize to unit hypersphere
        x = F.normalize(x, dim=1)

        return x

    def forward(self, x):
        # Implement from the encoder to the decoder network
        x = self._forward_impl_encoder(x)
        return self.linear(x)


def resnet20_contrastive(num_classes=10):
    return ResNetContrastive(BasicBlock, [3, 3, 3], num_classes)


def resnet32_contrastive(num_classes=10):
    return ResNetContrastive(BasicBlock, [5, 5, 5], num_classes)


def resnet44_contrastive(num_classes=10):
    return ResNetContrastive(BasicBlock, [7, 7, 7], num_classes)


def resnet56_contrastive(num_classes=10):
    return ResNetContrastive(BasicBlock, [9, 9, 9], num_classes)


def resnet110_contrastive(num_classes=10):
    return ResNetContrastive(BasicBlock, [18, 18, 18], num_classes)


def resnet1202_contrastive(num_classes=10):
    return ResNetContrastive(BasicBlock, [200, 200, 200], num_classes)


def get_resnet_contrastive(name='resnet20', num_classes=10):
    """
    Get back a resnet model tailored for cifar datasets and dimension which has been modified for a contrastive approach

    :param name: str
    :param num_classes: int
    :return: torch.nn.Module
    """
    name_to_resnet = {'resnet20': resnet20_contrastive, 'resnet32': resnet32_contrastive,
                      'resnet44': resnet44_contrastive, 'resnet56': resnet56_contrastive,
                      'resnet110': resnet110_contrastive, 'resnet1202': resnet1202_contrastive}

    if name in name_to_resnet:
        return name_to_resnet[name](num_classes)
    else:
        raise ValueError('Model name %s not found'.format(name))
