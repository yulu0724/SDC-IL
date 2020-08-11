from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb

__all__ = ['ResNet', 'resnet18','resnet18_cifar', 'resnet32','resnet34', 'resnet50', 'resnet101',
           'resnet152']

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resnet_cifar(nn.Module):

    def __init__(self, block, num_blocks,  pretrained=False, cut_at_pooling=False,
                 Embed_dim=0, norm=True, dropout=0, num_classes=0):
        super(resnet_cifar, self).__init__()
        self.Embed_dim = Embed_dim  #lu adds
        self.pretrained = pretrained
        self.in_planes = 16
        self.cut_at_pooling = cut_at_pooling
 
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.base_feature = nn.Sequential(self.conv1, self.bn1, self.relu, self.layer1, self.layer2, self.layer3)
        self.base = nn.Sequential()
        self.base.fc = nn.Linear(64, 1000)
        

        if not self.cut_at_pooling:
            self.num_features = Embed_dim
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = Embed_dim > 0
            self.num_classes = num_classes

            # Append new layers
            if self.has_embedding:
                self.base.fc = nn.Linear(64, self.num_features)
                init.kaiming_normal_(self.base.fc.weight, mode='fan_out')
                init.constant_(self.base.fc.bias, 0)
               
            self.num_features = 64
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

        self.Embed = Embedding(64, 1)
        if self.Embed_dim == 0: 
            pass
        else:
            self.Embed = Embedding(64, 1)

        if not self.pretrained:
            self.reset_params()
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        
        x = self.base_feature(x)
        if self.cut_at_pooling:
            return x
        
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x_feat = x

        if self.has_embedding:
            x = self.base.fc(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = F.relu(x)
            x = self.classifier(x)
        return x_feat, x
    
    def inference(self, x):
        x = self.base_feature(x)
        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.base.fc(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        
        if self.num_classes > 0:
            x = F.relu(x)

        return x 

    def forward_without_norm(self, x):
        x = self.base_feature(x)
        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.base.fc(x)
        if self.num_classes > 0:
            x = F.relu(x)
            x = self.classifier(x)
        
        return x
    
    def extract_feat(self, x):
        x = self.base_feature(x)
        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 Embed_dim=0, norm=True, dropout=0, num_classes=0):
        super(ResNet, self).__init__()
        self.Embed_dim = Embed_dim  #lu adds
        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = Embed_dim
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = Embed_dim > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.base.fc = nn.Linear(out_planes, self.num_features)
                init.kaiming_normal_(self.base.fc.weight, mode='fan_out')
                init.constant_(self.base.fc.bias, 0)
                # Change the num_features to CNN output channels
            self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

        self.Embed = Embedding(1024, 1)
        if self.Embed_dim == 0:  
            pass
        else:
            self.Embed = Embedding(1024, 1)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x_feat = x

        if self.has_embedding:
            x = self.base.fc(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = F.relu(x)
            x = self.classifier(x)
        return x_feat, x
    
    def inference(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.base.fc(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        
        if self.num_classes > 0:
            x = F.relu(x)

        return x 

    def forward_without_norm(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.base.fc(x)
        if self.num_classes > 0:
            x = F.relu(x)
            x = self.classifier(x)
        
        return x
    
    def extract_feat(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class Embedding(nn.Module): 
    def __init__(self, in_dim, out_dim, dropout=None, normalized=True):
        super(Embedding, self).__init__()
        self.bn = nn.BatchNorm2d(in_dim, eps=0.001)
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.dropout = dropout
        self.normalized = normalized

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        if self.dropout is not None:
            x = nn.Dropout(p=self.dropout)(x, inplace=True)
        x = self.linear(x)
        if self.normalized:
            norm = x.norm(dim=1, p=2, keepdim=True)
            x = x.div(norm.expand_as(x))
        return x


def resnet18(**kwargs):
    return ResNet(18, **kwargs)

def resnet32(**kwargs):
    return resnet_cifar(BasicBlock, [5, 5, 5], **kwargs)

def resnet34(**kwargs):
    return ResNet(34, **kwargs)

def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
