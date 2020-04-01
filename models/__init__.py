from .inception import inception_v3
from .BN_Inception import BNInception
from .resnet import resnet101
from .resnet import resnet50
from .resnet import resnet34
from .resnet import resnet32
from .resnet import resnet18
from .VGG import vgg16
from .alexnet import alexnet

__factory = {
    'bn': BNInception,
    'inception': inception_v3,
    'resnet101': resnet101,
    'resnet50': resnet50,
    'resnet34': resnet34,
    'resnet32': resnet32,
    'resnet18': resnet18,
    'vgg16': vgg16,
    'alexnet': alexnet,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown network:", name)
    return __factory[name](*args, **kwargs)
