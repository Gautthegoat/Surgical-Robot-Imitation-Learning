import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, ResNet34_Weights
import numpy as np

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################@
# PyTorch Modules

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
    
    def forward(self, x):
        scale = self.weight * (self.running_var + 1e-5).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

class KL_divergence(nn.Module):
    def __init__(self):
        super(KL_divergence, self).__init__()

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################@
# Functions and classes

def get_loss(loss_name):
    loss_name = loss_name.lower()
    if loss_name == 'l1':
        return nn.L1Loss()
    elif loss_name == 'l2' or loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == "kl_divergence" or loss_name == "kl" or loss_name == "kl_div" or loss_name == "kld":
        return KL_divergence()
    else:
        raise ValueError(f"Loss {loss_name} not found.")

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def reparameterize(mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.randn_like(std)
    return mu + eps*std

class ResNetFactory:
    def __init__(self, 
                 model_name='resnet18', 
                 pretrained=None, 
                 keep_last_layer=True, 
                 freeze_bn=False):
        """
        Args:
            model_name (str): The type of ResNet (e.g., 'resnet18', 'resnet34', 'resnet50').
            pretrained (str): The type of pre-trained weights to load, or None if no pre-trained weights are used.
            keep_last_layer (bool): Whether to keep the fully connected (FC) layer.
            freeze_bn (bool): Whether to freeze the batch normalization layers.
        """
        self.model_name = model_name
        self.pretrained = pretrained if pretrained is not None else None
        self.keep_last_layer = keep_last_layer
        self.freeze_bn = freeze_bn
        
        self.size_output = None
        if self.pretrained is None:
            self.model = self._get_resnet_model()
        elif self.pretrained == 'imagenet':
            self.model = self._get_resnet_model_imagenet()
        else:
            raise ValueError(f"Pretrained weights {self.pretrained} not supported.")
        
        if not self.keep_last_layer:
            self._configure_last_layer()

        if self.freeze_bn:
            self._freeze_batchnorm()

    def _get_resnet_model(self):
        # Load ResNet architecture from torchvision.models
        if self.model_name == 'resnet18':
            model = models.resnet18()
            self.size_output = 512
        elif self.model_name == 'resnet34':
            model = models.resnet34()
            self.size_output = 512
        elif self.model_name == 'resnet50':
            model = models.resnet50()
            self.size_output = 2048
        elif self.model_name == 'resnet101':
            model = models.resnet101()
            self.size_output = 2048
        elif self.model_name == 'resnet152':
            model = models.resnet152()
            self.size_output = 2048
        else:
            raise ValueError(f"Model {self.model_name} not supported.")

        return model
    
    def _get_resnet_model_imagenet(self):
        # Load ResNet architecture from torchvision.models with pre-trained weights
        if self.model_name == 'resnet18':
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.size_output = 512
        elif self.model_name == 'resnet34':
            model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.size_output = 512
        elif self.model_name == 'resnet50':
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.size_output = 2048
        elif self.model_name == 'resnet101':
            model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            self.size_output = 2048
        elif self.model_name == 'resnet152':
            model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
            self.size_output = 2048
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
        
        return model

    def _configure_last_layer(self):
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove the FC layer

    def _freeze_batchnorm(self):
        # Freeze all batch normalization layers
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()  # Set the batch norm layers to evaluation mode
                for param in module.parameters():
                    param.requires_grad = False