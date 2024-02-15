import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
from segmentation_models_pytorch.utils import base
import math
import torch
import numpy as np
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import models

def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)
class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss

class custom_loss(base.Loss):
    def __init__(self, batch_size, beta):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.mse = nn.MSELoss()
        self.contrast = ContrastiveLoss(batch_size)
        self.psnr = PeakSignalNoiseRatio()
        # self.L1 = nn.L1Loss()
        self.beta = beta
    def forward(self, y_pr, y_gt, ft1=None, ft2=None):
        x=self.mse(y_pr, y_gt)
        y=1-self.ssim(y_pr, y_gt)
        z=self.contrast(ft1, ft2)
        p=(1 - self.psnr(y_pr, y_gt)/40)
#         l = self.L1(y_pr, y_gt)
#         print(x,y,z)
        return x +y/10+p/10 + (z*self.beta)
        # return x+p/10+z/100+y/10  OG
    
class custom_lossv(base.Loss):
    def __init__(self):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.mse = nn.MSELoss()
        # self.contrast = ContrastiveLoss(batch_size)
        self.psnr = PeakSignalNoiseRatio()
        # self.L1 = nn.L1Loss()
    def forward(self, y_pr, y_gt, ft1=None, ft2=None):
        x=self.mse(y_pr, y_gt)
        y=1-self.ssim(y_pr, y_gt)
        # z=self.contrast(ft1, ft2)
        p=(1 - self.psnr(y_pr, y_gt)/40)
        return x +y/10 + p/10
        # return x+p/10+y/10  OG

class lossX3_mse(base.Loss):
    def __init__(self):
        super().__init__()
        self.mse=nn.MSELoss()
    def forward(self,img1,img2,img3,gt):
        x=self.mse(img1,gt)
        y=self.mse(img2,gt)
        z=self.mse(img3,gt)

        return (1/7)*x+(2/7)*y+(4/7)*z

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_name_list, vgg_type='vgg19', use_input_norm=True, range_norm=False):
        super(VGGFeatureExtractor, self).__init__()
        self.range_norm = range_norm
        self.use_input_norm = use_input_norm
        if vgg_type == 'vgg19':
            self.vgg = models.vgg19(pretrained=True).features
        else:
            raise NotImplementedError(f'{vgg_type} is not supported yet.')

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.selected_layers = self.get_selected_layers(layer_name_list)

    def get_selected_layers(self, layer_name_list):
        selected_layers = []
        for name, layer in self.vgg._modules.items():
            if name in layer_name_list:
                selected_layers.append(layer)
            elif len(selected_layers) == len(layer_name_list):
                break
        return nn.Sequential(*selected_layers)

    def forward(self, x):
        if self.range_norm:
            # Normalize from [-1, 1] to [0, 1]
            x = (x + 1) / 2
        if self.use_input_norm:
            # Normalize using ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
            x = (x - mean) / std
        return self.selected_layers(x)


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(
                        x_features[k] - gt_features[k],
                        p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(
                        x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) -
                        self._gram_mat(gt_features[k]),
                        p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(
                        self._gram_mat(x_features[k]),
                        self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss
    


import torch.nn as nn

class lossX3_p(nn.Module):
    def __init__(self, layer_weights={'conv3_3' : 1}, vgg_type='vgg19', use_input_norm=False, range_norm=False, perceptual_weight=1.0, criterion='l1'):
        super().__init__()
        self.perceptual_loss = PerceptualLoss(
            layer_weights=layer_weights,
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
            perceptual_weight=perceptual_weight,
            criterion=criterion
        )

    def forward(self, img1, img2, img3, gt):
        percep_loss_img1, _ = self.perceptual_loss(img1, gt)
        percep_loss_img2, _ = self.perceptual_loss(img2, gt)
        percep_loss_img3, _ = self.perceptual_loss(img3, gt)

        total_loss = (1/7) * percep_loss_img1 + (2/7) * percep_loss_img2 + (4/7) * percep_loss_img3

        return total_loss

        