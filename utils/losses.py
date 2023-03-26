
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp
from torchvision import models

##############
# other losses
##############

def get_soft_label(input_tensor, num_class, data_type = 'float'):
    """
        convert a label tensor to soft label 
        input_tensor: tensor with shae [B, 1, D, H, W]
        output_tensor: shape [B, num_class, D, H, W]
    """
    tensor_list = []
    for i in range(num_class):
        temp_prob = input_tensor == i*torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim = 1)
    if(data_type == 'float'):
        output_tensor = output_tensor.float()
    elif(data_type == 'double'):
        output_tensor = output_tensor.double()
    else:
        raise ValueError("data type can only be float and double: {0:}".format(data_type))

    return output_tensor

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, cls_num, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.cls_num = cls_num
    def forward(self, pred, soft_y):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w) or (N,C,H,W)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)
        
        if len(soft_y.shape) == 4:
            one_hot_gt = get_soft_label(soft_y, self.cls_num).cuda()
        else:    
            # one-hot vector of ground truth
            one_hot_gt = get_soft_label(soft_y.unsqueeze(1), self.cls_num).cuda()

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        # print(BF1.shape)
        loss = torch.mean(1 - BF1)
        # loss = torch.sum(1.0 - BF1, dim=1)
        return loss


class LossNet(nn.Module):
    """
    Automatic Polyp Segmentation via Multi-scale Subtraction Network
    """
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(pretrained=True).cuda()
        for param in resnet.parameters():
            param.requires_grad = False
        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()
        self.layer4.eval()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        
        with torch.no_grad():
            pred_msk = torch.argmax(pred, dim=1)
            pred_msk_ = torch.cat((pred_msk[:,None,:,:], pred_msk[:,None,:,:], pred_msk[:,None,:,:]),dim=1).float()
            target_ = torch.cat((target[:,None,:,:], target[:,None,:,:], target[:,None,:,:]), dim=1).float()
        
        loss = torch.tensor(0.0).cuda()
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            pred_f = layer(pred_msk_)
            targ_f = layer(target_)
            loss += self.mse(pred_f, targ_f)

            pred_msk_ = pred_f
            target_ = targ_f

        return loss


def dice_loss(input: torch.FloatTensor, target: torch.LongTensor, weights: torch.FloatTensor = None, k: int = 0, eps: float = 0.0001):
    """
    Returns the Generalized Dice Loss Coefficient associated to the input and target tensors, as well as to the input weights,\
    in case they are specified.
    Args:
        input (torch.FloatTensor): CHW tensor containing the classes predicted for each pixel.
        target (torch.LongTensor): CHW one-hot encoded tensor, containing the ground truth segmentation mask. 
        weights (torch.FloatTensor): 2D tensor of size C, containing the weight of each class, if specified.
        k (int): weight for pGD function. Default is 0 for ordinary dice loss.
    paper:Major vessel segmentation on x-ray coronary angiography using deep networks with a novel penalty loss function
    """  
    n_classes = input.size()[0]

    if weights is not None:
        for c in range(n_classes):
            intersection = (input[c] * target[c] * weights[c]).sum()
            union = (weights[c] * (input[c] + target[c])).sum() + eps
    else:
        intersection = torch.dot(input.view(-1), target.view(-1))
        union = torch.sum(input) + torch.sum(target) + eps    

    gd = (2 * intersection.float() + eps) / union.float()
    return 1 - (gd / (1 + k*(1-gd)))

class pGDice(nn.Module):
    """
    Returns the Generalized Dice Loss Coefficient of a batch associated to the input and target tensors. In case `use_weights` \
        is specified and is `True`, then the computation of the loss takes the class weights into account.
    Args:
        input (torch.FloatTensor): NCHW tensor containing the probabilities predicted for each class.
        target (torch.LongTensor): NCHW one-hot encoded tensor, containing the ground truth segmentation mask. 
        use_weights (bool): specifies whether to use class weights in the computation or not.
        k (int): weight for pGD function. Default is 0 for ordinary dice loss.
    https://github.com/jlcsilva/EfficientUNetPlusPlus/blob/fc123643f0c27c1619728cfeee5e7f377ab43b61/metrics.py#L7
    """
    def __init__(self, use_weights=True, k=0.75, eps=0.0001):
        super().__init__()
        self.use_weights = use_weights
        self.k = k
        self.eps = eps
    def forward(self, pred, target):
        if pred.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()
        # Multiple class case
        n_classes = pred.size()[1]
        if n_classes != 1:
            # Convert target to one hot encoding
            target = F.one_hot(target, n_classes).squeeze()
            if target.ndim == 3:
                target = target.unsqueeze(0)
            target = torch.transpose(torch.transpose(target, 2, 3), 1, 2).type(torch.FloatTensor).cuda().contiguous()
            pred = torch.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)   

        class_weights = None
        for i, c in enumerate(zip(pred, target)):
            if self.use_weights:
                class_weights = torch.pow(torch.sum(c[1], (1,2)) + self.eps, -2)
            s = s + dice_loss(c[0], c[1], class_weights, k=self.k)

        return s / (i + 1)

class OhemCELoss(nn.Module):
    def __init__(self, thresh=0.7, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        # self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        n_min = N * H * W // 16
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:n_min]
        return torch.mean(loss)

torch_losses = {
    'crossentropyloss': nn.CrossEntropyLoss,
    'bceloss': nn.BCEWithLogitsLoss,
    'softcrossentropyloss': smp.losses.SoftCrossEntropyLoss,
    'focalloss': smp.losses.FocalLoss,
    'jaccardloss': smp.losses.JaccardLoss,
    'diceloss': smp.losses.DiceLoss,
    'boundaryloss': BoundaryLoss,
    'lossnet': LossNet,
    'pgdice': pGDice,
    'ohemceloss': OhemCELoss,
    'softceloss':smp.losses.SoftCrossEntropyLoss
}

def get_loss(loss, loss_weights=None, custom_losses=None):
    """Load a loss function based on a config file for the specified framework.

    Arguments
    ---------
    loss : dict
        Dictionary of loss functions to use.  Each key is a loss function name,
        and each entry is a (possibly-empty) dictionary of hyperparameter-value
        pairs.
    loss_weights : dict, optional
        Optional dictionary of weights for loss functions.  Each key is a loss
        function name (same as in the ``loss`` argument), and the corresponding
        entry is its weight.
    custom_losses : dict, optional
        Optional dictionary of Pytorch classes of any
        user-defined loss functions.  Each key is a loss function name, and the
        corresponding entry is the Python object implementing that loss.
    """
    if not isinstance(loss, dict):
        raise TypeError('The loss description is formatted improperly.'
                        ' See the docs for details.')
    if len(loss) > 1:
        # get the weights for each loss within the composite
        if loss_weights is None:
            # weight all losses equally
            weights = {k: 1 for k in loss.keys()}
        else:
            weights = loss_weights

        # check if sublosses dict and weights dict have the same keys
        if list(loss.keys()).sort() != list(weights.keys()).sort():
            raise ValueError(
                'The losses and weights must have the same name keys.')

        return TorchCompositeLoss(loss, weights, custom_losses)
    
    else:  # parse individual loss functions
        loss_name, loss_dict = list(loss.items())[0]
        return get_single_loss(loss_name, loss_dict, custom_losses)


def get_single_loss(loss_name, params_dict, custom_losses=None):

    if params_dict is None:
        if custom_losses is not None and loss_name in custom_losses:
            return custom_losses.get(loss_name)()
        else:
            return torch_losses.get(loss_name.lower())()

    else:
        if custom_losses is not None and loss_name in custom_losses:
            return custom_losses.get(loss_name)(**params_dict)
        else:
            return torch_losses.get(loss_name.lower())(**params_dict)

class TorchCompositeLoss(nn.Module):
    """Composite loss function."""
    def __init__(self, loss_dict, weight_dict=None, custom_losses=None):
        """Create a composite loss function from a set of pytorch losses."""
        super().__init__()
        self.weights = weight_dict
        self.losses = {loss_name: get_single_loss(loss_name,
                                                  loss_params,
                                                  custom_losses)
                       for loss_name, loss_params in loss_dict.items()}
        self.values = {}  # values from the individual loss functions

    def forward(self, outputs, targets):
        loss = 0
        for func_name, weight in self.weights.items():
            self.values[func_name] = self.losses[func_name](outputs, targets)
            loss += weight*self.values[func_name]
        return loss


if __name__ == '__main__':

    # lossnet = LossNet()
    # x = torch.randn((3,2,512,512)).cuda()
    # y = torch.ones((3,512,512)).float().cuda()

    # loss = lossnet(x, y)

    # print(loss)

    # loss = torch_losses['pgdice']()
    # loss = get_single_loss('pgdice', None, torch_losses)
    loss = torch_losses.get('pgdice')()

