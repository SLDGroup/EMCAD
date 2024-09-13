
import torch
import torch.nn as nn
import numpy as np
from medpy import metric
from scipy.ndimage import zoom
import seaborn as sns
from PIL import Image 
import matplotlib.pyplot as plt
from segmentation_mask_overlay import overlay_masks
import matplotlib.colors as mcolors

import SimpleITK as sitk
import pandas as pd

from thop import profile
from thop import clever_format
from ptflops import get_model_complexity_info

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))

def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
    
def one_hot_encoder(input_tensor,dataset,n_classes = None):
    tensor_list = []
    if dataset == 'MMWHS':  
        dict = [0,205,420,500,550,600,820,850]
        for i in dict:
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    else:
        for i in range(n_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()    

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        asd = metric.binary.assd(pred, gt)
        return dice, hd95, jaccard, asd
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0

def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    else:
        return 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, class_names=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if class_names==None:
        mask_labels = np.arange(1,classes)
    else:
        mask_labels = class_names
    cmaps = mcolors.CSS4_COLORS
    my_colors=['red','darkorange','yellow','forestgreen','blue','purple','magenta','cyan','deeppink', 'chocolate', 'olive','deepskyblue','darkviolet']
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes-1]}
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                P = net(input)
                outputs = P[-1]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
                # saving the final output as a PNG file
                #print(test_save_path + '/'+case + '' +str(ind))
                #Image.fromarray((pred/8 * 255).astype(np.uint8)).save(test_save_path + '/'+case + '' +str(ind)+'_pred.png')
                #Image.fromarray((image[ind, :, :] * 255).astype(np.uint8)).save(test_save_path + '/'+case + '' +str(ind)+'_img.png')
                #Image.fromarray((label[ind, :, :]/8 * 255).astype(np.uint8)).save(test_save_path + '/'+case + '' +str(ind)+'_gt.png')
                #cmap = plt.cm.tab20(np.arange(len(mask_labels)))
                
                lbl = label[ind, :, :]
                masks = []
                for i in range(1, classes):
                    masks.append(lbl==i)
                preds_o = []
                for i in range(1, classes):
                    preds_o.append(pred==i)
                
                fig_gt = overlay_masks(image[ind, :, :], masks, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                fig_pred = overlay_masks(image[ind, :, :], preds_o, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                # Do with that image whatever you want to do.
                fig_gt.savefig(test_save_path + '/' + case + '_' +str(ind) + '_gt.png', bbox_inches="tight", dpi=300)
                fig_pred.savefig(test_save_path + '/' + case + '_' +str(ind) + '_pred.png', bbox_inches="tight", dpi=300)

    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            outputs = P[-1]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []    
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def val_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                P = net(input)
                outputs = 0.0
                outputs = P[-1]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            outputs = P[-1]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    return metric_list

def horizontal_flip(image):
    image = image[:, ::-1, :]
    return image

def vertical_flip(image):
    image = image[::-1, :, :]
    return image

def tta_model(model, image):
    n_image = image
    h_image = horizontal_flip(image)
    v_image = vertical_flip(image)

    n_mask = model.predict(np.expand_dims(n_image, axis=0))[0]
    h_mask = model.predict(np.expand_dims(h_image, axis=0))[0]
    v_mask = model.predict(np.expand_dims(v_image, axis=0))[0]

    n_mask = n_mask
    h_mask = horizontal_flip(h_mask)
    v_mask = vertical_flip(v_mask)

    mean_mask = (n_mask + h_mask + v_mask) / 3.0
    return mean_mask

def cal_params_flops(model, size, logger):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')

# Example function to calculate and print GMACs and parameter count for a given model
def print_model_stats(model, input_size=(3, 224, 224)):
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model created, param count: {total_params}')
    
    # Calculate GMACs using ptflops
    macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)
    
    # Display GMACs and params
    print(f'Model: {macs} GMACs, {params} parameters')