import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import argparse
from tqdm import tqdm

# Project-specific imports
from lib.networks import EMCADNet
from utils.dataloader_polyp import get_loader
from medpy.metric.binary import hd95

def dice_coefficient(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2. * intersection + smooth) / (total + smooth)

def iou(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def get_binary_metrics(pred, gt):
    tp = (pred * gt).sum().item()
    tn = ((1 - pred) * (1 - gt)).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    
    try:
        if pred.sum() > 0 and gt.sum() > 0:
            hd_val = hd95(pred.cpu().numpy(), gt.cpu().numpy())
        else:
            hd_val = 100.0
    except:
        hd_val = 100.0
        
    return sensitivity, specificity, precision, hd_val

def test(model, path, dataset, opt, save_base=None):
    data_path = os.path.join(path, dataset)
    image_root = f'{data_path}/images/'
    gt_root = f'{data_path}/masks/'
    model.eval()
    
    test_loader = get_loader(
        image_root=image_root, gt_root=gt_root, 
        batchsize=opt.test_batchsize, trainsize=opt.img_size,
        shuffle=False, split='test', color_image=opt.color_image
    )
    
    DSC, IOU, total_images = 0.0, 0.0, 0
    detailed_results = []

    with torch.no_grad():
        for pack in tqdm(test_loader, desc=f"Inference on {dataset}"):
            images, gts, original_shapes, names = pack       
            images, gts = images.cuda(), gts.cuda().float()

            ress = model(images)
            if not isinstance(ress, list):
                ress = [ress]
            # Take the primary output (EMCADNet usually uses the last item for final prediction)
            predictions = ress[-1]
            
            for i in range(len(images)):
                h_orig, w_orig = int(original_shapes[0][i]), int(original_shapes[1][i])
                
                p = predictions[i].unsqueeze(0)
                pred_resized = F.interpolate(p, size=(h_orig, w_orig), mode='bilinear', align_corners=False).sigmoid().squeeze()
                pred_resized = (pred_resized - pred_resized.min()) / (pred_resized.max() - pred_resized.min() + 1e-8)
                
                g = gts[i].unsqueeze(0)
                gt_resized = F.interpolate(g, size=(h_orig, w_orig), mode='nearest').squeeze()

                input_binary = (pred_resized >= 0.5).float()
                target_binary = (gt_resized >= 0.2).float()

                d = dice_coefficient(input_binary, target_binary).item()
                io = iou(input_binary, target_binary).item()
                sens, spec, prec, hd = get_binary_metrics(input_binary, target_binary)

                DSC += d
                IOU += io
                total_images += 1

                detailed_results.append({
                    'Name': names[i], 'Dice': d, 'IoU': io,
                    'Sensitivity': float('{:.4f}'.format(sens)),
                    'Specificity': float('{:.4f}'.format(spec)),
                    'Precision': float('{:.4f}'.format(prec)),
                    'HD95': float('{:.4f}'.format(hd))
                })

                if save_base:
                    pred_img = (input_binary.cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_base, names[i]), pred_img)

    return DSC / total_images, IOU / total_images, detailed_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--encoder', type=str, default='pvt_v2_b2')
    parser.add_argument('--expansion_factor', type=int, default=2)
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--lgag_ks', type=int, default=3)
    parser.add_argument('--activation_mscb', type=str, default='relu6')
    parser.add_argument('--no_dw_parallel', action='store_true', default=False)
    parser.add_argument('--concatenation', action='store_true', default=False)
    parser.add_argument('--dataset_name', type=str, default='ClinicDB')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--test_batchsize', type=int, default=1)
    parser.add_argument('--color_image', default=True)
    parser.add_argument('--test_path', type=str, default='../data/polyp/target/')
    opt = parser.parse_args()

    # --- Paths ---
    save_base = f'./predictions_polyp/{opt.run_id}/{opt.dataset_name}/{opt.split}'
    os.makedirs(save_base, exist_ok=True)
    os.makedirs('results_polyp', exist_ok=True)
    model_path = os.path.join(f'./model_pth/{opt.run_id}/', f'{opt.run_id}-best.pth')
    opt.test_path = f'{opt.test_path}/{opt.dataset_name}/'

    # --- Model Loading ---
    model = EMCADNet(
        num_classes=1, 
        kernel_sizes=opt.kernel_sizes, 
        expansion_factor=opt.expansion_factor, 
        dw_parallel=not opt.no_dw_parallel, 
        add=not opt.concatenation, 
        lgag_ks=opt.lgag_ks, 
        activation=opt.activation_mscb, 
        encoder=opt.encoder, 
        pretrain=False # Always False for inference
    ).cuda()
    
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    # --- Run Inference ---
    # Adjust test path to match main script behavior
    mean_dice, mean_iou, results = test(model, opt.test_path, opt.split, opt, save_base=save_base)

    # --- Individual Excel ---
    df = pd.DataFrame(results)
    mean_row = df.mean(numeric_only=True).to_dict()
    mean_row['Name'] = 'AVERAGE'
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df.to_excel(f'results_polyp/Results_{opt.run_id}_{opt.dataset_name}_{opt.split}.xlsx', index=False)

    # --- Persistent Summary ---
    summary_file = 'All_Runs_Summary_Polyp.xlsx'
    avg_data = {
        'run_id': opt.run_id, 'network': 'EMCADNet', 'dataset': opt.dataset_name,
        'split': opt.split, 'dice': mean_dice, 'iou': mean_iou,
        'sensitivity': mean_row['Sensitivity'], 'specificity': mean_row['Specificity'],
        'precision': mean_row['Precision'], 'HD95': mean_row['HD95']
    }
    df_new = pd.DataFrame([avg_data])

    if os.path.exists(summary_file):
        df_existing = pd.read_excel(summary_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(summary_file, index=False)
    else:
        df_new.to_excel(summary_file, index=False)

    print(f"Evaluation complete. Summary appended to {summary_file}")