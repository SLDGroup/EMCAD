import os
import shutil
from time import time

import numpy as np
import SimpleITK as sitk
import nibabel as nib
import scipy.ndimage as ndimage
import h5py

splits = ['train', 'test']
#train = True # Set True to process training set and set False for testset 

for split in splits:
    if(split == 'train'):
        ct_path = './data/synapse/Abdomen/RawData/TrainSet/img' # set your path to your trainset directory
        seg_path = './data/synapse/Abdomen/RawData/TrainSet/label' 
        save_path = './data/synapse/train_npz_new/'
    else:
        ct_path = './data/synapse/Abdomen/RawData/TestSet/img' # set your path to your testset directory
        seg_path = './data/synapse/Abdomen/RawData/TestSet/label'
        save_path = './data/synapse/test_vol_h5_new/'
    
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    upper = 275 
    lower = -125

    start_time = time()

    for ct_file in os.listdir(ct_path):

        ct = nib.load(os.path.join(ct_path, ct_file))
        seg = nib.load(os.path.join(seg_path, ct_file.replace('img', 'label')))

        #Convert them to numpy format, 
        ct_array = ct.get_fdata()
        seg_array = seg.get_fdata()

        ct_array = np.clip(ct_array, lower, upper)
    
        #print([np.min(ct_array), np.max(ct_array)])
    
        #normalize each 3D image to [0, 1] 
        ct_array = (ct_array - lower) / (upper - lower)
    
        #print([np.min(ct_array), np.max(ct_array)])
    
        ct_array = np.transpose(ct_array, (2, 0, 1))
        seg_array = np.transpose(seg_array, (2, 0, 1))
    
        print('file name:', ct_file)
        print('shape:', ct_array.shape)

        ct_number = ct_file.split('.')[0]
        if(split == 'test'):
    	    new_ct_name = ct_number.replace('img', 'case')+'.npy.h5'
    	    hf = h5py.File(os.path.join(save_path, new_ct_name), 'w')
    	    hf.create_dataset('image', data=ct_array)
    	    hf.create_dataset('label', data=seg_array)
    	    hf.close()
    	    continue
    	
        for s_idx in range(ct_array.shape[0]):
    	    ct_array_s = ct_array[s_idx, :, :]
    	    seg_array_s = seg_array[s_idx, :, :]
    	    slice_no = "{:03d}".format(s_idx)
    	    new_ct_name = ct_number.replace('img', 'case') + '_slice' + slice_no
    	    np.savez(os.path.join(save_path, new_ct_name), image=ct_array_s, label=seg_array_s)

        print('already use {:.3f} min'.format((time() - start_time) / 60))
        print('-----------')

