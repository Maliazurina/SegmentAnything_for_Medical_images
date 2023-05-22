import cv2 
import pathlib
import matplotlib.pyplot as plt
import os, sys
import nibabel as nib
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
#import matplotlib
#matplotlib.use('TkAgg')
import argparse, shutil
from skimage.measure import label,regionprops
import time

#----functions------------------------
def nifti2RGB(nifti_data, show=False):
    
    HOUNSFIELD_MAX = np.max(nifti_data)
    HOUNSFIELD_MIN = np.min(nifti_data)

    HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

    nifti_data[nifti_data < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    nifti_data[nifti_data > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    normalized_image = (nifti_data - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE
    uint8_image = np.uint8(normalized_image*255)

    opencv_image = cv2.cvtColor(uint8_image, cv2.COLOR_GRAY2BGR)

    if show:
        cv2_imshow(opencv_image) # for Google Colab

    return opencv_image  


def segmentAnything(img,input_box):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    
    input_box = np.array(input_box)
    masks, _, _ = predictor.predict(point_coords=None,point_labels=None,box=input_box[None, :],multimask_output=False)
    mask = masks[0]
    #mask_nifty = nib.Nifti1Image(output, affine=np.eye(4))
    #nib.save(mask_nifty, os.path.join(recon_dir,mask_name))
    return mask
   

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))
#----------------------------------------------------------------

root_dir = pathlib.Path.cwd()
recon_dir = os.path.join(root_dir, "Data", "Segment_Anything")
if os.path.exists(recon_dir):
    shutil.rmtree(recon_dir)
os.makedirs(recon_dir)
    
input_dir = os.path.join(root_dir, "Data", "CT")
mask_dir = os.path.join(root_dir, "Data", "RTS")
image_names = sorted([ele for ele in os.listdir(input_dir) if ele.endswith(".nii.gz")])
image_paths = [os.path.join(input_dir, ele) for ele in image_names]
for ind, cur_img_path in enumerate(image_paths):
    start_time = time.time()

    file_name = os.path.basename(cur_img_path).split('.', 1)[0]
    print('-----------------------------------------')
    print("Segmenting {} {:3d}/{:3d}".format(file_name, ind+1, len(image_paths)))    
    print('-----------------------------------------')
    image = nib.load(cur_img_path).get_fdata().astype(np.float32)
    
    mask_name = file_name.split('_',1)[1]
    mask_name = os.path.join('RTS_' + mask_name + '.nii.gz')
    ref_mask = os.path.join(mask_dir,mask_name)
    ref_mask = nib.load(ref_mask).get_fdata().astype(np.float32)
    points = []
    
    output = np.zeros_like(image)
    
    for i in range(image.shape[2]):
        mask = ref_mask[:,:,i]
        if np.max(mask) != 0:
            print('Segment Slice... :',i)
            mask_labels = label(mask)
            props = regionprops(mask_labels)
            pred = np.zeros_like(mask_labels)
            for regions in props :
                input_box = ([regions.bbox[1],regions.bbox[0],regions.bbox[3],regions.bbox[2]])
                img = nifti2RGB(image[:,:,i])
                out = segmentAnything(img,input_box)
                pred = (out * 1) | (pred) 
                output[:,:,i] = pred
    mask_name = file_name.split('_',1)[1]
    new_name = os.path.join('SAMR_' + mask_name + '.nii.gz')            
    mask_nifty = nib.Nifti1Image(output, affine=np.eye(4))
    nib.save(mask_nifty, os.path.join(recon_dir,new_name))
    end_time = time.time()
    time_lapsed = end_time - start_time
    time_convert(time_lapsed)
                
            
        

                
    
    