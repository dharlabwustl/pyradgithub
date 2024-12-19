#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Timporthu Mar 12 15:40:53 2020

@author: atul
"""
import subprocess,os,sys,glob,datetime,time
import csv
import pandas as pd
import numpy as np
import cv2
import nibabel as nib
from skimage import exposure
import smtplib,math,argparse,inspect
from github import Github
#############################################################
from dateutil.parser import parse
import pandas as pd
import h5py

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
# g = Github()
# repo = g.get_repo("dharlabwustl/EDEMA_MARKERS")
# contents = repo.get_contents("module_NWU_CSFCompartment_Calculations.py")
# dt = parse(contents.last_modified)

Version_Date="VersionDate-" + "08102023" #dt.strftime("%m%d%Y")
# import matplotlib.pyplot as plt
def demo():
    print(" i m in demo")
# def histogram_sidebyside(infarct_data,noninfarct_data,image_filename):

#     dflux = pd.DataFrame(dict(INFARCT=infarct_data))
#     dflux2 = pd.DataFrame(dict(NONINFARCT=noninfarct_data))

#     fig, axes = plt.subplots(1, 2)

#     dflux.hist('INFARCT', bins=255, ax=axes[0])
#     dflux2.hist('NONINFARCT', bins=255, ax=axes[1])
#     fig.savefig(image_filename)

################## REGISTRATION STEPS #####################################

# Function for Z-score normalization
def call_separate_masks_from_multivalue_mask(args):
    multivaluemaskfile=args.stuff[1]
    masks_output_directory=args.stuff[2]
    separate_masks_from_multivalue_mask(multivaluemaskfile,masks_output_directory)

def separate_masks_from_multivalue_mask(multivaluemaskfile,masks_output_directory):
    multivaluemaskfile_nib=nib.load(multivaluemaskfile)
    multivaluemaskfile_nib_data=multivaluemaskfile_nib.get_fdata()
    multivaluemaskfile_nib_data=np.rint(multivaluemaskfile_nib_data).astype(int)
    unique_values=np.unique(multivaluemaskfile_nib_data)
    for x in unique_values:
        if x !=0:
            this_mask_filename=multivaluemaskfile.split('.nii')[0]+'_'+str(x)+'.nii.gz'
            this_mask_data=np.copy(multivaluemaskfile_nib_data)
            this_mask_data[this_mask_data!=x]=0
            this_mask_data[this_mask_data==x]=1
            arraynib=nib.Nifti1Image(this_mask_data,affine=multivaluemaskfile_nib.affine,header=multivaluemaskfile_nib.header)
            nib.save(arraynib,os.path.join(masks_output_directory,os.path.basename(this_mask_filename)))
def z_score_normalization(image_data):
    mean = np.mean(image_data)
    std = np.std(image_data)
    normalized_data = (image_data - mean) / std
    return normalized_data

# Function for Min-Max normalization
def min_max_normalization(image_data, new_min=0, new_max=1):
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    normalized_data = (image_data - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    return normalized_data

# Function for resampling to a specified voxel size
def resample_image_to_voxel_size(image_data, current_voxel_size, target_voxel_size):
    # Calculate the zoom factors for each dimension
    zoom_factors = [current / target for current, target in zip(current_voxel_size, target_voxel_size)]
    # Resample the image using scipy.ndimage.zoom
    resampled_data = zoom(image_data, zoom_factors, order=1)  # Linear interpolation
    return resampled_data
def normalization_N_resample_to_fixed(moving_image_file,fixed_image_file):
# Load the NIfTI file and extract the image data
    moving_image_nii = nib.load(moving_image_file) #'moving_image.nii.gz')
    fixed_image_nii = nib.load(fixed_image_file) ##'fixed_image.nii.gz')

    # Extract image data as NumPy arrays
    moving_image_data = moving_image_nii.get_fdata()
    fixed_image_data = fixed_image_nii.get_fdata()
    fixed_image_nii_max=np.max(fixed_image_nii.get_fdata())

    # Extract voxel sizes from the NIfTI headers
    moving_voxel_size = moving_image_nii.header.get_zooms()[:3]
    fixed_voxel_size = fixed_image_nii.header.get_zooms()[:3]

    # Step 1: Normalize intensities
    moving_image_normalized = min_max_normalization(moving_image_data)*fixed_image_nii_max
    fixed_image_normalized = min_max_normalization(fixed_image_data)*fixed_image_nii_max

    # Step 2: Resample the moving image to match the fixed image voxel size
    resampled_moving_image_data = resample_image_to_voxel_size(moving_image_normalized, moving_voxel_size, fixed_voxel_size)

    # Convert resampled data back to a NIfTI image using the fixed image's affine matrix and header
    resampled_moving_image_nii = nib.Nifti1Image(resampled_moving_image_data, affine=fixed_image_nii.affine, header=fixed_image_nii.header)
    fixed_image_normalized_nii = nib.Nifti1Image(fixed_image_normalized, affine=fixed_image_nii.affine, header=fixed_image_nii.header)

# Save the normalized and resampled image
    nib.save(resampled_moving_image_nii, moving_image_file.split('.nii')[0]+'resampled_normalized_mov.nii.gz')
    nib.save(fixed_image_normalized_nii, fixed_image_file.split('.nii')[0]+'_normalized_fix.nii.gz')
def only_resample_to_fixed(moving_image_file,fixed_image_file):
    # Load the NIfTI file and extract the image data
    moving_image_nii = nib.load(moving_image_file) #'moving_image.nii.gz')
    fixed_image_nii = nib.load(fixed_image_file) ##'fixed_image.nii.gz')
    command="echo I AM  at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
    subprocess.call(command,shell=True)
    # Extract image data as NumPy arrays

    moving_image_data = moving_image_nii.get_fdata()
    print(moving_image_data.shape)
    fixed_image_data = fixed_image_nii.get_fdata()
    print(fixed_image_data.shape)

    if len(moving_image_data.shape) >3:
        moving_image_data=moving_image_data[:,:,:,0]
    print(moving_image_data.shape)


    # Extract voxel sizes from the NIfTI headers
    moving_voxel_size = moving_image_nii.header.get_zooms()[:3] #.astype('float')
    fixed_voxel_size = fixed_image_nii.header.get_zooms()[:3] #.astype('float')

    # # Step 1: Normalize intensities
    # moving_image_normalized = z_score_normalization(moving_image_data)
    # fixed_image_normalized = z_score_normalization(fixed_image_data)

    # Step 2: Resample the moving image to match the fixed image voxel size
    resampled_moving_image_data = resample_image_to_voxel_size(moving_image_data, moving_voxel_size, fixed_voxel_size)

    # Convert resampled data back to a NIfTI image using the fixed image's affine matrix and header
    resampled_moving_image_nii = nib.Nifti1Image(resampled_moving_image_data, affine=fixed_image_nii.affine, header=fixed_image_nii.header)
    # fixed_image_normalized_nii = nib.Nifti1Image(fixed_image_normalized, affine=fixed_image_nii.affine, header=fixed_image_nii.header)

    # Save the normalized and resampled image
    nib.save(resampled_moving_image_nii, moving_image_file.split('.nii')[0]+'resampled_mov.nii.gz')
    # nib.save(fixed_image_normalized_nii, fixed_image_file.split('.nii')[0]+'_normalized_fix.nii.gz')
def call_only_resample_to_fixed(args):
    success=0
    try:
        moving_image=args.stuff[1]
        fixed_image=args.stuff[2]
        only_resample_to_fixed(moving_image,fixed_image)
        command="echo passed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        success=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        pass
    return success


def call_normalization_N_resample_to_fixed(args):
    success=0
    try:
        moving_image=args.stuff[1]
        fixed_image=args.stuff[2]
        normalization_N_resample_to_fixed(moving_image,fixed_image)
        command="echo passed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        success=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        pass
    return success
# # Step 3: Perform initial affine registration for a rough alignment
# affine_transform = sitk.CenteredTransformInitializer(
#     fixed_image_normalized,
#     moving_image_resampled,
#     sitk.Euler3DTransform(),
#     sitk.CenteredTransformInitializerFilter.GEOMETRY
# )
#
# # Use Mutual Information as the similarity metric for the affine registration
# affine_registration = sitk.ImageRegistrationMethod()
# affine_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
# affine_registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=300, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
# affine_registration.SetInitialTransform(affine_transform, inPlace=False)
# affine_registration.SetInterpolator(sitk.sitkLinear)
#
# # Execute the affine registration
# affine_transform = affine_registration.Execute(fixed_image_normalized, moving_image_resampled)
#
# # Step 4: Perform non-linear registration using the affine result as initialization
# nonlinear_registration = sitk.ImageRegistrationMethod()
# nonlinear_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
# nonlinear_registration.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=200)
# nonlinear_registration.SetInitialTransformAsBSpline(affine_transform,
#                                                     numberOfControlPoints=[8, 8, 8], order=3)
# nonlinear_registration.SetInterpolator(sitk.sitkLinear)
#
# # Execute the non-linear registration
# nonlinear_transform = nonlinear_registration.Execute(fixed_image_normalized, moving_image_resampled)
#
# # Step 5: Apply the transformation to warp the moving image to the fixed image space
# warped_image = sitk.Resample(moving_image, fixed_image, nonlinear_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
#
# # Step 6: Save the resulting registered image
# sitk.WriteImage(warped_image, 'registered_moving_image_to_fixed.nii.gz')
#
# print("Non-linear registration completed. Registered image saved as 'registered_moving_image_to_fixed.nii.gz'.")


def call_separate_mask_regions_into_individual_image(args):
    success=0
    try:
        nifti_file_path=args.stuff[1]
        output_dir=args.stuff[2]
        separate_mask_regions_into_individual_image(nifti_file_path,output_dir)
        command="echo passed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        success=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        pass
    return success

def separate_mask_regions_into_individual_image(nifti_file_path,output_dir):
    success=0
    try:
        ##############################
        # Load the NIfTI file
        # nifti_file_path = args.stuff[1] #'path_to_your_mask_file.nii.gz'  # Replace with your actual file path
        # output_dir=args.stuff[2]
        nifti_image = nib.load(nifti_file_path)
        nifti_data = nifti_image.get_fdata()

        # Get unique region values (excluding zero, if it represents the background)
        unique_values = np.unique(nifti_data)
        unique_values = unique_values[unique_values != 0]  # Exclude background value if needed

        # # Directory to save individual region files
        # output_dir = 'output_regions'
        # os.makedirs(output_dir, exist_ok=True)

        # Loop through each unique region value and create individual masks
        for region_value in unique_values:
            region_mask = (nifti_data == region_value).astype(np.float32)
            region_mask[region_mask>0]=1
            # Create a new NIfTI image for the region
            region_img = nib.Nifti1Image(region_mask, affine=nifti_image.affine, header=nifti_image.header)

            # Save the region image
            region_output_path = os.path.join(output_dir, f"mri_region_{int(region_value)}.nii.gz")
            nib.save(region_img, region_output_path)
            print(f'Saved region {int(region_value)} as {region_output_path}')
    ########################
        command="echo passed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        success=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        pass
    return success
def coninuous2binary0_255(coninuous_image_file):
    coninuous_image_file_nib=nib.load(coninuous_image_file)
    coninuous_image_file_nib_data=coninuous_image_file_nib.dataobj.get_unscaled() #""
    coninuous_image_file_nib_data[coninuous_image_file_nib_data>0]=255
    array_mask = nib.Nifti1Image(coninuous_image_file_nib_data, affine=coninuous_image_file_nib.affine, header=coninuous_image_file_nib.header)
    # niigzfilenametosave2=os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
    nib.save(array_mask, coninuous_image_file)





def whenOFsize512x512(levelset_file,OUTPUT_DIRECTORY):
    if "WUSTL" in levelset_file:
        image_levelset_nib=nib.load(levelset_file)
        image_levelset_data=image_levelset_nib.dataobj.get_unscaled()
        flipped_mask=np.copy(image_levelset_data)
        for idx in range(image_levelset_data.shape[2]):
            flipped_mask[:,:,idx]=cv2.flip(image_levelset_data[:,:,idx],0)
        array_mask = nib.Nifti1Image(flipped_mask, affine=image_levelset_nib.affine, header=image_levelset_nib.header)
        niigzfilenametosave2=os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
        nib.save(array_mask, niigzfilenametosave2)
    else:
        command= "cp  " + levelset_file + "  " + os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file))
        subprocess.call(command, shell=True)
    return "X"

def whenOFsize512x512_new(levelset_file,original_file,OUTPUT_DIRECTORY):
#     if "WUSTL" in levelset_file:
    original_file_nib=nib.load(original_file)
    image_levelset_nib=nib.load(levelset_file)
    image_levelset_data1=image_levelset_nib.dataobj.get_unscaled()
#         flipped_mask=np.copy(image_levelset_data)
#         for idx in range(image_levelset_data.shape[2]):
#             flipped_mask[:,:,idx]=cv2.flip(image_levelset_data[:,:,idx],0)
#         array_mask = nib.Nifti1Image(flipped_mask, affine=image_levelset_nib.affine, header=image_levelset_nib.header)
#         niigzfilenametosave2=os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
#         nib.save(array_mask, niigzfilenametosave2)
#     else:
    print("I am in whenOFsize512x512_new()")
    array_mask = nib.Nifti1Image(image_levelset_data1, affine=original_file_nib.affine, header=original_file_nib.header)
    niigzfilenametosave2=os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
    nib.save(array_mask, niigzfilenametosave2)


#     command= "cp  " + levelset_file + "  " + os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file))
#     subprocess.call(command, shell=True)
    return "X"

def whenOFsize512x512_new_flip_np(image_levelset_data1,original_file,levelset_file,OUTPUT_DIRECTORY):
    #     if "WUSTL" in levelset_file:
    original_file_nib=nib.load(original_file)
    # image_levelset_nib=nib.load(levelset_file)
    # image_levelset_data1=image_levelset_nib.dataobj.get_unscaled()
    # for x in range(image_levelset_data1.shape[2]):
    #     image_levelset_data1[:,:,x]=cv2.flip(image_levelset_data1[:,:,x],0)
    #         flipped_mask=np.copy(image_levelset_data)
    #         for idx in range(image_levelset_data.shape[2]):
    #             flipped_mask[:,:,idx]=cv2.flip(image_levelset_data[:,:,idx],0)
    #         array_mask = nib.Nifti1Image(flipped_mask, affine=image_levelset_nib.affine, header=image_levelset_nib.header)
    #         niigzfilenametosave2=os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
    #         nib.save(array_mask, niigzfilenametosave2)
    #     else:
    print("I am in whenOFsize512x512_new()")
    array_mask = nib.Nifti1Image(image_levelset_data1, affine=original_file_nib.affine, header=original_file_nib.header)
    niigzfilenametosave2=os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
    nib.save(array_mask, niigzfilenametosave2)


    #     command= "cp  " + levelset_file + "  " + os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file))
    #     subprocess.call(command, shell=True)
    return "X"
def whenOFsize512x512_new_flip(levelset_file,original_file,OUTPUT_DIRECTORY):
#     if "WUSTL" in levelset_file:
    print('original_file.shape::{}'.format(original_file))
    print('levelset_file.shape::{}'.format(levelset_file))
    original_file_nib=nib.load(original_file)
    image_levelset_nib=nib.load(levelset_file)
    print('original_file_nib.shape::{}'.format(original_file_nib.shape))
    print('image_levelset_nib.shape::{}'.format(image_levelset_nib.shape))
    image_levelset_data1=image_levelset_nib.dataobj.get_unscaled()
    print('image_levelset_data1.shape::{}'.format(image_levelset_data1.shape))
    for x in range(image_levelset_data1.shape[2]):
        image_levelset_data1[:,:,x]=cv2.flip(image_levelset_data1[:,:,x],0)
#         flipped_mask=np.copy(image_levelset_data)
#         for idx in range(image_levelset_data.shape[2]):
#             flipped_mask[:,:,idx]=cv2.flip(image_levelset_data[:,:,idx],0)
#         array_mask = nib.Nifti1Image(flipped_mask, affine=image_levelset_nib.affine, header=image_levelset_nib.header)
#         niigzfilenametosave2=os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
#         nib.save(array_mask, niigzfilenametosave2)
#     else:
    print("I am in whenOFsize512x512_new()")

    print('image_levelset_data1_modified.shape::{}'.format(image_levelset_data1.shape))
    array_mask = nib.Nifti1Image(image_levelset_data1, affine=original_file_nib.affine, header=original_file_nib.header)
    niigzfilenametosave2=os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
    nib.save(array_mask, niigzfilenametosave2)


#     command= "cp  " + levelset_file + "  " + os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file))
#     subprocess.call(command, shell=True)
    return "X"

def whenOFsize512x5xx(original_file,levelset_file,OUTPUT_DIRECTORY="./"):
    image_nib_nii_file=nib.load(original_file)  #,header=False)
    image_nib_nii_file_data=image_nib_nii_file.get_fdata() #
    image_levelset_data=nib.load(levelset_file).dataobj.get_unscaled()
    array_mask1=nib.load(levelset_file)
    if image_nib_nii_file.get_fdata().shape[1]>512 : #image_levelset_data.shape[1]:
#    print("a")
        print("ORIGINAL IMAGE SIZE")
        print(image_nib_nii_file.get_fdata().shape)
        print("YES DIFFER")
        # print(imagefile_levelset)
#    npad = ((0, 0), (difference_size//2, difference_size//2), (0, 0))
        size_diff=(image_nib_nii_file.get_fdata().shape[1]-512)
        if (size_diff % 2 )== 0 :
            size_diff=int(size_diff/2)
            npad = ((0, 0), (size_diff-1, size_diff+1), (0, 0))
        else :
            size_diff=int(size_diff/2)
            npad = ((0, 0), (size_diff, size_diff+1), (0, 0))  #abs(np.min(image_levelset_data)
        image_levelset_data1=np.pad(image_levelset_data, pad_width=npad, mode='constant', constant_values=np.min(image_levelset_data)) #image_nib_nii_file_data[0:image_nib_nii_file_data.shape[0],0+(difference_size//2):image_nib_nii_file_data.shape[1]-(difference_size//2),0:image_nib_nii_file_data.shape[2]]
        if "WUSTL" in levelset_file:
            flipped_mask=np.copy(image_levelset_data1)
            for idx in range(image_levelset_data1.shape[2]):
                flipped_mask[:,:,idx]=cv2.flip(image_levelset_data1[:,:,idx],0)

                image_levelset_data1=np.copy(flipped_mask)
        array_mask = nib.Nifti1Image(image_levelset_data1, affine=image_nib_nii_file.affine, header=image_nib_nii_file.header)
        niigzfilenametosave2=os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
        nib.save(array_mask, niigzfilenametosave2)
        # array_mask1=nib.load(niigzfilenametosave2)

    return "X"
def whenOFsize512x5xx_new(original_file,levelset_file,OUTPUT_DIRECTORY="./"):
    image_nib_nii_file=nib.load(original_file)  #,header=False)
    image_nib_nii_file_data=image_nib_nii_file.get_fdata() #
    image_levelset_data=nib.load(levelset_file).dataobj.get_unscaled()
    array_mask1=nib.load(levelset_file)
    size_diff_x=np.abs(image_nib_nii_file.get_fdata().shape[0]-512)
    size_diff_y=np.abs(image_nib_nii_file.get_fdata().shape[1]-512)
    temp_array=np.copy(image_levelset_data)

    if image_nib_nii_file.get_fdata().shape[0] >512:
        if (size_diff_x % 2 )== 0 :
            size_diff_x=int(size_diff_x/2)
            npad = ((size_diff_x-1, size_diff_x+1), (0, 0), (0, 0))
        else :
            size_diff_x=int(size_diff_x/2)
            npad = ((size_diff_x, size_diff_x+1), (0, 0), (0, 0))  #abs(np.min(image_levelset_data)
        temp_array=np.pad(temp_array, pad_width=npad, mode='constant', constant_values=np.min(temp_array))
    if image_nib_nii_file.get_fdata().shape[1] >512:
        if (size_diff_y % 2 )== 0 :
            size_diff_y=int(size_diff_y/2)
            npad = ((0, 0),(size_diff_y-1, size_diff_y+1),  (0, 0))
        else :
            size_diff_y=int(size_diff_y/2)
            npad = ( (0, 0),(size_diff_y, size_diff_y+1), (0, 0))  #abs(np.min(image_levelset_data)
        temp_array=np.pad(temp_array, pad_width=npad, mode='constant', constant_values=np.min(temp_array))

    if image_nib_nii_file.get_fdata().shape[0] < 512:
        if (size_diff_x % 2 )== 0 :
            size_diff_x=int(size_diff_x/2)
            temp_array=temp_array[size_diff_x:temp_array.shape[0]-size_diff_x,0:temp_array.shape[1],0:temp_array.shape[2]]
        else :
            size_diff_x=int(size_diff_x/2)
            temp_array=temp_array[size_diff_x:temp_array.shape[0]-size_diff_x-1,0:temp_array.shape[1],0:temp_array.shape[2]]

    if image_nib_nii_file.get_fdata().shape[1] < 512:
        if (size_diff_y % 2 )== 0 :
            size_diff_y=int(size_diff_y/2)
            temp_array=temp_array[0:temp_array.shape[0],size_diff_y:temp_array.shape[1]-size_diff_y,0:temp_array.shape[2]]
        else :
            size_diff_y=int(size_diff_y/2)
            temp_array=temp_array[0:temp_array.shape[0],size_diff_y:temp_array.shape[1]-size_diff_y-1,0:temp_array.shape[2]]

    image_levelset_data1=temp_array
    array_mask = nib.Nifti1Image(image_levelset_data1, affine=image_nib_nii_file.affine, header=image_nib_nii_file.header)
    niigzfilenametosave2=os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
    nib.save(array_mask, niigzfilenametosave2)
    print("image_nib_nii_file_data.shape: {}::image_levelset_data1.shape{} ".format(image_nib_nii_file_data.shape,image_levelset_data1.shape))

    return "X"

def whenOFsize512x5xx_new_flip_np(original_file,image_levelset_data,levelset_file,OUTPUT_DIRECTORY):
    image_nib_nii_file=nib.load(original_file)  #,header=False)
    # image_nib_nii_file_data=image_nib_nii_file.get_fdata() #
    # image_levelset_data=nib.load(levelset_file).dataobj.get_unscaled()
    # array_mask1=nib.load(levelset_file)
    size_diff_x=np.abs(image_nib_nii_file.get_fdata().shape[0]-512)
    size_diff_y=np.abs(image_nib_nii_file.get_fdata().shape[1]-512)
    temp_array=np.copy(image_levelset_data)

    if image_nib_nii_file.get_fdata().shape[0] >512:
        if (size_diff_x % 2 )== 0 :
            size_diff_x=int(size_diff_x/2)
            npad = ((size_diff_x-1, size_diff_x+1), (0, 0), (0, 0))
        else :
            size_diff_x=int(size_diff_x/2)
            npad = ((size_diff_x, size_diff_x+1), (0, 0), (0, 0))  #abs(np.min(image_levelset_data)
        temp_array=np.pad(temp_array, pad_width=npad, mode='constant', constant_values=np.min(temp_array))
    if image_nib_nii_file.get_fdata().shape[1] >512:
        if (size_diff_y % 2 )== 0 :
            size_diff_y=int(size_diff_y/2)
            npad = ((0, 0),(size_diff_y-1, size_diff_y+1),  (0, 0))
        else :
            size_diff_y=int(size_diff_y/2)
            npad = ( (0, 0),(size_diff_y, size_diff_y+1), (0, 0))  #abs(np.min(image_levelset_data)
        temp_array=np.pad(temp_array, pad_width=npad, mode='constant', constant_values=np.min(temp_array))

    if image_nib_nii_file.get_fdata().shape[0] < 512:
        if (size_diff_x % 2 )== 0 :
            size_diff_x=int(size_diff_x/2)
            temp_array=temp_array[size_diff_x:temp_array.shape[0]-size_diff_x,0:temp_array.shape[1],0:temp_array.shape[2]]
        else :
            size_diff_x=int(size_diff_x/2)
            temp_array=temp_array[size_diff_x:temp_array.shape[0]-size_diff_x-1,0:temp_array.shape[1],0:temp_array.shape[2]]

    if image_nib_nii_file.get_fdata().shape[1] < 512:
        if (size_diff_y % 2 )== 0 :
            size_diff_y=int(size_diff_y/2)
            temp_array=temp_array[0:temp_array.shape[0],size_diff_y:temp_array.shape[1]-size_diff_y,0:temp_array.shape[2]]
        else :
            size_diff_y=int(size_diff_y/2)
            temp_array=temp_array[0:temp_array.shape[0],size_diff_y:temp_array.shape[1]-size_diff_y-1,0:temp_array.shape[2]]


    # for x in range(temp_array.shape[2]):
    #     temp_array[:,:,x]=cv2.flip(temp_array[:,:,x],0)
    image_levelset_data1=temp_array
    array_mask = nib.Nifti1Image(image_levelset_data1, affine=image_nib_nii_file.affine, header=image_nib_nii_file.header)
    niigzfilenametosave2=os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
    nib.save(array_mask, niigzfilenametosave2)
    # print("image_nib_nii_file_data.shape: {}::image_levelset_data1.shape{} ".format(image_nib_nii_file_data.shape,image_levelset_data1.shape))

    return "X"
def whenOFsize512x5xx_new_flip(original_file,levelset_file,OUTPUT_DIRECTORY="./"):
    image_nib_nii_file=nib.load(original_file)  #,header=False)
    image_nib_nii_file_data=image_nib_nii_file.get_fdata() #
    image_levelset_data=nib.load(levelset_file).dataobj.get_unscaled()
    array_mask1=nib.load(levelset_file)
    size_diff_x=np.abs(image_nib_nii_file.get_fdata().shape[0]-512)
    size_diff_y=np.abs(image_nib_nii_file.get_fdata().shape[1]-512)
    temp_array=np.copy(image_levelset_data)

    if image_nib_nii_file.get_fdata().shape[0] >512:
        if (size_diff_x % 2 )== 0 :
            size_diff_x=int(size_diff_x/2)
            npad = ((size_diff_x-1, size_diff_x+1), (0, 0), (0, 0))
        else :
            size_diff_x=int(size_diff_x/2)
            npad = ((size_diff_x, size_diff_x+1), (0, 0), (0, 0))  #abs(np.min(image_levelset_data)
        temp_array=np.pad(temp_array, pad_width=npad, mode='constant', constant_values=np.min(temp_array))
    if image_nib_nii_file.get_fdata().shape[1] >512:
        if (size_diff_y % 2 )== 0 :
            size_diff_y=int(size_diff_y/2)
            npad = ((0, 0),(size_diff_y-1, size_diff_y+1),  (0, 0))
        else :
            size_diff_y=int(size_diff_y/2)
            npad = ( (0, 0),(size_diff_y, size_diff_y+1), (0, 0))  #abs(np.min(image_levelset_data)
        temp_array=np.pad(temp_array, pad_width=npad, mode='constant', constant_values=np.min(temp_array))

    if image_nib_nii_file.get_fdata().shape[0] < 512:
        if (size_diff_x % 2 )== 0 :
            size_diff_x=int(size_diff_x/2)
            temp_array=temp_array[size_diff_x:temp_array.shape[0]-size_diff_x,0:temp_array.shape[1],0:temp_array.shape[2]]
        else :
            size_diff_x=int(size_diff_x/2)
            temp_array=temp_array[size_diff_x:temp_array.shape[0]-size_diff_x-1,0:temp_array.shape[1],0:temp_array.shape[2]]

    if image_nib_nii_file.get_fdata().shape[1] < 512:
        if (size_diff_y % 2 )== 0 :
            size_diff_y=int(size_diff_y/2)
            temp_array=temp_array[0:temp_array.shape[0],size_diff_y:temp_array.shape[1]-size_diff_y,0:temp_array.shape[2]]
        else :
            size_diff_y=int(size_diff_y/2)
            temp_array=temp_array[0:temp_array.shape[0],size_diff_y:temp_array.shape[1]-size_diff_y-1,0:temp_array.shape[2]]


    for x in range(temp_array.shape[2]):
        temp_array[:,:,x]=cv2.flip(temp_array[:,:,x],0)
    image_levelset_data1=temp_array
    array_mask = nib.Nifti1Image(image_levelset_data1, affine=image_nib_nii_file.affine, header=image_nib_nii_file.header)
    niigzfilenametosave2=os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
    nib.save(array_mask, niigzfilenametosave2)
    print("image_nib_nii_file_data.shape: {}::image_levelset_data1.shape{} ".format(image_nib_nii_file_data.shape,image_levelset_data1.shape))

    return "X"
#     if image_nib_nii_file.get_fdata().shape[1] >512:
#         pass
#     ## condition 1: when x-dimension and y-dimension differ in the same image:
#     if image_nib_nii_file.get_fdata().shape[1] != image_nib_nii_file.get_fdata().shape[0]  :
# #         size_diff_x=np.abs(image_nib_nii_file.get_fdata().shape[0]-512)
# #         size_diff_y=np.abs(image_nib_nii_file.get_fdata().shape[1]-512)

#         if image_nib_nii_file.get_fdata().shape[0] > 512: # and  image_nib_nii_file.get_fdata().shape[1] >512 :

#             if (size_diff_x % 2 )== 0 : 
#                 size_diff_x=int(size_diff_x/2)
#                 npad = ((size_diff-1, size_diff+1), (size_diff-1, size_diff+1), (0, 0)) 
#             else :
#                 size_diff=int(size_diff/2)
#                 npad = ((size_diff-1, size_diff+1), (size_diff, size_diff+1), (0, 0))  #abs(np.min(image_levelset_data)
#             image_levelset_data1=np.pad(image_levelset_data, pad_width=npad, mode='constant', constant_values=np.min(image_levelset_data)) 
#         elif image_nib_nii_file.get_fdata().shape[0] < 512 and  image_nib_nii_file.get_fdata().shape[1] < 512 :
#             pass
#         elif image_nib_nii_file.get_fdata().shape[0] > 512 and  image_nib_nii_file.get_fdata().shape[1] < 512 :
#             pass
#         elif image_nib_nii_file.get_fdata().shape[0] < 512 and  image_nib_nii_file.get_fdata().shape[1] > 512 :
#             pass

#     ## condition 1a: when x and y-dimension is > 512 
#     ## condition 1b: when x and y-dimension is < 512 
#     ## condition 1c: when x > 512 and y-dimension is < 512 
#     ## condition 1d: when x < 512 and y-dimension is > 512 

#     ## condition 2: when x-dimension and y-dimension are same in the image:
#     ## condition 1a: when x and y-dimension is > 512 
#     ## condition 1b: when x and y-dimension is < 512 
#     if image_nib_nii_file.get_fdata().shape[1] == image_nib_nii_file.get_fdata().shape[0]  : 

#         if image_nib_nii_file.get_fdata().shape[1]>512  : #image_levelset_data.shape[1]:
#     #    print("a")
#             print("I am in whenOFsize512x5xx_new()")
#             print("ORIGINAL IMAGE SIZE")
#             print(image_nib_nii_file.get_fdata().shape)
#             print("YES DIFFER")
#             # print(imagefile_levelset)
#     #    npad = ((0, 0), (difference_size//2, difference_size//2), (0, 0))
#             size_diff=np.abs(image_nib_nii_file.get_fdata().shape[1]-512)

#             if (size_diff % 2 )== 0 : 
#                 size_diff=int(size_diff/2)
#                 npad = ((size_diff-1, size_diff+1), (size_diff-1, size_diff+1), (0, 0)) 
#             else :
#                 size_diff=int(size_diff/2)
#                 npad = ((size_diff-1, size_diff+1), (size_diff, size_diff+1), (0, 0))  #abs(np.min(image_levelset_data)
#             image_levelset_data1=np.pad(image_levelset_data, pad_width=npad, mode='constant', constant_values=np.min(image_levelset_data)) 
#             array_mask = nib.Nifti1Image(image_levelset_data1, affine=image_nib_nii_file.affine, header=image_nib_nii_file.header)
#             niigzfilenametosave2=os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
#             nib.save(array_mask, niigzfilenametosave2)
#         if image_nib_nii_file.get_fdata().shape[1]<512  : #image_levelset_data.shape[1]:
#     #    print("a")
#             print("I am in whenOFsize512x5xx_new()")
#             print("ORIGINAL IMAGE SIZE")
#             print(image_nib_nii_file.get_fdata().shape)
#             print("YES DIFFER")

#     #    npad = ((0, 0), (difference_size//2, difference_size//2), (0, 0))
#             size_diff=np.abs(image_nib_nii_file.get_fdata().shape[1]-512)
#             print("size_diff:{}".format(size_diff))
#             if (size_diff % 2 )== 0 : 
#                 size_diff=int(size_diff/2)
#     #             npad = ((0, 0), (size_diff-1, size_diff+1), (0, 0))
#                 image_levelset_data1=image_levelset_data[0+size_diff:image_levelset_data.shape[0]-size_diff,0+size_diff:image_levelset_data.shape[1]-size_diff,0:image_levelset_data.shape[2]]
#             else :
#                 size_diff=int(size_diff/2)
#                 image_levelset_data1=image_levelset_data[0+size_diff:image_levelset_data.shape[0]-size_diff,0+size_diff:image_levelset_data.shape[1]-size_diff,0:image_levelset_data.shape[2]] 
######################################################################################################

def levelset2originalRF() : #original_file,levelset_file,OUTPUT_DIRECTORY="./"):
    original_file=sys.argv[1]
    levelset_file=sys.argv[2]
    OUTPUT_DIRECTORY=sys.argv[3]
    original_file_nib=nib.load(original_file)
    original_file_nib_data=original_file_nib.get_fdata()
    if original_file_nib_data.shape[1] == 512 :
        whenOFsize512x512(levelset_file,OUTPUT_DIRECTORY)
    else:
        whenOFsize512x5xx(original_file,levelset_file,OUTPUT_DIRECTORY)

def levelset2originalRF_new() : #original_file,levelset_file,OUTPUT_DIRECTORY="./"):
#     print(sys.argv[1])

    original_file=sys.argv[1]
    levelset_file=sys.argv[2]
    OUTPUT_DIRECTORY=sys.argv[3]
    original_file_nib=nib.load(original_file)

    original_file_nib_data=original_file_nib.get_fdata()
    print("For the file {}".format(levelset_file))
    print("I am in levelset2originalRF_new()")
    print("original_file_nib_data.shape[1]: {}".format(original_file_nib_data.shape[1]))
    if original_file_nib_data.shape[1] == 512 :
        whenOFsize512x512_new(levelset_file,original_file,OUTPUT_DIRECTORY)
    else:
        whenOFsize512x5xx_new(original_file,levelset_file,OUTPUT_DIRECTORY)

def levelset2originalRF_new_flip() : #original_file,levelset_file,OUTPUT_DIRECTORY="./"):
#     print(sys.argv[1])

    original_file=sys.argv[1]
    levelset_file=sys.argv[2]
    OUTPUT_DIRECTORY=sys.argv[3]
    original_file_nib=nib.load(original_file)

    original_file_nib_data=original_file_nib.get_fdata()
    print("For the file {}".format(levelset_file))
    print("I am in levelset2originalRF_new_flip()")
    print("original_file_nib_data.shape[1]: {}".format(original_file_nib_data.shape[1]))
    if original_file_nib_data.shape[1] == 512 :
        whenOFsize512x512_new_flip(levelset_file,original_file,OUTPUT_DIRECTORY)
    else:
        whenOFsize512x5xx_new_flip(original_file,levelset_file,OUTPUT_DIRECTORY)
def levelset2originalRF_new_flip_with_params(original_file,levelset_file,OUTPUT_DIRECTORY) : #original_file,levelset_file,OUTPUT_DIRECTORY="./"):
    #     print(sys.argv[1])

    # original_file=sys.argv[1]
    # levelset_file=sys.argv[2]
    # OUTPUT_DIRECTORY=sys.argv[3]
    original_file_nib=nib.load(original_file)

    original_file_nib_data=original_file_nib.get_fdata()
    print("For the file {}".format(levelset_file))
    print("I am in levelset2originalRF_new_flip()")
    print("original_file_nib_data.shape[1]: {}".format(original_file_nib_data.shape[1]))
    if original_file_nib_data.shape[1] == 512 :
        whenOFsize512x512_new_flip(levelset_file,original_file,OUTPUT_DIRECTORY)
    else:
        whenOFsize512x5xx_new_flip(original_file,levelset_file,OUTPUT_DIRECTORY)

def levelset2originalRF_new_py(original_file,levelset_file,OUTPUT_DIRECTORY) : #original_file,levelset_file,OUTPUT_DIRECTORY="./"):
#     print(sys.argv[1])

#     original_file=sys.argv[1]
#     levelset_file=sys.argv[2]
#     OUTPUT_DIRECTORY=sys.argv[3]
    original_file_nib=nib.load(original_file)

    original_file_nib_data=original_file_nib.get_fdata()
    print("For the file {}".format(levelset_file))
    print("I am in levelset2originalRF_new()")
    print("original_file_nib_data.shape[1]: {}".format(original_file_nib_data.shape[1]))
    if original_file_nib_data.shape[1] == 512 :
        whenOFsize512x512_new(levelset_file,original_file,OUTPUT_DIRECTORY)
    else:
        whenOFsize512x5xx_new(original_file,levelset_file,OUTPUT_DIRECTORY)
def hdr2niigz() : #,headerfiledata): hdrfilename,niigzfilenametosave
    filenameniigz=sys.argv[1]
    original_grayfile=sys.argv[2]
    original_grayfile_nib=nib.load(original_grayfile)
    niigzfilenametosave=sys.argv[3]
    # hdrfilename
    analyzedata=nib.AnalyzeImage.from_filename(filenameniigz)
    array_img = nib.Nifti1Image(analyzedata.dataobj.get_unscaled(), affine=original_grayfile_nib.affine, header=original_grayfile_nib.header)
    nib.save(array_img, niigzfilenametosave)


def hdr2niigz_py(filenameniigz,original_grayfile,niigzfilenametosave) : #,headerfiledata): hdrfilename,niigzfilenametosave
#     filenameniigz=sys.argv[1]
#     original_grayfile=sys.argv[2]
    original_grayfile_nib=nib.load(original_grayfile)
#     niigzfilenametosave=sys.argv[3]
    # hdrfilename
    analyzedata=nib.AnalyzeImage.from_filename(filenameniigz)
    array_img = nib.Nifti1Image(analyzedata.dataobj.get_unscaled(), affine=original_grayfile_nib.affine, header=original_grayfile_nib.header)
    nib.save(array_img, niigzfilenametosave)
# def resizeinto_512by512(image_nib_nii_file_data):
#     if image_nib_nii_file_data.shape[1]>512 : #image_levelset_data.shape[1]:
#     #    print("a")
#         print("YES DIFFER")
# #        print(imagefile_levelset)
#     #    npad = ((0, 0), (difference_size//2, difference_size//2), (0, 0))
#         size_diff=(image_nib_nii_file_data.shape[1]-512)
#         if (size_diff % 2 )== 0 : 
#             size_diff=int(size_diff/2)
#     #            npad = ((0, 0), (size_diff-1, size_diff+1), (0, 0)) 
#             image_nib_nii_file_data=image_nib_nii_file_data[0:image_nib_nii_file_data.shape[0], size_diff:(image_nib_nii_file_data.shape[1]-size_diff), 0:image_nib_nii_file_data.shape[2]]  #+ 1024 #np.pad(image_levelset_data, pad_width=npad, mode='constant', constant_values=np.min(image_levelset_data)) #image_nib_nii_file_data[0:image_nib_nii_file_data.shape[0],0+(difference_size//2):image_nib_nii_file_data.shape[1]-(difference_size//2),0:image_nib_nii_file_data.shape[2]]
#             print("I am EVEN")
#         else :
#             size_diff=int(size_diff/2)
#             image_nib_nii_file_data=image_nib_nii_file_data[0:image_nib_nii_file_data.shape[0], size_diff:(image_nib_nii_file_data.shape[1]-size_diff-1), 0:image_nib_nii_file_data.shape[2]] # +1024 #np.pad(image_levelset_data, pad_width=npad, mode='constant', constant_values=np.min(image_levelset_data)) #image_nib_nii_file_data[0:image_nib_nii_file_data.shape[0],0+(difference_size//2):image_nib_nii_file_data.shape[1]-(difference_size//2),0:image_nib_nii_file_data.shape[2]]
#     return image_nib_nii_file_data
def resizeinto_512by512(image_nib_nii_file_data):
    print('I am in utilities_simple_trimmed.py')
    size_diff_x=np.abs(image_nib_nii_file_data.shape[0]-512)
    size_diff_y=np.abs(image_nib_nii_file_data.shape[1]-512)
    temp_array=np.copy(image_nib_nii_file_data)

    if image_nib_nii_file_data.shape[0] <512:
        if (size_diff_x % 2 )== 0 :
            size_diff_x=int(size_diff_x/2)
            npad = ((size_diff_x-1, size_diff_x+1), (0, 0), (0, 0))
        else :
            size_diff_x=int(size_diff_x/2)
            npad = ((size_diff_x, size_diff_x+1), (0, 0), (0, 0))  #abs(np.min(image_levelset_data)
        temp_array=np.pad(temp_array, pad_width=npad, mode='constant', constant_values=np.min(temp_array))
    if image_nib_nii_file_data.shape[1] <512:
        if (size_diff_y % 2 )== 0 :
            size_diff_y=int(size_diff_y/2)
            npad = ((0, 0),(size_diff_y-1, size_diff_y+1),  (0, 0))
        else :
            size_diff_y=int(size_diff_y/2)
            npad = ( (0, 0),(size_diff_y, size_diff_y+1), (0, 0))  #abs(np.min(image_levelset_data)
        temp_array=np.pad(temp_array, pad_width=npad, mode='constant', constant_values=np.min(temp_array))

    if image_nib_nii_file_data.shape[0] > 512:
        if (size_diff_x % 2 )== 0 :
            size_diff_x=int(size_diff_x/2)
            temp_array=temp_array[size_diff_x:temp_array.shape[0]-size_diff_x,0:temp_array.shape[1],0:temp_array.shape[2]]
        else :
            size_diff_x=int(size_diff_x/2)
            temp_array=temp_array[size_diff_x:temp_array.shape[0]-size_diff_x-1,0:temp_array.shape[1],0:temp_array.shape[2]]

    if image_nib_nii_file_data.shape[1] > 512:
        if (size_diff_y % 2 )== 0 :
            size_diff_y=int(size_diff_y/2)
            temp_array=temp_array[0:temp_array.shape[0],size_diff_y:temp_array.shape[1]-size_diff_y,0:temp_array.shape[2]]
        else :
            size_diff_x=int(size_diff_x/2)
            temp_array=temp_array[0:temp_array.shape[0],size_diff_y:temp_array.shape[1]-size_diff_y-1,0:temp_array.shape[2]]

    image_nib_nii_file_data=temp_array
    return image_nib_nii_file_data


def rotate_image(img,center1=[0,0],angle=0):
    (h,w)= (img.shape[0],img.shape[1])
    scale = 1.0
    # calculate the center of the image
#    center = (w / 2, h / 2)
    center = (center1[0], center1[1])
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotatedimg = cv2.warpAffine(img, M, (h, w), flags= cv2.INTER_NEAREST)
    return rotatedimg

def rotate_around_point_highperf(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point.
    
    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy
def angle_bet_two_vector(v1,v2):
#    angle = np.arctan2(np.linalg.norm(np.cross(v1,v2)), np.dot(v1,v2))
    angle =(np.arctan2(v2[1], v2[0]) -  np.arctan2(v1[1], v1[0]))* 180 / np.pi
    return angle
def angle_bet_two_vectorRad(v1,v2):
#    angle = np.arctan2(np.linalg.norm(np.cross(v1,v2)), np.dot(v1,v2))
    angle =np.arctan2(v2[1], v2[0]) -  np.arctan2(v1[1], v1[0])
    return angle

def copy_nifti_parameters_scaleintensity_1(file,output_directoryname):
#     file = sys.argv[1]
#     output_directoryname=  sys.argv[2]
#    files=glob.glob(directoryname+"/*_levelset.nii.gz")
#    for file in files:
    print(file)
#        template="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/CLINICALMASTER/scct_strippedResampled1.nii.gz"
    target= file #sys.argv[2] #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/CTswithObviousMLS/datafromyasheng/NIIGZFILES/Helsinki2000_414_02052013_1437_Head_4.0_ax_Tilt_1_levelset.nii.gz"
    #savefile_extension=sys.argv[3] #"gray" #sys.argv[3]
#        template_nii=nib.load(template)
    target_nii= nib.load(target)
    target_save=os.path.basename(target)#.split(".nii")[0] + savefile_extension + ".nii.gz"
    print(os.path.join(output_directoryname,target_save))
    new_header=target_nii.header
    new_data=target_nii.dataobj.get_unscaled()-1024.0
    new_header['glmax']=np.max(new_data)
    new_header['glmin']=np.min(new_data)
    print("changed header and save too")
#        new_header['dim']= target_nii.header['dim']
#        new_header['pixdim']= target_nii.header['pixdim']
    array_img = nib.Nifti1Image(new_data,affine=target_nii.affine, header=new_header)
    nib.save(array_img, os.path.join(output_directoryname,target_save))
    return "x"

def copy_nifti_parameters_scaleintensity_sh():
    file = sys.argv[1]
    output_directoryname=  sys.argv[2]
#    files=glob.glob(directoryname+"/*_levelset.nii.gz")
#    for file in files:
    print(file)
#        template="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/CLINICALMASTER/scct_strippedResampled1.nii.gz"
    target= file #sys.argv[2] #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/CTswithObviousMLS/datafromyasheng/NIIGZFILES/Helsinki2000_414_02052013_1437_Head_4.0_ax_Tilt_1_levelset.nii.gz"
    #savefile_extension=sys.argv[3] #"gray" #sys.argv[3]
#        template_nii=nib.load(template)
    target_nii= nib.load(target)
    target_save=os.path.basename(target)#.split(".nii")[0] + savefile_extension + ".nii.gz"
    print(os.path.join(output_directoryname,target_save))
    new_header=target_nii.header
#        new_header['dim']= target_nii.header['dim']
#        new_header['pixdim']= target_nii.header['pixdim']
    array_img = nib.Nifti1Image(target_nii.dataobj.get_unscaled()-1024.0,affine=target_nii.affine, header=new_header)
    nib.save(array_img, os.path.join(output_directoryname,target_save))
    return "x"

def dummy_copy_nifti_parameters_scaleintensity_sh():
    file = sys.argv[1]
    output_directoryname=  sys.argv[2]
#    files=glob.glob(directoryname+"/*_levelset.nii.gz")
#    for file in files:
    print(file)
#        template="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/CLINICALMASTER/scct_strippedResampled1.nii.gz"
    target= file #sys.argv[2] #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/CTswithObviousMLS/datafromyasheng/NIIGZFILES/Helsinki2000_414_02052013_1437_Head_4.0_ax_Tilt_1_levelset.nii.gz"
    #savefile_extension=sys.argv[3] #"gray" #sys.argv[3]
#        template_nii=nib.load(template)
    target_nii= nib.load(target)
    target_save=os.path.basename(target)#.split(".nii")[0] + savefile_extension + ".nii.gz"
    print(os.path.join(output_directoryname,target_save))
    new_header=target_nii.header
#        new_header['dim']= target_nii.header['dim']
#        new_header['pixdim']= target_nii.header['pixdim']
    array_img = nib.Nifti1Image(target_nii.get_fdata(),affine=target_nii.affine, header=new_header)
    nib.save(array_img, os.path.join(output_directoryname,target_save))
    return "x"


def betgrayfrombetbinary1_sh():
    inputfile=sys.argv[1]
    bet_inputfile_dir=sys.argv[2]
    betgrayfile=os.path.join(bet_inputfile_dir,os.path.basename(inputfile).split(".nii.gz")[0] + "_bet.nii.gz")   #sys.argv[2]
    ## take the grayscalefiles in the inputdirectory betgrayfile== betbinary
#    allgrayfiles=glob.glob(inputdirectory+ "/*" + betgrayfileext)
#    for eachgrayfiles in allgrayfiles:
    eachgrayfiles=inputfile
    niifilenametosave=os.path.join("/output",os.path.basename(inputfile).split(".nii")[0] + "_brain_f.nii.gz") #outputfilename #os.path.join(outputdirectory,os.path.basename(eachgrayfiles).split(".nii")[0]+"_bet.nii.gz")
    print('eachgrayfiles')
    print(eachgrayfiles)
    gray_nifti=nib.load(eachgrayfiles)
    gray_nifti_data=gray_nifti.get_fdata()
    bet_nifti=nib.load(betgrayfile)
    bet_nifti_data=bet_nifti.get_fdata()
    gray_nifti_data[bet_nifti_data<np.max(bet_nifti_data)]=np.min(gray_nifti_data)
    array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
    nib.save(array_img, niifilenametosave)
    return niifilenametosave

def betgrayfrombetbinary1_sh_v1():
    inputfile=sys.argv[1]
    bet_inputfile_dir=sys.argv[2]
    output_directory=sys.argv[3]
    betgrayfile=os.path.join(bet_inputfile_dir,os.path.basename(inputfile).split(".nii")[0] + "_bet.nii.gz")   #sys.argv[2]
    ## take the grayscalefiles in the inputdirectory betgrayfile== betbinary
#    allgrayfiles=glob.glob(inputdirectory+ "/*" + betgrayfileext)
#    for eachgrayfiles in allgrayfiles:
    if os.path.exists(betgrayfile):
        eachgrayfiles=inputfile
        niifilenametosave=os.path.join(output_directory,os.path.basename(inputfile).split(".nii")[0] + "_brain_f.nii.gz") #outputfilename #os.path.join(outputdirectory,os.path.basename(eachgrayfiles).split(".nii")[0]+"_bet.nii.gz")
        print('eachgrayfiles')
        print(eachgrayfiles)
        gray_nifti=nib.load(eachgrayfiles)
        gray_nifti_data=gray_nifti.dataobj.get_unscaled() #.get_fdata()
        bet_nifti=nib.load(betgrayfile)
        bet_nifti_data=bet_nifti.dataobj.get_unscaled() #.get_fdata()
        gray_nifti_data[bet_nifti_data<np.max(bet_nifti_data)]=np.min(gray_nifti_data)
        array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
        nib.save(array_img, niifilenametosave)
        return niifilenametosave
    else:
        print("BET FILE DOES NOT EXIST")

def betgrayfrombetbinary1_sh_v2():
    inputfile=sys.argv[1]
#    bet_inputfile_dir=os.path.dirname(sys.argv[2])
    output_directory=sys.argv[3]
    betgrayfile=sys.argv[2] #os.path.join(bet_inputfile_dir,os.path.basename(inputfile).split(".nii.gz")[0] + "_bet.nii.gz")   #sys.argv[2]
    ## take the grayscalefiles in the inputdirectory betgrayfile== betbinary
#    allgrayfiles=glob.glob(inputdirectory+ "/*" + betgrayfileext)
#    for eachgrayfiles in allgrayfiles:
    eachgrayfiles=inputfile
    niifilenametosave=os.path.join(output_directory,os.path.basename(inputfile).split(".nii")[0] + "_brain_f.nii.gz") #outputfilename #os.path.join(outputdirectory,os.path.basename(eachgrayfiles).split(".nii")[0]+"_bet.nii.gz")
    print('eachgrayfiles')
    print(eachgrayfiles)
    gray_nifti=nib.load(eachgrayfiles)
    gray_nifti_data=gray_nifti.get_fdata()
    bet_nifti=nib.load(betgrayfile)
    bet_nifti_data=bet_nifti.get_fdata()
    gray_nifti_data[bet_nifti_data<np.max(bet_nifti_data)]=np.min(gray_nifti_data)
    array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
    nib.save(array_img, niifilenametosave)
    return niifilenametosave
def betgrayfrombetbinary1_sh_v3():
    inputfile_gray=sys.argv[1]
    # inputfile_bet=sys.argv[2]
    # bet_inputfile_dir=sys.argv[2]
    output_directory=sys.argv[3]
    betgrayfile=sys.argv[2] #os.path.join(bet_inputfile_dir,os.path.basename(inputfile).split(".nii")[0] + "_bet.nii.gz")   #sys.argv[2]
    ## take the grayscalefiles in the inputdirectory betgrayfile== betbinary
#    allgrayfiles=glob.glob(inputdirectory+ "/*" + betgrayfileext)
#    for eachgrayfiles in allgrayfiles:
    if os.path.exists(betgrayfile):
        eachgrayfiles=inputfile_gray
        niifilenametosave=os.path.join(output_directory,os.path.basename(inputfile_gray).split(".nii")[0] + "_brain_f.nii.gz") #outputfilename #os.path.join(outputdirectory,os.path.basename(eachgrayfiles).split(".nii")[0]+"_bet.nii.gz")
        print('eachgrayfiles')
        print(eachgrayfiles)
        gray_nifti=nib.load(eachgrayfiles)
        gray_nifti_data=gray_nifti.get_fdata() #.get_unscaled() #.get_fdata() dataobj.
        bet_nifti=nib.load(betgrayfile)
        bet_nifti_data=bet_nifti.dataobj.get_unscaled() #.get_fdata()
        gray_nifti_data[bet_nifti_data<np.max(bet_nifti_data)]= 0 #np.min(gray_nifti_data)
        array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
        nib.save(array_img, niifilenametosave)
        return niifilenametosave
    else:
        print("BET FILE DOES NOT EXIST")

def latex_start(filename):
    file1 = open(filename,"w")
    file1.writelines("\\documentclass{article}\n")
    file1.writelines("\\usepackage[margin=0.5in]{geometry}\n")
    file1.writelines("\\usepackage{graphicx}\n")
    file1.writelines("\\usepackage[T1]{fontenc} \n")
    file1.writelines("\\usepackage{datetime} \n")
    file1.writelines("\\usepackage{booktabs} \n")
    file1.writelines("\\usepackage{xcolor} \n")
    # file1.writelines("\\usepackage{latexcolors} \n")
    file1.writelines("\\definecolor{darkmagenta}{rgb}{0.55, 0.0, 0.55}\n")
    file1.writelines("\\definecolor{fuchsia}{rgb}{1.0, 0.0, 1.0}\n")
    file1.writelines("\\definecolor{upmaroon}{rgb}{0.48, 0.07, 0.07}\n")
    file1.writelines("\\definecolor{lime(colorwheel)}{rgb}{0.75, 1.0, 0.0}\n")
    file1.writelines("\\definecolor{aqua}{rgb}{0.0, 1.0, 1.0}\n")
    file1.writelines("\\definecolor{olive}{rgb}{0.5, 0.5, 0.0}\n") #006B3C
    file1.writelines("\\definecolor{cadmiumgreen}{rgb}{0.0, 0.42, 0.24}\n")




#    file1.writelines("\\begin{document}\n")
    return file1
def call_latex_start(args):
    returnvalue=0
    try:
        filename=args.stuff[1]
        latex_start(filename)
        latex_begin_document(filename)
        command="echo successful at :: {}::filename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_latex_start')
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    print(returnvalue)
    return  returnvalue
def create_a_latex_filename(filename_prefix,filename_to_write):
    returnvalue=0
    try:
        now=time.localtime()
        date_time = time.strftime("_%m_%d_%Y",now)
        latexfilename=filename_prefix +"_"+ Version_Date + date_time+".tex"
        returnvalue=latexfilename
        left_right_ratio_df=pd.DataFrame([latexfilename])
        left_right_ratio_df.columns=['latexfilename']
        left_right_ratio_df.to_csv(filename_to_write,index=False)
        command="echo successful at :: {}::filename::{} >> /software/error.txt".format(inspect.stack()[0][3],'create_a_latex_filename')
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    print(returnvalue)
    return  returnvalue

def call_create_a_latex_filename(args):
    returnvalue=0
    try:
        filename_prefix=args.stuff[1]
        filename_to_write=args.stuff[2]
        create_a_latex_filename(filename_prefix,filename_to_write)
        command="echo successful at :: {}::filename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_create_a_latex_filename')
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    print(returnvalue)
    return  returnvalue
def write_panda_df(latexfilename,table_df):
    latex_start_tableNc_noboundary(latexfilename,1)
    latex_insert_line_nodek(latexfilename,text=table_df.to_latex(index=False))
    latex_end_table2c(latexfilename)
    return
def remove_a_column(csvfilename,columnnamelist,outputfilename):
    returnvalue=0
    try:
        csvfilename_df=pd.read_csv(csvfilename)
        csvfilename_df = csvfilename_df.drop(columnnamelist, axis=1)
        csvfilename_df.to_csv(outputfilename,index=False)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    return  returnvalue
def call_remove_a_column(args):
    returnvalue=0
    try:
        csvfilename=args.stuff[1]
        columnnamelist=args.stuff[3:]
        outputfilename=args.stuff[2]
        remove_a_column(csvfilename,columnnamelist,outputfilename)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    return  returnvalue
def write_a_col_on_tex(args):
    returnvalue=0

    try:
        # table_df=pd.read_csv(args.stuff[1])
        latexfilename=args.stuff[1]
        column_name=args.stuff[2]
        column_value=args.stuff[3]

        # try:
        column_value_df=pd.DataFrame([column_value])
        column_value_df.columns=[str(column_name)]
        # session_label_df = pd.DataFrame(table_df.pop(str(column_name)))
        write_panda_df(latexfilename,column_value_df)
        returnvalue=1
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'write_a_col_on_tex')
        subprocess.call(command,shell=True)
        # except:
        #     pass


    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    return returnvalue
def csvtable_on_tex(args):
    returnvalue=0

    try:
        table_df=pd.read_csv(args.stuff[1])
        latexfilename=args.stuff[2]
        write_panda_df(latexfilename,table_df)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_write_panda_df')
        subprocess.call(command,shell=True)
        returnvalue=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    return returnvalue
def call_write_panda_df(args):
    returnvalue=0

    try:
        table_df=pd.read_csv(args.stuff[1])
        latexfilename=args.stuff[2]

        # title_df=pd.DataFrame([os.path.basename(args.stuff[1]).split('.csv')[0]])
        # title_df.columns=["FILENAME"]

        # table_df['FILENAME']=os.path.basename(args.stuff[1]).split('.csv')[0]

        try:
            session_label_df = pd.DataFrame(table_df.pop('SESSION_LABEL'))
            write_panda_df(latexfilename,session_label_df)
        except:
            pass
        try:
            title_df = pd.DataFrame(table_df.pop('FILENAME'))
            write_panda_df(latexfilename,title_df)
        except:
            pass
        try:
            SLICE_NUM_df = pd.DataFrame(table_df.pop('SLICE_NUM'))
            write_panda_df(latexfilename,SLICE_NUM_df)
        except:
            pass
        # table_df.insert(0, 'FILENAME', column_to_move)
        table_df1=table_df.unstack().reset_index()
        table_df1.columns=["Region","IDX","Volume(ml)"]
        table_df1=table_df1.drop(["IDX"],axis=1)

        # write_panda_df(latexfilename,session_label_df)
        # write_panda_df(latexfilename,title_df)
        write_panda_df(latexfilename,table_df1)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_write_panda_df')
        subprocess.call(command,shell=True)
        returnvalue=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    return returnvalue
def latex_end(filename):
    file1 = open(filename,"a")
    file1.writelines("\\end{document}\n")
    file1.close()
    return "X"
def call_latex_end(args):
    returnvalue=0
    try:
        filename=args.stuff[1]
        latex_end(filename)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_latex_insertimage_tableNc')
        subprocess.call(command,shell=True)
        returnvalue=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    # print(returnvalue)
    return  returnvalue
def latex_begin_document(filename):
    file1 = open(filename,"a")
    file1.writelines("\\begin{document}\n")
    return file1
def latex_write_items(filename,text_items):
    file1 = open(filename,"a")
    file1.writelines("\\begin{itemize}\n")
    for each_text_item in text_items:
        file1.writelines("\\item  " + each_text_item+ "\n")

    file1.writelines("\\end{itemize}\n")
    return file1
def latex_insert_line(filename,text="ATUL KUMAR"):
    command= "sed -i 's#\\end{document}##'   " +  filename
    subprocess.call(command,shell=True)
    file1 = open(filename,"a")
    currentDT = str(datetime.datetime.now())
    file1.writelines("\\section{" + currentDT + "}" ) #\\today : \\currenttime}")
#    file1.writelines("\\date{\\today}")
    file1.writelines("\\detokenize{")
    file1.writelines(text)
    file1.writelines("\n")
    file1.writelines("}")
    return file1
def latex_insert_line_nodek(filename,text="ATUL KUMAR"):
    command= "sed -i 's#\\end{document}##'   " +  filename
    subprocess.call(command,shell=True)
    file1 = open(filename,"a")
    file1.writelines(text)
    file1.writelines("\n")

    return file1
def latex_insert_line_nodate(filename,text="ATUL KUMAR"):
    command= "sed -i 's#\\end{document}##'   " +  filename
    subprocess.call(command,shell=True)
    file1 = open(filename,"a")
#    currentDT = str(datetime.datetime.now())
#    file1.writelines("\\section{" + currentDT + "}" ) #\\today : \\currenttime}")
#    file1.writelines("\\date{\\today}")
    file1.writelines("\\detokenize{")
    file1.writelines(text)
    file1.writelines("\n")
    file1.writelines("}")
    return file1
def writetolabnotebook(labnotebook,text):
#    latex_start(labnotebook)
#    latex_begin_document(labnotebook)
    latex_insert_line(labnotebook,text)
    latex_end(labnotebook)
def writetoanewlabnotebook(labnotebook):
    latex_start(labnotebook)
    latex_begin_document(labnotebook)
    latex_insert_line(labnotebook,"THIS IS A NEW LABNOTEBOOK")
    latex_end(labnotebook)
#def write_latex_body(labnotebook,text="HELLO WORLD"):
#    command= "sed -i 's#\\end{document}##'   " +  labnotebook
#    subprocess.call(command,shell=True)
#    command= "echo  " + "\\\section >>   " +  labnotebook
#    subprocess.call(command,shell=True)
#    command= 'echo $(date) >>   ' +  labnotebook
#    subprocess.call(command,shell=True)
#    command= 'echo  ' + text  +  '>>   ' +  labnotebook
#    subprocess.call(command,shell=True)
#    return "x"
#
#def write_latex_end(labnotebook):
#    command= "echo  " + "\\\end{document}  >>  " +  labnotebook
#    subprocess.call(command,shell=True)
#    return "x"

def combinecsvs(inputdirectory,outputdirectory,outputfilename):
    outputfilepath=os.path.join(outputdirectory,outputfilename)
    extension = 'csv'
    all_filenames = [i for i in glob.glob(os.path.join(inputdirectory,'*.{}'.format(extension)))]
#    os.chdir(inputdirectory)
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    #export to csv
    combined_csv.to_csv(outputfilepath, index=False, encoding='utf-8-sig')

def combinecsvs_sh():
    inputdirectory=sys.argv[1]
    outputdirectory=sys.argv[2]
    outputfilename=sys.argv[3]
    outputfilepath=os.path.join(outputdirectory,outputfilename)
    extension = 'csv'
    all_filenames = [i for i in glob.glob(os.path.join(inputdirectory,'*.{}'.format(extension)))]
#    os.chdir(inputdirectory)
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    #export to csv
    combined_csv.to_csv(outputfilepath, index=False, encoding='utf-8-sig')

def write_csv(csv_file_name,csv_columns,data_csv):
    try:
        with open(csv_file_name, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in data_csv:
                print("data")
                print(data)
                writer.writerow(data)
    except IOError:
        print("I/O error")

def diff_two_csv(file1,file2,outputfile="diff.csv"):


    return "XX"

def write_tex_im_in_afolder(foldername,max_num_img,img_ext="*.png"):
    # get the folder name
#    foldername="" # complete path
    # start writing tex file
    latexfilename=foldername+".tex"
    latex_start(latexfilename)
    latex_begin_document(latexfilename)
    # for each image file in the folder insert text to include the image a figure
    png_files=glob.glob(os.path.join(foldername,img_ext))
    counter=0
    for each_png_file in png_files:
        if counter < max_num_img:
            thisfilebasename=os.path.basename(each_png_file)
            latex_start_table1c(latexfilename)
            latex_insertimage_table1c(latexfilename,image1=each_png_file,caption= thisfilebasename.split('.png'),imagescale=0.3)
            latex_end_table2c(latexfilename)
            latex_insert_line_nodate(latexfilename, thisfilebasename.split('.png')[0] )
            counter=counter+1
    latex_end(latexfilename)
#    command= "mv  " + latexfilename.split('.')[0] + "*     " + os.path.dirname(foldername)
#    subprocess.call(command,shell=True)    

def filename_replace_dots(foldername,img_ext):
    files=glob.glob(os.path.join(foldername,"*"+img_ext))
    for each_file in files:
        each_f_basename=os.path.basename(each_file)
        each_f_basename_wo_ext=each_f_basename.split(img_ext)
        each_f_basename_wo_extNew= each_f_basename_wo_ext[0].replace(".","_") #re.sub('[^a-zA-Z0-9 \n\.]', '_', each_f_basename_wo_ext[0])
        each_f_basename_new=each_f_basename_wo_extNew + img_ext
        each_f_basename_new_w_path=os.path.join(foldername,each_f_basename_new)
        command = "mv   "  + each_file  + "   " + each_f_basename_new_w_path
        print(each_file)
#        print()
        subprocess.call(command,shell=True)
def filename_replace_dots1(foldername,img_ext):
    files=glob.glob(os.path.join(foldername,"*_"+img_ext))
    for each_file in files:
        each_f_basename=os.path.basename(each_file)
        each_f_basename_wo_ext=each_f_basename.split("_"+img_ext)
        each_f_basename_wo_extNew= each_f_basename_wo_ext[0] +"." + img_ext #.replace(".","_") #re.sub('[^a-zA-Z0-9 \n\.]', '_', each_f_basename_wo_ext[0])
        each_f_basename_new=each_f_basename_wo_extNew # + img_ext
        each_f_basename_new_w_path=os.path.join(foldername,each_f_basename_new)
        command = "mv   "  + each_file  + "   " + each_f_basename_new_w_path
        print(each_f_basename_new)
#        print()
        subprocess.call(command,shell=True)





def write_tex_im_in_3folders(foldername1,foldername2,foldername3,max_num_img,extension=".png"):
    # get the folder name
#    foldername="" # complete path
    # start writing tex file
    foldername1="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/GaborOnly"
    foldername2="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/RegistrationOnly"
    foldername3="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/RegnGabor"
    foldername=os.path.join(os.path.dirname(foldername1),os.path.basename(foldername1)+os.path.basename(foldername2)+os.path.basename(foldername3))
    latexfilename=foldername+".tex"
    latex_start(latexfilename)
    latex_begin_document(latexfilename)
    # for each image file in the folder insert text to include the image a figure
    png_files=glob.glob(os.path.join(foldername1,"*" + extension )) #.png"))
    counter=0
#    max_num_img=5

    for each_png_file in png_files:
        if counter < max_num_img:
            images=[]
            thisfilebasename=os.path.basename(each_png_file)
            path2=os.path.join(foldername2,thisfilebasename)
            path3=os.path.join(foldername3,thisfilebasename)
            if os.path.exists(path2) and os.path.exists(path3):
                images.append(each_png_file)
                images.append(path2)
                images.append(path2)
                latex_start_tableNc(latexfilename,3)
                images.append(each_png_file)
                latex_insertimage_tableNc(latexfilename,images,3,caption= thisfilebasename.split(extension),imagescale=0.3)
                latex_end_table2c(latexfilename)
                latex_insert_line_nodate(latexfilename, thisfilebasename.split(extension)[0] )
                counter=counter+1
    latex_end(latexfilename)
    command= "mv  " + latexfilename.split('.')[0] + "*     " + os.path.dirname(foldername)
    subprocess.call(command,shell=True)

def write_tex_im_in_3folders_(foldername1,foldername2,foldername3,max_num_img,extension=".png"):
    # get the folder name
#    foldername="" # complete path
    # start writing tex file
    foldername1="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/GaborOnly"
    foldername2="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/RegistrationOnly"
    foldername3="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/RegnGabor"
    foldername=os.path.join(os.path.dirname(foldername1),os.path.basename(foldername1)+os.path.basename(foldername2)+os.path.basename(foldername3))
    latexfilename=foldername+".tex"
    latex_start(latexfilename)
    latex_begin_document(latexfilename)
    # for each image file in the folder insert text to include the image a figure
    png_files=sorted(glob.glob(os.path.join(foldername1,"*GTvsGABOR*" + extension ))) #.png"))
    counter=0
#    max_num_img=5

    for each_png_file in png_files:
        if counter < max_num_img:
            images=[]
            thisfilebasename=os.path.basename(each_png_file)
            numberofslice=thisfilebasename.split("GTvsGABOR")[1].split(".jpg")[0]
            secondfile=thisfilebasename.split("GTvs")[0]+"GTvsRegist" + str(numberofslice) + ".jpg"
            thirddfile=thisfilebasename.split("GTvs")[0]+"GTvsGaborNRegist" + str(numberofslice) + ".jpg"
            path2=os.path.join(foldername2,secondfile)
            path3=os.path.join(foldername3,thirddfile)
            if os.path.exists(path2) and os.path.exists(path3):
                images.append(each_png_file)
                images.append(path2)
                images.append(path3)
                latex_start_tableNc(latexfilename,3)
                images.append(each_png_file)
                latex_insertimage_tableNc(latexfilename,images,3,caption= thisfilebasename.split(extension),imagescale=0.3)
                latex_end_table2c(latexfilename)
                latex_insert_line_nodate(latexfilename, thisfilebasename.split(extension)[0] )
                counter=counter+1
    latex_end(latexfilename)
    command= "mv  " + latexfilename.split('.')[0] + "*     " + os.path.dirname(foldername)
    subprocess.call(command,shell=True)

def write_tex_im_in_3folders_sh():
    # get the folder name
#    foldername="" # complete path
    # start writing tex file
    foldername1=sys.argv[1] #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/GaborOnly"
    foldername2=sys.argv[2] #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/RegistrationOnly"
    foldername3=sys.argv[3] #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/RegnGabor"
    foldername=os.path.join(os.path.dirname(foldername1),os.path.basename(foldername1)+os.path.basename(foldername2)+os.path.basename(foldername3))
    latexfilename=foldername+".tex"
    latex_start(latexfilename)
    latex_begin_document(latexfilename)
    # for each image file in the folder insert text to include the image a figure
    png_files=glob.glob(os.path.join(foldername1,"*.png"))
    counter=0
    max_num_img=int(sys.argv[4])

    for each_png_file in png_files:
        if counter < max_num_img:
            images=[]
            thisfilebasename=os.path.basename(each_png_file)
            path2=os.path.join(foldername2,thisfilebasename)
            path3=os.path.join(foldername3,thisfilebasename)
            if os.path.exists(path2) and os.path.exists(path3):
                images.append(each_png_file)
                images.append(path2)
                images.append(path3)
                latex_start_tableNc(latexfilename,3)
                images.append(each_png_file)
                latex_insertimage_tableNc(latexfilename,images,3,caption= thisfilebasename.split('.png'),imagescale=0.3)
                latex_end_table2c(latexfilename)
                latex_insert_line_nodate(latexfilename, thisfilebasename.split('.png')[0] )
                counter=counter+1
    latex_end(latexfilename)
    command= "mv  " + latexfilename.split('.')[0] + "*     " + os.path.dirname(foldername)
    subprocess.call(command,shell=True)
def write_tex_im_in_afolder_sh():
    foldername=sys.argv[1]
    max_num_img=int(sys.argv[2])
    # get the folder name
#    foldername="" # complete path
    # start writing tex file
    latexfilename=foldername+".tex"
    latex_start(latexfilename)
    latex_begin_document(latexfilename)
    # for each image file in the folder insert text to include the image a figure
    png_files=glob.glob(os.path.join(foldername,"*.png"))
    png_files=png_files.sort(key=os.path.getmtime)
    counter=0
    for each_png_file in png_files:
        if counter < max_num_img:
            thisfilebasename=os.path.basename(each_png_file)
            latex_start_table1c(latexfilename)
            latex_insertimage_table1c(latexfilename,image1=each_png_file,caption= thisfilebasename.split('.png'),imagescale=0.3)
            latex_end_table2c(latexfilename)
            latex_insert_line_nodate(latexfilename, thisfilebasename.split('.png')[0] )
            counter=counter+1
    latex_end(latexfilename)
    command= "mv  " + latexfilename.split('.')[0] + "*     " + os.path.dirname(foldername)
    subprocess.call(command,shell=True)

def write_tex_im_in_afolder_py(foldername,max_num_img=200,fileext="png"):

    # get the folder name
#    foldername="" # complete path
    # start writing tex file

    # for each image file in the folder insert text to include the image a figure
    png_files=glob.glob(os.path.join(foldername,"*."+ fileext))
    png_files.sort(key=os.path.getmtime)
    counter=0
    filecount=0
    latexfilename=foldername+ str(filecount) + ".tex"
    latex_start(latexfilename)
    latex_begin_document(latexfilename)
    for each_png_file in png_files:
        if counter%1000 == 0:
            latex_end(latexfilename)
            filecount=filecount+1
            latexfilename=foldername+ str(filecount) + ".tex"
            latex_start(latexfilename)
            latex_begin_document(latexfilename)
        if counter < max_num_img:
            thisfilebasename=os.path.basename(each_png_file)
            thisfilebasename_S=thisfilebasename.split("."+ fileext)
            thisfilebasename_S=thisfilebasename_S[0].replace(".","_")
            thisfilebasenameNPath=os.path.join(foldername,thisfilebasename_S+"."+ fileext)
            command= "mv   " + each_png_file +  "   " + thisfilebasenameNPath
            subprocess.call(command,shell=True)
            thisfilebasename=os.path.basename(thisfilebasenameNPath)
            latex_start_table1c(latexfilename)
            latex_insertimage_table1c(latexfilename,image1=thisfilebasenameNPath,caption= thisfilebasename.split('.' + fileext),imagescale=0.3)
            latex_end_table2c(latexfilename)
            latex_insert_line_nodate(latexfilename, thisfilebasename.split('.' + fileext)[0] )
            counter=counter+1
    latex_end(latexfilename)
    command= "mv  " + latexfilename.split('.')[0] + "*     " + os.path.dirname(foldername)
    subprocess.call(command,shell=True)

def write_tex_im_in_afolder_v1(foldername,max_num_img=200,filenamepattern=".png"):

    # get the folder name
#    foldername="" # complete path
    # start writing tex file

    # for each image file in the folder insert text to include the image a figure
    fileext=filenamepattern.split(".")[1]
    png_files=glob.glob(os.path.join(foldername,"*"+ filenamepattern))
    png_files.sort(key=os.path.getmtime)
    counter=0
    filecount=0
    latexfilename=foldername+ str(filecount) + ".tex"
    latex_start(latexfilename)
    latex_begin_document(latexfilename)
    for each_png_file in png_files:
        if counter%1000 == 0:
            latex_end(latexfilename)
            filecount=filecount+1
            latexfilename=foldername+ str(filecount) + ".tex"
            latex_start(latexfilename)
            latex_begin_document(latexfilename)
        if counter < max_num_img:
            thisfilebasename=os.path.basename(each_png_file)
            thisfilebasename_S=thisfilebasename.split("."+ fileext)
            thisfilebasename_S=thisfilebasename_S[0].replace(".","_")
            thisfilebasenameNPath=os.path.join(foldername,thisfilebasename_S+"."+ fileext)
            command= "mv   " + each_png_file +  "   " + thisfilebasenameNPath
            subprocess.call(command,shell=True)
            thisfilebasename=os.path.basename(thisfilebasenameNPath)
            latex_start_table1c(latexfilename)
            latex_insertimage_table1c(latexfilename,image1=thisfilebasenameNPath,caption= thisfilebasename.split('.' + fileext),imagescale=0.3)
            latex_end_table2c(latexfilename)
            latex_insert_line_nodate(latexfilename, thisfilebasename.split('.' + fileext)[0] )
            counter=counter+1
    latex_end(latexfilename)
    command= "mv  " + latexfilename.split('.')[0] + "*     " + os.path.dirname(foldername)
    subprocess.call(command,shell=True)

def latex_start_table2c(filename):
    print("latex_start_table2c")
    print(filename)
    file1 = open(filename,"a")
    file1.writelines("\\begin{center}\n")
    file1.writelines("\\begin{tabular}{ c c  }\n")
    return file1
def latex_start_tableNc(filename,N):
    print("latex_start_table2c")
    print(filename)
    file1 = open(filename,"a")
    file1.writelines("\\begin{center}\n")
    texttowrite=""
    for x in range(N):
        texttowrite = texttowrite + " | " + "c" + " | "
    file1.writelines("\\begin{tabular}{ " + texttowrite + "  }\n")
    return file1

def latex_start_tableNc_noboundary(filename,N):
    print("latex_start_table2c")
    print(filename)
    file1 = open(filename,"a")
    file1.writelines("\\begin{center}\n")
    texttowrite=""
    for x in range(N):
        texttowrite = texttowrite +  "c" # + " "
    file1.writelines("\\begin{tabular}{ " + texttowrite + "  }\n")
    return file1

def latex_start_tableNc_noboundary_withcolsize(filename,N,colsize=0.1):
    print("latex_start_table2c")
    print(filename)
    file1 = open(filename,"a")
    file1.writelines("\\begin{center}\n")
    texttowrite=""
    for x in range(N):
        texttowrite = texttowrite +  "p{" + str(colsize) +"\\textwidth}" # + " "
    file1.writelines("\\begin{tabular}{ " + texttowrite + "  }\n")
    return file1
def latex_start_table1c(filename):
    print("latex_start_table2c")
    print(filename)
    file1 = open(filename,"a")
    file1.writelines("\\begin{center}\n")
    file1.writelines("\\begin{tabular}{ c  }\n")
    return file1

def latex_end_table2c(filename):
    file1 = open(filename,"a")
    file1.writelines("\n")
    file1.writelines("\\end{tabular}\n")
    file1.writelines("\\end{center}\n")
    return file1
def space_between_lines(filename,space=1):
    file1 = open(filename,"a")
    file1.writelines("\n")
    file1.writelines("\\vspace{" + str(space)+"em}\n")
    return file1
def call_space_between_lines(args):

    returnvalue=0
    try:
        filename=args.stuff[1]
        space=int(args.stuff[2])
        space_between_lines(filename,space=space)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_space_between_lines')
        subprocess.call(command,shell=True)
        returnvalue=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    # print(returnvalue)
    return  returnvalue
def latex_insertimage_table2c(filename,image1="lion.jpg", image2="lion.jpg",caption="ATUL",imagescale=0.5):
    file1 = open(filename,"a")
    file1.writelines("\\includegraphics[width=" + str(imagescale) + "\\textwidth]{" + image1 + "}\n")
    file1.writelines("&")
    file1.writelines("\\includegraphics[width=" + str(imagescale) + "\\textwidth]{"+  image2 + "}\n")

    return file1
def latex_insertimage_tableNc(filename,images,N, caption="ATUL",imagescale=0.5, angle=0,space=1):
    file1 = open(filename,"a")
    # file1.writelines("\\vspace{-2em}\n")

    for x in range(N):
        if x < N-1:
            file1.writelines("\\includegraphics[angle="+ str(angle) + ",width=" + str(imagescale) + "\\textwidth]{" + images[x] + "}\n")
            file1.writelines("&")
        else:
            file1.writelines("\\includegraphics[angle="+ str(angle) + ",width=" + str(imagescale) + "\\textwidth]{" + images[x] + "}\n")

#    file1.writelines("\\includegraphics[width=" + str(imagescale) + "\\textwidth]{"+  image2 + "}\n")
#     file1.writelines("\\vspace{" + str(space)+"em}\n")
    return file1

def latex_inserttext_tableNc_colored(filename,texts,text_colors,N,space=1):
    file1 = open(filename,"a")
    # file1.writelines("\\vspace{-2em}\n")

    for x in range(N):
        if x < N-1:
            file1.writelines("\\textbf{\\textcolor{"+ text_colors[x]+"}{" + texts[x] + "}}\n")
            file1.writelines("&")
        else:
            file1.writelines("\\textbf{\\textcolor{"+ text_colors[x]+"}{" + texts[x] + "}}\n")

    #    file1.writelines("\\includegraphics[width=" + str(imagescale) + "\\textwidth]{"+  image2 + "}\n")
    # file1.writelines("\\vspace{" + str(-2*space)+"em}\n")
    return file1
def latex_inserttext_tableNc_colored_with_bullet(filename,texts,text_colors,bullet_sign_list,N,space=1):
    file1 = open(filename,"a")
    # file1.writelines("\\vspace{-2em}\n")

    for x in range(N):
        if x < N-1:
            file1.writelines("\\textbf{\\textcolor{" + text_colors[x]+"}{" + "$\\" + bullet_sign_list[x] +"$ " + texts[x] + "}}\n")
            file1.writelines("&")
        else:
            file1.writelines("\\textbf{\\textcolor{" + text_colors[x]+"}{" + "$\\" + bullet_sign_list[x] +"$ " + texts[x] + "}}\n")

    #    file1.writelines("\\includegraphics[width=" + str(imagescale) + "\\textwidth]{"+  image2 + "}\n")
    # file1.writelines("\\vspace{" + str(-2*space)+"em}\n")
    return file1
def latex_inserttext_tableNc_colored_with_item(filename,texts,text_colors,N,space=1):
    file1 = open(filename,"a")

    # file1.writelines("\\begin{itemize} \n")
    for x in range(N):
        if x < N-1:
            text_to_write="\\begin{itemize}\n"
            # for x in range(N):
            text_to_write=text_to_write +"  "  +"\\item  " + "\\textbf{\\textcolor{"+ text_colors[x]+"}{" + texts[x] + "}}" + "\n"

            text_to_write=text_to_write +"  "  +"\\end{itemize}\n"
            file1.writelines(text_to_write)
            file1.writelines("&")
        else:
            file1.writelines(text_to_write)
    # file1.writelines("\\end{itemize} \n")
    #    file1.writelines("\\includegraphics[width=" + str(imagescale) + "\\textwidth]{"+  image2 + "}\n")
    # file1.writelines("\\vspace{" + str(-2*space)+"em}\n")
    return file1

def call_latex_insertimage_tableNc(args):

    returnvalue=0
    try:
        filename=args.stuff[1]
        imagescale=float(args.stuff[2])
        angle=float(args.stuff[3])
        space=float(args.stuff[4])
        images=args.stuff[5:]
        N=len(images)
        latex_start_tableNc_noboundary(filename,N)
        latex_insertimage_tableNc(filename,images,N, caption="NONE",imagescale=imagescale, angle=angle,space=space)
        latex_end_table2c(filename)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_latex_insertimage_tableNc')
        subprocess.call(command,shell=True)
        returnvalue=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    # print(returnvalue)
    return  returnvalue
def call_latex_inserttext_tableNc_colored_with_bullet(args):

    returnvalue=0
    try:
        filename=args.stuff[1]
        # imagescale=float(args.stuff[2])
        # angle=float(args.stuff[3])
        # space=float(args.stuff[4])
        text_color_list=args.stuff[2].split('_')
        bullet_sign_list=args.stuff[3].split('_')
        texts=args.stuff[4:]
        N=len(texts)
        colsize=(1/N)
        # latex_start_tableNc_noboundary(filename,N)
        latex_start_tableNc_noboundary_withcolsize(filename,N,colsize=colsize)
        latex_inserttext_tableNc_colored_with_bullet(filename,texts,text_color_list,bullet_sign_list,N,space=1)
        # latex_inserttext_tableNc_colored_with_item(filename,texts,text_color_list,N,space=1)
        # latex_inserttext_tableNc(filename,texts,N, caption="NONE",imagescale=imagescale, angle=angle,space=space)
        latex_end_table2c(filename)
        # space_between_lines(filename,space=-2)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_latex_inserttext_tableNc')
        subprocess.call(command,shell=True)
        returnvalue=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    # print(returnvalue)
    return  returnvalue

def call_latex_inserttext_tableNc(args):

    returnvalue=0
    try:
        filename=args.stuff[1]
        # imagescale=float(args.stuff[2])
        # angle=float(args.stuff[3])
        # space=float(args.stuff[4])
        text_color_list=args.stuff[2].split('_')
        texts=args.stuff[3:]
        N=len(texts)
        colsize=(1/N)
        # latex_start_tableNc_noboundary(filename,N)
        latex_start_tableNc_noboundary_withcolsize(filename,N,colsize=colsize)
        latex_inserttext_tableNc_colored(filename,texts,text_color_list,N,space=1)
        # latex_inserttext_tableNc(filename,texts,N, caption="NONE",imagescale=imagescale, angle=angle,space=space)
        latex_end_table2c(filename)
        # space_between_lines(filename,space=-2)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_latex_inserttext_tableNc')
        subprocess.call(command,shell=True)
        returnvalue=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    # print(returnvalue)
    return  returnvalue
def latex_insertimage_tableNc_v1(filename,images,N, caption="ATUL",imagescale=0.5, angle=0,space=1):
    file1 = open(filename,"a")
    # file1.writelines("\\vspace{-2em}\n")

    for x in range(N):
        if x < N-1:
            file1.writelines("\\includegraphics[angle="+ str(angle) + ",width=" + str(imagescale) + "\\textwidth]{" + images[x] + "}\n")
            file1.writelines("&")
        else:
            file1.writelines("\\includegraphics[angle="+ str(angle) + ",width=" + str(imagescale) + "\\textwidth]{" + images[x] + "}\n")
            file1.writelines("\\" + "\\")

#    file1.writelines("\\includegraphics[width=" + str(imagescale) + "\\textwidth]{"+  image2 + "}\n")
    # file1.writelines("\\vspace{" + str(space)+"em}\n")
    return file1


def latex_inserttext_tableNc(filename,text1,N,space=1):
    file1 = open(filename,"a")
    # file1.writelines("\\vspace{-2em}\n")

    for x in range(N):
        if x < N-1:
            file1.writelines(text1[x])
            file1.writelines("&")
        else:
            file1.writelines(text1[x] + "\\" +"\\")

#    file1.writelines("\\includegraphics[width=" + str(imagescale) + "\\textwidth]{"+  image2 + "}\n")
    # file1.writelines("\\vspace{" + str(space)+"em}\n")
    return file1

def latex_insertimage_table1c(filename,image1="lion.jpg",caption="ATUL",imagescale=0.5,angle=0):
    file1 = open(filename,"a")
    file1.writelines("\\includegraphics[angle="+ str(angle) + ",width=" + str(imagescale) + "\\textwidth]{" + image1 + "}\n")
    return file1
def latex_inserttext_table2c(filename,text1="lion.jpg", text2="lion.jpg"):
    file1 = open(filename,"a")
    file1.writelines(text1)
    file1.writelines("&")
    file1.writelines(text2)

def latex_inserttext_table1c(filename,text1="lion.jpg"):
    file1 = open(filename,"a")
    file1.writelines(text1)

    return file1

def saveslicesofniftimat(filename_gray_data_np,filename,savetodir=""):
    # filename_nib=nib.load(filename)
    # filename_gray_data_np=filename_nib.get_fdata()
    min_img_gray=np.min(filename_gray_data_np)
    img_gray_data=0
    if not os.path.exists(savetodir):
        savetodir=os.path.dirname(filename)
    if min_img_gray>=0:
        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(1020, 1100))
        # img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(1000, 1200))
    else:
        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(20, 60))
        # img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(0, 200))
    for x in range(img_gray_data.shape[2]):
        cv2.imwrite(os.path.join(savetodir,os.path.basename(filename).split(".nii")[0]+str(x)+".jpg" ),img_gray_data[:,:,x]*255 )

def saveslicesofnifti(filename,savetodir=""):
    filename_nib=nib.load(filename)
    filename_gray_data_np=filename_nib.get_fdata()
    min_img_gray=np.min(filename_gray_data_np)
    img_gray_data=0
    if not os.path.exists(savetodir):
        savetodir=os.path.dirname(filename)
    if min_img_gray>=0:
        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(1020, 1100))
        # img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(1000, 1200))
    else:
        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(20, 60))
        # img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(0, 200))
    for x in range(img_gray_data.shape[2]):
        slice_number="{0:0=3d}".format(x)
        cv2.imwrite(os.path.join(savetodir,os.path.basename(filename).split(".nii")[0]+"_"+slice_number+".jpg" ),img_gray_data[:,:,x]*255 )

def call_saveslicesofnifti(args):
    try:
        filename=args.stuff[1]
        savetodir=args.stuff[2]
        saveslicesofnifti(filename,savetodir=savetodir)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'masks_on_grayscale_colored')
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
def savesingleslicesofnifti(filename,slicenumber=0,savetodir=""):
    filename_nib=nib.load(filename)
    filename_gray_data_np=filename_nib.get_fdata()
    min_img_gray=np.min(filename_gray_data_np)
    img_gray_data=0
    if not os.path.exists(savetodir):
        savetodir=os.path.dirname(filename)
    if min_img_gray>=0:
        # img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(1000, 1200))
        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(1020, 1100))
    else:
        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(20, 60))
        # img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(0, 200))
#    for x in range(img_gray_data.shape[2]):
    x=slicenumber
    filenamejpg=os.path.join(savetodir,os.path.basename(filename).split(".nii")[0]+str(x)+".jpg" )
    cv2.imwrite(filenamejpg,img_gray_data[:,:,x]*255 )
    return filenamejpg

def sas7bdatTOcsv(inputfilename,outputfilename=""):
    if len(outputfilename)==0:
        outputfilename=inputfilename.split(".sas7bdat")[0] + ".csv"
#    inputfilename="/home/atul/Downloads/dra.sas7bdat"
#    outputfilename="/home/atul/Downloads/dra.csv"
    inputdataframe=pd.read_sas(inputfilename, format = 'sas7bdat', encoding="latin-1")
    inputdataframe.to_csv(outputfilename, index=False)

def print_number_slices(inputdirectory):
    return "X"

def contrast_stretch(img,threshold_id):
    if threshold_id==1:
        # ct_image=exposure.rescale_intensity(img.get_fdata() , in_range=(0, 200))
        ct_image=exposure.rescale_intensity(img.get_fdata() , in_range=(20, 60))
    if threshold_id==2:
        # ct_image=exposure.rescale_intensity(img.get_fdata() , in_range=(1000, 1200))
        ct_image=exposure.rescale_intensity(img.get_fdata() , in_range=(1020, 1100))
    return ct_image
def contrast_stretch_stroke_range(img):
    # if threshold_id==1:
    ct_image=exposure.rescale_intensity(img.get_fdata() , in_range=(20, 60))
    # if threshold_id==2:
    #     ct_image=exposure.rescale_intensity(img.get_fdata() , in_range=(1020, 1100))
    return ct_image
def contrast_stretch_np(img,threshold_id):
    if threshold_id==1:
        # ct_image=exposure.rescale_intensity(img.get_fdata() , in_range=(0, 200))
        ct_image=exposure.rescale_intensity(img.get_fdata() , in_range=(20, 60))
    if threshold_id==2:
        # ct_image=exposure.rescale_intensity(img.get_fdata() , in_range=(1000, 1200))
        ct_image=exposure.rescale_intensity(img.get_fdata() , in_range=(1020, 1100))
    return ct_image
def saveslicesofnumpy3D(img_gray_data,savefilename="",savetodir=""):
##    filename_nib=nib.load(filename)
##    filename_gray_data_np=filename_nib.get_fdata()
#    min_img_gray=np.min(filename_gray_data_np)
#    img_gray_data=0
#    if not os.path.exists(savetodir):
#        savetodir=os.path.dirname(filename)
#    if min_img_gray>=0:
#        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(1000, 1200))
#    else:
#        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(0, 200))
    for x in range(img_gray_data.shape[2]):
        slice_num="{0:0=3d}".format(x)
        cv2.imwrite(os.path.join(savetodir,os.path.basename(savefilename).split(".nii")[0]+str(slice_num)+".jpg" ),img_gray_data[:,:,x] )


#def tex_for_each_subject():
    # find unique CT names:

    #create a tex file:
    # for each file of a CT:
    # find the files with that name arranged in ascending order of time:

    # write those images into the tex file

    # come out of for loop and close the latex file



def send_email():

    gmail_user = 'booktonotesbtn@gmail.com'
    gmail_password = 'AtulAtul1!@#$'

    sent_from = gmail_user
    to = ['sharmaatul11@gmail.com']
    subject = 'Program execution'
    body = "Your program has completed its task"

    email_text = """\
    From: %s
    To: %s
    Subject: %s
    
    %s
    """ % (sent_from, ", ".join(to), subject, body)

    try:
        server = smtplib.SMTP_SSL('mail.smtp2go.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, email_text)
        server.close()

        print('Email sent!')
    except:
        print('Something went wrong...')



def normalizeimage0to1(data):
    epi_img_data_max=np.max(data)
    epi_img_data_min=np.min(data)
    thisimage=data
    thisimage=(thisimage-epi_img_data_min)/(epi_img_data_max-epi_img_data_min)
    return thisimage


def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

def combine_csv_files():
    print("WE ARE COMBINING CSV TOTAL FILES")
    os.chdir(sys.argv[1]) ###"/fileinput_directory") #/storage1/fs1/dharr/Active/ATUL/PROJECTS/NWU/DATA/CTPDATA/Atul_20210408/output_directory/OUTPUT_20_80/CSF_RL_VOL_OUTPUT_infarct_manual/thres2080") #/storage1/fs1/dharr/Active/ATUL/PROJECTS/NWU/DATA/Missing_CTs_2021/OUTPUT/CSF_RL_VOL_OUTPUT/CSVS")
    extension="csv"
    all_filenames=[i for i in glob.glob('*.{}'.format(extension)) ]
    combined_csv=pd.concat([pd.read_csv(f) for f in all_filenames])
    combined_csv.to_csv("combined_csv_TOTAL.csv" , index=False, encoding='utf-8-sig')
    print("{0}  files combined".format(len(all_filenames)))


############################################################################################################################
def flipnifti3Dslicebysclie(numpy3D,flipdir=0):
    numpy3D_copy=np.copy(numpy3D)
    for x in range(numpy3D_copy.shape[2]):
        numpy3D_copy[:,:,x]=cv2.flip(numpy3D_copy[:,:,x],flipdir)
    return numpy3D_copy
def call_continous_to_binary_identical_ouputname(args):
    returnvalue=0
    try:
        filename=args.stuff[1]
        threshold=float(args.stuff[2])
        continous_to_binary_identical_ouputname(filename,threshold)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_gray2binary')
        subprocess.call(command,shell=True)
        returnvalue=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    # print(returnvalue)
    return  returnvalue
def continous_to_binary_identical_ouputname(filename_template,threshold=0):
    #     filename_template='/home/atul/Documents/DEEPREG/DeepReg/demos/classical_mr_prostate_nonrigid/dataset/brain/scct_strippedResampled1.nii.gz'
    I=nib.load(filename_template)
    I_data=I.get_fdata()
    min_val=0 #np.min(I_data)
    print(min_val)
    I_data[I_data>threshold]=1
    I_data[I_data<1]=0
    array_mask = nib.Nifti1Image(I_data, affine=I.affine, header=I.header)
    niigzfilenametosave2=filename_template #os.path.join(output_directory,os.path.basename(filename_template).split('.nii')[0]+ '_BET.nii.gz' )#os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
    nib.save(array_mask, niigzfilenametosave2)
    return niigzfilenametosave2
def gray2binary(filename_template,output_directory,threshold=0):
    #     filename_template='/home/atul/Documents/DEEPREG/DeepReg/demos/classical_mr_prostate_nonrigid/dataset/brain/scct_strippedResampled1.nii.gz'
    I=nib.load(filename_template)
    I_data=I.get_fdata()
    min_val=0 #np.min(I_data)
    print(min_val)
    I_data[I_data>threshold]=1
    I_data[I_data<1]=0
    array_mask = nib.Nifti1Image(I_data, affine=I.affine, header=I.header)
    niigzfilenametosave2=os.path.join(output_directory,os.path.basename(filename_template).split('.nii')[0]+ '_BET.nii.gz' )#os.path.join(OUTPUT_DIRECTORY,os.path.basename(levelset_file)) #.split(".nii")[0]+"RESIZED.nii.gz")
    nib.save(array_mask, niigzfilenametosave2)
    return niigzfilenametosave2
def call_gray2binary(args):
    returnvalue=0
    try:
        filename=args.stuff[1]
        output_directory=args.stuff[2]
        threshold=float(args.stuff[3])
        gray2binary(filename,output_directory,threshold)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_gray2binary')
        subprocess.call(command,shell=True)
        returnvalue=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    # print(returnvalue)
    return  returnvalue
def createh5file(image0_file,image1_file,label0_file,label1_file,output_dir="./"):
    returnvalue=0
    try:
        image0=nib.load(image0_file).get_fdata() #'/home/atul/Documents/DEEPREG/DeepReg/demos/classical_mr_prostate_nonrigid/dataset/fixed_images/case_001.nii.gz').get_fdata()

        ## import gray 1
        image1=nib.load(image1_file).get_fdata() #nib.load('/home/atul/Documents/DEEPREG/DeepReg/demos/classical_mr_prostate_nonrigid/dataset/fixed_images/case_009.nii.gz').get_fdata()
        # import label 0
        label0=nib.load(label0_file).get_fdata() # nib.load('/home/atul/Documents/DEEPREG/DeepReg/demos/classical_mr_prostate_nonrigid/dataset/fixed_labels/case_001.nii.gz').get_fdata()
        # import label 1
        label0[label0>0]=1
        label0[label0<1]=0
        image0=image0*label0
        label1=nib.load(label1_file).get_fdata() #nib.load('/home/atul/Documents/DEEPREG/DeepReg/demos/classical_mr_prostate_nonrigid/dataset/fixed_labels/case_009.nii.gz').get_fdata()
        label1[label1>0]=1
        label1[label1<1]=0
        image1=image1*label1
        h5filename=os.path.join(output_dir,os.path.basename(image1_file).split('.nii')[0] + '_h5data.h5')
        print("{}:{}".format('h5filename',h5filename))
        hf = h5py.File(h5filename, 'w')
        hf.create_dataset('image0',data=image0,dtype='i2') # moving image
        hf.create_dataset('image1',data=image1,dtype='i2') # fixed image
        hf.create_dataset('label0',data=label0,dtype='i') # moving mask
        hf.create_dataset('label1',data=label1,dtype='i') # fixed mask
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'createh5file')
        subprocess.call(command,shell=True)
        returnvalue=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    # print(returnvalue)
    return  returnvalue


def call_createh5file(args):
    returnvalue=0
    try:
        template_file=args.stuff[1]
        target_file=args.stuff[2]
        template_mask_file=args.stuff[3]
        target_mask_file=args.stuff[4]
        output_dir=args.stuff[5]
        createh5file(template_file,target_file,template_mask_file,target_mask_file,output_dir=output_dir)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_createh5file')
        subprocess.call(command,shell=True)
        returnvalue=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    # print(returnvalue)
    return  returnvalue
def main():
    command="echo i am main at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
    subprocess.call(command,shell=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('stuff', nargs='+')
    args = parser.parse_args()
    name_of_the_function=args.stuff[0]
    return_value=0
    if name_of_the_function == "call_create_a_latex_filename":
        return_value=call_create_a_latex_filename(args)
    if name_of_the_function == "call_latex_start":
        return_value=call_latex_start(args)
    if name_of_the_function == "call_latex_insertimage_tableNc":
        return_value=call_latex_insertimage_tableNc(args)
    if name_of_the_function == "call_latex_end":
        return_value=call_latex_end(args)
    if name_of_the_function == "call_write_panda_df":
        return_value=call_write_panda_df(args)
    if name_of_the_function == "call_saveslicesofnifti":
        return_value=call_saveslicesofnifti(args)
    if name_of_the_function == "call_remove_a_column":
        return_value=call_remove_a_column(args)
    if name_of_the_function == "call_latex_inserttext_tableNc":
        return_value=call_latex_inserttext_tableNc(args)
    if name_of_the_function == "call_space_between_lines":
        return_value=call_space_between_lines(args)
    if name_of_the_function == "call_latex_inserttext_tableNc_colored_with_bullet":
        return_value=call_latex_inserttext_tableNc_colored_with_bullet(args)
    if name_of_the_function == "call_gray2binary":
        return_value=call_gray2binary(args)
    if name_of_the_function == "call_createh5file":
        return_value=call_createh5file(args) #
    if name_of_the_function == "call_separate_mask_regions_into_individual_image":
        return_value=call_separate_mask_regions_into_individual_image(args) #
    if name_of_the_function == "call_normalization_N_resample_to_fixed":
        return_value=call_normalization_N_resample_to_fixed(args)
    if name_of_the_function == "call_only_resample_to_fixed":
        return_value=call_only_resample_to_fixed(args) #
    if name_of_the_function == "call_separate_masks_from_multivalue_mask":
        return_value=call_separate_masks_from_multivalue_mask(args)
    if "call" not in name_of_the_function:
        return_value=0
        globals()[args.stuff[0]](args)
        return return_value



    return return_value
if __name__ == '__main__':
    main()
import ast
import colorsys
# from github import Github
def saveslicesofnifti_withgiventhresh(filename,min_intensity,max_intensity,savetodir=""):
    # if min_img_gray>=0:
    img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(min_intensity, max_intensity))
    # else:
    #     img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(20, 60))
    #     # img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(0, 200))
def opencv_to_latex_rgb(bgr_tuple):
    # Convert BGR to RGB
    r, g, b = bgr_tuple[2], bgr_tuple[1], bgr_tuple[0]
    # Scale to 0-1 for LaTeX
    return (round(r / 255, 2), round(g / 255, 2), round(b / 255, 2))
def measure_mask_volume(mask_filename,gray_image_filename):
    mask_filename_data=nib.load(mask_filename).get_fdata()
    gray_image_filename_nib=nib.load(gray_image_filename)
    mask_filename_data[mask_filename_data>0.5]=1
    mask_filename_data[mask_filename_data<1]=0
    return (np.sum(mask_filename_data)*np.product(gray_image_filename_nib.header["pixdim"][1:4]))/1000
def color_box(rgb_string):
    r, g, b = ast.literal_eval(rgb_string)
    r,g,b=opencv_to_latex_rgb((r, g, b ))
    # r, g, b = round(r / 255, 2), round(g / 255, 2), round(b / 255, 2)
    return rf'\textcolor[rgb]{{{r},{g},{b}}}{{\rule{{20pt}}{{10pt}}}}'
def single_color_image(idx,rgb_string,out_dir,image_height=100,image_width=100):
    r, g, b = ast.literal_eval(rgb_string)
    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    # OpenCV uses BGR, so reverse RGB to BGR
    img[:] = (r, g, b)
    # Save the image
    filename = os.path.join(out_dir,f"color_{idx}.png")
    cv2.imwrite(filename, img)
    print(f"Image saved: {filename}")
    # r, g, b = round(r / 255, 2), round(g / 255, 2), round(b / 255, 2)
    return filename
def generate_contrasting_colors(n):
    colors = []
    used_hues = set()
    for i in range(n):
        base_hue = i / n
        hue = (base_hue + 0.5) % 1 if i % 2 == 0 else base_hue  # Alternate hues
        while hue in used_hues:  # Ensure no repeat hues
            hue = (hue + 0.2) % 1  # Skip to a new hue with a step of 0.2
        used_hues.add(hue)
        # Alternate saturation and brightness levels
        saturation = 0.9 if i % 3 == 0 else 0.6
        value = 0.9 if i % 4 == 0 else 0.7
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue % 1, saturation, value)
        # Scale RGB values to 0-255 range and convert to integer
        rgb = tuple(int(c * 255) for c in rgb)
        # Format the tuple as a string "(r,g,b)"
        colors.append(f"({rgb[0]},{rgb[1]},{rgb[2]})")
    return colors
# Helper function to escape LaTeX special characters
def escape_latex(value):
    if isinstance(value, str):
        replacements = {
            "_": r"\_",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            # "{": r"\{",
            # "}": r"\}",
            "~": r"\~",
            "^": r"\^",
            # "\\": r"\\",
        }
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value
# Convert the DataFrame into a LaTeX table
def df_to_latex(df):
    # Escape special characters in headers and cells
    headers = [escape_latex(str(col)) for col in df.columns]  # Convert to string
    rows = [[escape_latex(str(cell)) for cell in row] for row in df.values]
    # Begin LaTeX table
    latex = "\\begin{tabular}{" + " | ".join(["l"] * len(headers)) + "}\n\\hline\n"
    # Add headers
    latex += " & ".join(headers) + " \\\\\n\\hline\n"
    # Add rows
    for row in rows:
        latex += " & ".join(row) + " \\\\\n"
        latex += "\\hline\n"
    # End LaTeX table
    latex += "\\hline\n\\end{tabular}"
    return latex

def df_to_latex(df):
    # Ensure all headers are strings
    headers = [escape_latex(str(col)) for col in df.columns]  # Convert to string
    rows = [[escape_latex(str(cell)) for cell in row] for row in df.values]

    # Begin LaTeX table
    latex = "\\begin{tabular}{" + " | ".join(["l"] * len(headers)) + "}\n\\hline\n"

    # Add headers
    latex += " & ".join(headers) + " \\\\\n\\hline\n"

    # Add rows
    for row in rows:
        latex += " & ".join(row) + " \\\\\n"
        latex += "\\hline\n"

    # End LaTeX table
    latex += "\\hline\n\\end{tabular}"
    return latex

# def escape_latex(value):
#     """Escape special LaTeX characters in a string."""
#     special_chars = {
#         '&': r'\&',
#         '%': r'\%',
#         '$': r'\$',
#         '#': r'\#',
#         '_': r'\_',
#         '{': r'\{',
#         '}': r'\}',
#         '~': r'\textasciitilde{}',
#         '^': r'\textasciicircum{}',
#         '\\': r'\textbackslash{}'
#     }
#     for char, replacement in special_chars.items():
#         value = value.replace(char, replacement)
#     return value
