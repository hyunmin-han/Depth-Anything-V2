
import argparse
import cv2
import numpy as np
import os
import torch
import open3d as o3d
import time
import copy
import matplotlib.pyplot as plt
import pandas as pd

from depth_anything_v2.dpt import DepthAnythingV2
from nuviAPI.s3 import s3_api
from util.utils import *

def draw_registration_result(source, target, transformation):
    
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate depth maps and point clouds from images.')
    parser.add_argument('--encoder', default='vitl', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder to use.')
    parser.add_argument('--load-from', default='', type=str, required=True,
                        help='Path to the pre-trained model weights.')
    parser.add_argument('--max-depth', default=20, type=float,
                        help='Maximum depth value for the depth map.')
    parser.add_argument('--img-path', type=str, required=False,
                        help='Path to the input image or directory containing images.')
    parser.add_argument('--outdir', type=str, default='./vis_pointcloud',
                        help='Directory to save the output point clouds.')
    parser.add_argument('--focal-length-x', default=470.4, type=float,
                        help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=470.4, type=float,
                        help='Focal length along the y-axis.')
    parser.add_argument('--is-img-s3-uri', type=bool, default=True,
                        help='boolean of S3 containing images.')
    parser.add_argument('--s3-bucket', type=str, required=False, default='nuvi-depth',
                        help='S3 bucket name.')
    parser.add_argument('--crop',  action='store_true',
                        help='Whether to crop the image or not.')
    parser.add_argument('--save-name', default='', type=str, 
                        help='Name of the saved image.')

    args = parser.parse_args()


    # Determine the device to use (CUDA, MPS, or CPU)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Model configuration based on the chosen encoder
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize the DepthAnythingV2 model with the specified configuration
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Create the output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    
    bucket = args.s3_bucket
    if args.is_img_s3_uri :
        df = pd.read_csv('/home/seungu/Downloads/sauce filtering test - 시트1.csv')
        filenames = df['s3_key']
    else :
        pass

    food_results_list = []
    for k, filename in enumerate(filenames):


        if args.is_img_s3_uri :
            filename = filename.replace('-inferenced.json', '.png')
            image = s3_api.get_image(bucket, filename)
        else :
            filename = args.img_path
            image = cv2.imread(filename)


        # Read the image using OpenCV
        json_key = filename.replace('.png', '-inferenced.json')
        labelme_json = s3_api.get_json(bucket, json_key)

        depth_key = '/'.join(labelme_json['imageUri'].split('/')[3:]).replace('.png', '.npy')
        depth_array = s3_api.get_npy(bucket, depth_key)
        results = json_to_result(labelme_json, depth_array.shape)


        if args.crop:
            tray_mask = get_tray_mask(results['masks'], results['class_names'])
            # Get the bounding box of the tray mask
            cropped_image, crop_loc = crop_by_tray_area(image, tray_mask)
            top, bottom, left, right = crop_loc
            height, width = cropped_image.shape[:2]

            print('Cropped image h, w:', height, width)

            s = time.time()
            pred = depth_anything.infer_image(cropped_image, height)
            print('crop infer time:', time.time()-s)
        else :
            height, width = image.shape[:2]
            s = time.time()
            pred = depth_anything.infer_image(image, height)
            print('infer time:', time.time()-s)

        
        ## tray top 점들로 linear regression 하여 평면의 방정식을 획득한다.
        tray_top_mask = get_tray_top_mask(pred, tray_mask[top:bottom+1, left:right+1])

        scale_factor = 100
        plane_params, plane_pcd, plane_area_pcd, not_plane_pcd = calc_plane_params(pred, tray_top_mask, scale_factor)

        food_masks, food_indices = get_food_masks(results, image.shape[:2], crop_loc)

        # if is_exist_foods_at_all_compart(): 
        #     pass
        # else :
        bottom_height = calc_height_of_bottom_from_top(pred, plane_params, tray_mask[top:bottom+1, left:right+1], food_masks, scale_factor)
        
        food_pcds = []
        food_results=""
        for idx in food_indices:
            food_mask = food_masks[idx]

            ## 평면과 각 음식들의 평균 거리와 std 값을 구한다.
            height, std_h, food_pcd = get_height_per_food(pred, food_mask, scale_factor, plane_params)
            food_pcds.append(food_pcd)

            food_h = np.abs(bottom_height) - np.abs(height)
            result_str = f"{results['class_names'][idx]}, {food_h}, {std_h}\n"
            print(result_str)
            if 'food_results' not in locals():
                food_results = result_str
            else:
                food_results += result_str
        food_results_list.append(food_results)

    # Save the results to a DataFrame and write to a CSV file
    df_results = pd.DataFrame(food_results_list, columns=['food_heights'])
    df_results.to_csv(os.path.join(args.outdir, 'food_heights.csv'), index=False) 





if __name__ == '__main__':
    main()