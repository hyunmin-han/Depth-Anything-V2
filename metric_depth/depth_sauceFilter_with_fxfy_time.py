
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

    cont_point = 0
    scale_factor = 100

    food_results_list = []
    bottom_heights = []
    is_zeros_list = []

    zero_uris, zero_height_percentages = [], []
    # filenames = [
    #     "s3://nuvi-data/sawoo-es/241119/L/A/sawoo-es_241119_122957_17155_L_A_10164000044A0002A_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241113/L/A/sawoo-es_241113_122720_16978_L_A_cb00cbbe143e6d68_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241113/L/A/sawoo-es_241113_033701_16877_L_A_VS-2407080010_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241113/L/A/sawoo-es_241113_041930_17329_L_A_VS-23101002_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241119/L/A/sawoo-es_241119_131759_17616_L_A_10164000044A0002A_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241120/L/A/sawoo-es_241120_131212_17228_L_A_10064008642000011_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_133054_17411_L_A_10064008642000025_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_123430_16899_L_A_10064008642000011_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241120/L/A/sawoo-es_241120_123544_16871_L_A_10064008642000011_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241120/L/A/sawoo-es_241120_122805_17063_L_A_1006400864200000B_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241113/L/A/sawoo-es_241113_034008_16938_L_A_VS-2407080010_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241120/L/A/sawoo-es_241120_122725_16956_L_A_10064008642000011_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241113/L/A/sawoo-es_241113_033329_17052_L_A_VS-23101004_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241120/L/A/sawoo-es_241120_132934_17326_L_A_10064008642000025_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241114/L/A/sawoo-es_241114_030634_17054_L_A_VS-23101004_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241113/L/A/sawoo-es_241113_034224_16936_L_A_VS-2407080010_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241119/L/A/sawoo-es_241119_124350_16842_L_A_10064008642000011_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241120/L/A/sawoo-es_241120_131112_17660_L_A_1006400864200000B_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241120/L/A/sawoo-es_241120_122951_17154_L_A_10064008642000025_Trayfile.png",
    #     "s3://nuvi-data/sawoo-es/241113/L/A/sawoo-es_241113_034211_16941_L_A_VS-2407080010_Trayfile.png",
    # ]

    fx = args.focal_length_x / 2
    fy = args.focal_length_y / 2

    for k, filename in enumerate(filenames[:100]):
        # if k != cont_point:
        #     continue
        print('k :', k)


        if args.is_img_s3_uri :
            filename = filename.replace('s3://nuvi-data/', '')
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

        tray_mask = get_tray_mask(results['masks'], results['class_names'])

        if args.crop:
            # Get the bounding box of the tray mask
            cropped_image, crop_loc = crop_by_tray_area(image, tray_mask)
            top, bottom, left, right = crop_loc
            height, width = cropped_image.shape[:2]

            print('Cropped image h, w:', height, width)

            s = time.time()
            pred = depth_anything.infer_image(cropped_image, height)
            print('crop infer time:', time.time()-s)

            tray_mask = tray_mask[top:bottom+1, left:right+1]

        else :
            height, width = image.shape[:2]
            crop_loc = None
            s = time.time()
            pred = depth_anything.infer_image(image, height)
            print('infer time:', time.time()-s)

        t = time.time()
        food_masks, food_indices, food_masks_original = get_food_masks(results, image.shape[:2], crop_loc)
        tray_top_mask, initial_point = get_tray_top_mask2(pred, tray_mask, list(food_masks.values()))

        count_nonzero = np.count_nonzero(tray_top_mask)

        if count_nonzero < 100 : 

            # ## tray top 점들로 linear regression하여 평면의 방정식을 획득한다.
            tray_top_mask, depth_uint8 = get_tray_top_mask(pred, 
                                                        tray_mask, 
                                                        list(food_masks.values()),
                                                        depth_saturate_threshold=1.3)
            

        plane_params = calc_plane_params(pred, tray_top_mask, scale_factor,
                                                                                 fx=fx, fy=fy, is_vis=False)


        bottom_height, depth_colored = calc_height_of_bottom_from_top(pred, plane_params, tray_mask,
                                                        food_masks, scale_factor, fx=fx, fy=fy)


        for idx in food_indices:
            food_mask = food_masks[idx]
            food_mask_original = food_masks_original[idx]

            ## 평면과 각 음식들의 평균 거리와 std 값을 구한다.
            height, std_h, food_pcd = get_height_from_top_per_food(pred, food_mask, scale_factor, plane_params,
                                                                fx=fx, fy=fy)

            height_from_bottom = -(bottom_height - height)

            # Convert image to 4 channels (RGBA)

            # Calculate height percentage
            height_percentage = -height_from_bottom / bottom_height

        print('algorithm time:', time.time()-t)
        
    # Save the results to a DataFrame and write to a CSV file
    df_results = pd.DataFrame({
        'food_heights': food_results_list,
        'bottom_heights': bottom_heights,
    }) 
    df_results.to_csv(os.path.join('_output', f'{args.save_name}/food_heights.csv'), index=False) 

    df_results2 = pd.DataFrame({
        'file_name': zero_uris,
        'food_heights_percentage': zero_height_percentages
    }) 
    df_results2.to_csv(os.path.join('_output', f'{args.save_name}/zero.csv'), index=False) 






if __name__ == '__main__':
    main()