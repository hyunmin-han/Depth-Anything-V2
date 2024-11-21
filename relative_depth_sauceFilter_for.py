
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
from metric_depth.util.utils import *

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
    parser.add_argument('--outdir', type=str, default='./vis_depth',
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

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Create the output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    bucket = args.s3_bucket
    if args.is_img_s3_uri :
        df = pd.read_csv('/home/seungu/Downloads/sauce filtering test - 시트1.csv')
        filenames = df['s3_key']

    else :
        pass

    cont_point = 13
    scale_factor = 100000

    food_results_list = []
    bottom_heights = []

    # filenames = [
    #     "sawoo-es/240927/L/A/sawoo-es_240927_034618_16844_L_A_VS-2407080010_Trayfile-inferenced.json"

    #     # "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_122815_16936_L_A_10064008642000011_Trayfile.png",
    #     # "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_133826_17404_L_A_10064008642000025_Trayfile.png",
    #     # "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_133054_17411_L_A_10064008642000025_Trayfile.png",
    #     # "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_133718_17424_L_A_10064008642000011_Trayfile.png",
    #     # "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_131944_17361_L_A_10164000044A0002A_Trayfile.png",
    #     # "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_122637_17104_L_A_1006400864200000B_Trayfile.png",
    #     # "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_114145_16860_L_A_10064008642000011_Trayfile.png",
    #     # "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_114204_16860_L_A_10164000044A0002A_Trayfile.png",
    #     # "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_133814_17404_L_A_10064008642000025_Trayfile.png",
    #     # "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_114213_16860_L_A_10164000044A0002A_Trayfile.png",
    #     # "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_123430_16899_L_A_10064008642000011_Trayfile.png",
    #     # "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_131956_17346_L_A_10164000044A0002A_Trayfile.png",
    #     # "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_132021_17499_L_A_10064008642000011_Trayfile.png",
    #     # "s3://nuvi-data/sawoo-es/241115/L/A/sawoo-es_241115_131936_17361_L_A_10164000044A0002A_Trayfile.png",
    # ]
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

        ## tray top 점들로 linear regression하여 평면의 방정식을 획득한다.
        tray_top_mask = get_tray_top_mask(pred, tray_mask[top:bottom+1, left:right+1], args.save_name)
        # np.save(f'tray_top_masks/{k}.npy', tray_top_mask)
        cv2.imwrite(f"tray_top_masks/{k}.jpg", tray_top_mask.astype(np.uint8)*255)

        continue
        pred = np.where(pred == 0, 0, 1 / pred)
        plane_params, plane_pcd, plane_area_pcd, not_plane_pcd = calc_plane_params(pred, tray_top_mask, scale_factor)

        food_masks, food_indices = get_food_masks(results, image.shape[:2], crop_loc)

        # if is_exist_foods_at_all_compart(): 
        #     pass
        # else :
        bottom_height = calc_height_of_bottom_from_top(pred, plane_params, tray_mask[top:bottom+1, left:right+1], food_masks, scale_factor)
        bottom_heights.append(bottom_height)
        print(f"bottom_height: {bottom_height}")

        food_pcds = []
        food_results=""
        for idx in food_indices:
            food_mask = food_masks[idx]

            ## 평면과 각 음식들의 평균 거리와 std 값을 구한다.
            height, std_h, food_pcd = get_height_from_top_per_food(pred, food_mask, scale_factor, plane_params)
            food_pcds.append(food_pcd)
            print('food height from top:', height)

            height_from_bottom = -(bottom_height - height)

            ## height 값이 음값이기에, -1을 곱해준다.
            result_str = f"{results['class_names'][idx]} {height:.3f} {height_from_bottom:.3f}\n" 
            print(result_str)
            if 'food_results' not in locals():
                food_results = result_str
            else:
                food_results += result_str
        food_results_list.append(food_results)

        print('total time:', time.time()-s)
    # Save the results to a DataFrame and write to a CSV file
    df_results = pd.DataFrame({
        'food_heights': food_results_list,
        'bottom_heights': bottom_heights
    }) 
    df_results.to_csv(os.path.join(args.outdir, f'food_heights_{args.save_name}.csv'), index=False) 





if __name__ == '__main__':
    main()