import os
import re
import numpy as np
import logging
from skimage import draw
from scipy.spatial import KDTree
import trimesh
import open3d as o3d
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger



def poly2mask(vertex_row_coords: np.ndarray, vertex_col_coords: np.ndarray,
              shape: np.ndarray) -> np.ndarray:
    """create mask from polygon.

    Parameters
    ----------
    vertex_row_coords (np.ndarray): vertex row coordinates
    vertex_col_coords (np.ndarray): vertex column coordinates
    shape (tuple): shape of the mask

    Returns
    -------
    np.ndarray: mask of the polygon
    """
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords,
                                                    vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool_)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def get_tray_mask(masks: np.ndarray, class_names: np.ndarray):
    
    tray_mask = None
    height, width = 720, 1280
    no_tray = False
    if 'tray_loc' in class_names:
        tray_mask = np.copy(masks[class_names == 'tray_loc'][0])  # TODO: what if there are both tray_loc and tray?
    else:
        tray_idx = np.where(np.isin(class_names, 'tray'))[0]
        tray_num = len(tray_idx)
        
        if len(tray_idx) == 0:
            tray_mask = np.ones((height, width))
            no_tray = True
        elif len(tray_idx) == 1:
            tray_mask = np.copy(masks[tray_idx[0]])
        elif len(tray_idx) > 1:
            edge_cropped_mask = np.zeros((height, width))
            edge_cropped_mask[int(height / 8):int(height * 7 / 8),
                                int(width / 8):int(width * 7 / 8)] = 1

            edge_cropped_tray_masks = edge_cropped_mask * masks[tray_idx]
            tray_pixel_sizes = np.array([np.sum(mask) for mask in edge_cropped_tray_masks])
            tray_mask = np.copy(masks[tray_idx[np.argmax(tray_pixel_sizes)]])

    return tray_mask

def json_to_result(labelme_json, image_shape):
    results = {'class_names': [], 'scores': [], 'masks': []}

    for shape in labelme_json['shapes']:
        results['class_names'].append(shape['label'])
        if 'scores' in shape.keys():
            results['scores'].append(shape['scores']['cls'])
        else:
            results['scores'].append(None)

        mask_points = np.array(shape['points']).astype(np.int64)
        if shape['shape_type'] == 'rectangle':
            mask = np.zeros(image_shape).astype('uint8')
            mask[mask_points[0, 1]:mask_points[1, 1],
                 mask_points[0, 0]:mask_points[1, 0]] = 1
        else:
            mask = poly2mask(mask_points[:, 1], mask_points[:, 0],
                                    np.array(image_shape)).astype('uint8')

        results['masks'].append(mask.astype(int))

    for k, v in results.items():
        results[k] = np.array(v)
    results['imageFileName'] = labelme_json['imagePath']
    results['tray_label'] = labelme_json.get('tray_label')
    return results


def statistical_outlier_removal(points, k=50, std_multiplier=3.0):
    # k-최근접 이웃 구하기
    
    kdtree = KDTree(points)
    distances, _ = kdtree.query(points, k + 1)  # 포인트 자신을 포함하기 때문에 k+1

    # 각 포인트에 대한 k-최근접 이웃들과의 평균 거리
    mean_distances = distances.mean(axis=1)

    # 평균 거리의 평균 및 표준편차
    mean = mean_distances.mean()
    std = mean_distances.std()

    # 이상치를 제거
    inliers = points[mean_distances <= mean + std_multiplier * std]

    return inliers


def remove_outlier_points(points: np.ndarray,
                          buffer_for_min: float = 3,
                          buffer_for_max: float = 1.5) -> np.ndarray:
    """remove outlier points using IQR.

    Parameters
    ----------
    points (np.ndarray): points
    buffer_for_min (float): buffer for min value
    buffer_for_max (float): buffer for max value

    Returns
    -------
    np.ndarray: outlier removed points
    """
    # IQR remove outlier
    points_removed_zero = points[points[..., 2] > 0]
    supply_Q1 = np.percentile(points_removed_zero[..., 2], 25, axis=None)
    supply_Q3 = np.percentile(points_removed_zero[..., 2], 75, axis=None)
    supply_IQR = supply_Q3 - supply_Q1

    min_value = supply_Q1 - buffer_for_min * supply_IQR
    max_value = supply_Q3 + buffer_for_max * supply_IQR

    points_removed = np.ones_like(points) * np.zeros((1, 3))
    masked_idx = (points[..., 2] > 0) & (points[..., 2] > min_value) & (
        points[..., 2] < max_value)
    points_removed[masked_idx] = points[masked_idx]

    return points_removed




def calculate_volume(mesh, food_points, ray_origin, ray_direction_scale=1.0):
    volume = 0.0

    for point in food_points:
        # 각 food point에서 ray direction 생성 (point를 향하는 방향)
        ray_direction = (point - ray_origin) * ray_direction_scale
        ray_direction = ray_direction / np.linalg.norm(ray_direction)  # Normalize the direction

        # ray_arrays = np.append(food_points, proj_dir_array, axis=1)
        # rays = o3d.core.Tensor(ray_arrays, dtype=o3d.core.Dtype.Float32)
        # cast_result = scene.cast_rays(rays)
        # height_array = cast_result['t_hit']

        # ray casting: mesh와의 교차점 찾기
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction]
        )

        # 교차점이 없으면 무시
        if len(locations) < 2:
            continue

        # 첫 번째 교차점과 두 번째 교차점 사이의 거리가 높이
        entry_point = locations[0]
        exit_point = locations[1]
        height = np.linalg.norm(exit_point - entry_point)

        # 작은 부피 요소로 간주하고 합산
        volume += height

    return volume

def open3d_to_trimesh(open3d_mesh):
    # Open3D mesh에서 vertices와 triangles 추출
    vertices = np.asarray(open3d_mesh.vertices)
    faces = np.asarray(open3d_mesh.triangles)
    
    # Trimesh 객체 생성
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return trimesh_mesh


def cast_rays(food_points: np.ndarray,
                scene: o3d.t.geometry.RaycastingScene) -> np.ndarray:
    """cast rays to the scene to get height array.

    Parameters
    ----------
    food_points (np.ndarray): real coordinates of food points
    scene (o3d.t.geometry.RaycastingScene): raycasting scene

    Returns
    -------
    np.ndarray: height array
    """
    proj_dir_array = np.zeros(food_points.shape)
    proj_dir_array[:, 2] = 1  # rays are casted parallel to z direction

    ray_arrays = np.append(food_points, proj_dir_array, axis=1)
    rays = o3d.core.Tensor(ray_arrays, dtype=o3d.core.Dtype.Float32)
    cast_result = scene.cast_rays(rays)
    height_array = cast_result['t_hit']

    return height_array

def get_volume_each_food(bottom_mesh,
                            food_idx: int, food_points: np.ndarray, fx: float,
                            fy: float):
    """get volume of food class with the given food points.

    Parameters
    ----------
    scene (o3d.t.geometry.RaycastingScene): raycasting scene
    food_idx (int): index of food class
    food_points (np.ndarray): real coordinates of food points
    fx (float): focal length of x axis
    fy (float): focal length of y axis
    """


    scene = o3d.t.geometry.RaycastingScene()
    bottom_mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(
        bottom_mesh)

    bottom_mesh_id = scene.add_triangles(bottom_mesh_)


    if food_points is None:
        return

    height_array = cast_rays(food_points, scene)

    is_hit = height_array.isfinite().numpy()
    height_array_hit = height_array.numpy()[is_hit]

    hit_pixel_areas, not_hit_pixel_areas = get_pixel_areas(
        food_points, is_hit, fx, fy)

    if np.sum(is_hit) != 0:
        average_height = np.average(height_array_hit)
        volume = np.sum(np.multiply(hit_pixel_areas,
                                    height_array_hit)) / 1000
    else:
        average_height = None
        volume = None

    # volume_info[food_idx].hit_points = food_points[is_hit]
    # volume_info[food_idx].not_hit_points = food_points[~is_hit]
    print('volume: ', volume)
    print('hit pixel area', hit_pixel_areas.sum())
    print('average height', average_height) 
    print('not_hit_point_ratio:', np.sum(~is_hit) / len(food_points))

    return volume


def get_pixel_areas(food_points: np.ndarray, is_hit: np.ndarray,
                    fx: float, fy: float) -> (np.ndarray, np.ndarray):
    """get pixel areas of hit and not hit food points.

    Parameters
    ----------
    food_points (np.ndarray): real coordinates of food points
    is_hit (np.ndarray): boolean array of hit food points
    fx (float): focal length of x axis
    fy (float): focal length of y axis

    Returns
    -------
    np.ndarray: pixel areas of hit food points
    np.ndarray: pixel areas of not hit food points
    """
    hit_food_points_z = food_points[is_hit][:, 2]
    not_hit_points_z = food_points[~is_hit][:, 2]

    hit_pixel_areas = np.multiply(hit_food_points_z,
                                    hit_food_points_z) / (fx * fy)
    not_hit_pixel_areas = np.multiply(not_hit_points_z,
                                        not_hit_points_z) / (fx * fy)

    return hit_pixel_areas, not_hit_pixel_areas

def transform_food_points(x, y, z, top, height, left, width,food_masks, translation, ratio, center, transformation):

    pcd_dict = {}
    for i, food_mask in zip(food_masks.keys(), food_masks.values()):

        food_mask = food_mask[top:top+height, left:left+width]
        x_food = x[food_mask]
        y_food = y[food_mask]
        z_food = z[food_mask]
        points_food = np.stack((x_food, y_food, z_food), axis=-1).reshape(-1, 3)

        pcd_food = o3d.geometry.PointCloud()
        pcd_food.points = o3d.utility.Vector3dVector(points_food)


        pcd_food.translate(translation)
        pcd_food.scale(ratio, center=center)

        pcd_food.transform(transformation)

        pcd_dict[i] = pcd_food

    return pcd_dict

def crop_by_tray_area(image, tray_mask):

    rows, cols = np.where(tray_mask == 1)
    top, bottom = rows.min(), rows.max()
    left, right = cols.min(), cols.max()

    # 이미지의 가로 세로 비율 계산
    image_height, image_width = image.shape[:2]
    image_aspect_ratio = image_width / image_height

    # 바운딩 박스의 가로 세로 크기와 비율 계산
    bbox_height = bottom - top + 1
    bbox_width = right - left + 1
    bbox_aspect_ratio = bbox_width / bbox_height

    # 바운딩 박스의 가로 세로 비율을 이미지 비율에 맞추기 위한 조건
    # if bbox_aspect_ratio > image_aspect_ratio:
    #     # 바운딩 박스가 더 넓은 경우, 세로를 확장
    #     new_height = int(bbox_width / image_aspect_ratio)
    #     padding = (new_height - bbox_height) // 2
    #     top = max(0, top - padding)
    #     bottom = min(image_height, bottom + padding)
    # else:
    #     # 바운딩 박스가 더 높은 경우, 가로를 확장
    #     new_width = int(bbox_height * image_aspect_ratio)
    #     padding = (new_width - bbox_width) // 2
    #     left = max(0, left - padding)
    #     right = min(image_width, right + padding)

    # 수정된 바운딩 박스를 사용하여 이미지 crop
    cropped_image = image[top:bottom+1, left:right+1]
    return cropped_image, [top, bottom, left, right]


def get_tray_top_mask(depth, tray_mask):

    ## tray top 획득
    ## 1. kmeans clustering
    ## 2. 가장 큰 3개의 cluster들의 mask 를 만든다. 
    ## 3. mask 들의 convexhull를 계산하여 영역의 넓이가 가장 큰 mask 를 획득한다. 
    ## 4. tray mask 와 AND 연산을 하여 최종 mask 를 얻는다.

    depth_uint8 = (depth - depth.min()) / (depth.max() - depth.min())  * 255.0
    depth_uint8 = depth_uint8.astype(np.uint8)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth_uint8 = (cmap(depth_uint8)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite("vis_depth/depth_uint8.png", depth_uint8)  

    # 이미지 로드 및 전처리
    masks_ordered = get_cluster_masks_ordered(depth_uint8)

    ## convex hull이 가장 큰 클러스터를 tray top으로. 
    tray_top_masks = select_tray_tops_with_convexhull(masks_ordered)

    # tray_top_mask = select_tray_top_with_center_area(tray_top_masks)

    tray_top_mask = tray_top_masks[0] & (tray_mask == 1)

    cv2.imwrite("vis_depth/tray_top_mask_b.png", tray_top_mask.astype(np.uint8)*255)
    tray_top_mask = cv2.erode(tray_top_mask.astype(np.uint8), np.ones((10, 10), np.uint8), iterations=1)
    # 원본 이미지에서 해당 클러스터 영역만 추출
    # largest_cluster_image = np.zeros_like(image_rgb)
    # largest_cluster_image[mask] = image_rgb[mask]

    cv2.imwrite("vis_depth/tray_top_mask.png", tray_top_mask.astype(np.uint8)*255)
    return tray_top_mask

def select_tray_top_with_center_area(tray_top_masks):

    max_area = 0
    max_area_index = 0
    for i, mask in enumerate(tray_top_masks):
        center_area = get_center_area(mask)
        if center_area > max_area:
            max_area = center_area
            max_area_index = i

    return tray_top_masks[max_area_index]

def get_center_area(mask):

    h, w = mask.shape[:2]

    center_l = w // 3
    center_r = 2 * w // 3
    center_t = h // 3
    center_b = 2 * h // 3

    center = mask[center_t:center_b, center_l:center_r]
    center_area = np.count_nonzero(center)

    return center_area

def get_cluster_masks_ordered(image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape((-1, 3))

    # K-means clustering 수행
    k = 7  # 클러스터 수 설정
    kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
    labels = kmeans.labels_

    # 각 클러스터의 픽셀 수 계산
    label_counts = Counter(labels)
    label_ordered = label_counts.most_common(3)
    # largest_cluster_label = label_counts.most_common(1)[0][0]

    masks = []
    for i in range(len(label_ordered)):
        label = label_ordered[i][0]
        mask = (labels == label).reshape(image.shape[:2])
        masks.append(mask)
        cv2.imwrite(f"vis_depth/mask_{i}.png", mask.astype(np.uint8) * 255)
    return masks


def select_tray_tops_with_convexhull(masks, n_top=2):

    areas = []
    for i, mask in enumerate(masks):
        area, _= calc_convexhull_area(mask, i)
        areas.append((i, area))

    areas = sorted(areas, key=lambda x: x[1], reverse=True)

    tray_top_masks = []
    for i in range(min(n_top, len(masks))):
        tray_top_masks.append(masks[areas[i][0]])
    tray_top_masks = np.array(tray_top_masks)

    return tray_top_masks
    

def calc_convexhull_area(mask, i):

    input_mask = mask.astype(np.uint8) * 255
    # 이진화 처리 (이미 흑백 이미지이므로 0, 255로 변환)
    _, binary_mask = cv2.threshold(input_mask, 127, 255, cv2.THRESH_BINARY)

    # 모든 흰색 영역에 대해 컨투어 찾기
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convex Hull을 계산하여 새로운 마스크에 그리기
    convex_mask = np.zeros_like(binary_mask)
    for contour in contours:
        hull = cv2.convexHull(contour)
        cv2.drawContours(convex_mask, [hull], -1, 255, thickness=cv2.FILLED)

    # Convex Hull 영역 넓이 계산 (각 Hull의 넓이를 더함)
    total_area = sum(cv2.contourArea(cv2.convexHull(contour)) for contour in contours)
    
    output_path = f"vis_depth/convex_hull_mask_{i}.png"
    cv2.imwrite(output_path, convex_mask)
    return total_area, mask

def calc_plane_params(depth, tray_top_mask, scale_factor):

    # 예제 데이터 생성
    H, W = depth.shape  # pred는 H x W 크기의 numpy 배열이라고 가정
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))  # x, y 좌표 생성

    # tray_top_mask가 1인 위치에서 x, y, z 좌표 추출
    mask_indices = tray_top_mask == 1
    x = x_coords[mask_indices]
    y = y_coords[mask_indices]
    z = depth[mask_indices] * scale_factor

    # 평면 피팅 (z = ax + by + c 형태)
    XY = np.column_stack((x, y))  # x와 y 좌표를 열로 묶어 피팅 데이터 생성
    model = LinearRegression().fit(XY, z)
    a, b = model.coef_
    c = model.intercept_

    # print(f"Plane equation: z = {a}*x + {b}*y + {c}")

    # 평면 방정식을 기반으로 점 생성 (예: 시각화용 샘플링)
    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = np.linspace(y.min(), y.max(), 100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = a * x_grid + b * y_grid + c  # 피팅된 평면의 z 값 계산

    # Open3D로 시각화
    # 평면 포인트 생성
    points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(points)
    plane_pcd.paint_uniform_color([0, 1, 0])

    # 원본 포인트 클라우드도 추가 (optional)
    plane_area_points = np.column_stack((x, y, z))
    plane_area_pcd = o3d.geometry.PointCloud()
    plane_area_pcd.points = o3d.utility.Vector3dVector(plane_area_points)
    plane_area_pcd.paint_uniform_color([0, 0, 1])

    not_plane_indices = tray_top_mask == 0
    not_plane_points = np.column_stack((x_coords[not_plane_indices], y_coords[not_plane_indices], depth[not_plane_indices] * scale_factor))
    not_plane_points = not_plane_points[(not_plane_points[:, 2] > 0) & (not_plane_points[:, 2] <= 1000)]
    not_plane_pcd = o3d.geometry.PointCloud()
    not_plane_pcd.points = o3d.utility.Vector3dVector(not_plane_points)
    not_plane_pcd.paint_uniform_color([1, 0, 0])

    return [a, b, c], plane_pcd, plane_area_pcd, not_plane_pcd

def get_height_from_top_per_food(depth, food_mask, scale_factor, plane_params):

    food_mask = cv2.erode(food_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)

    H, W = depth.shape  # pred는 H x W 크기의 numpy 배열이라고 가정
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))  # x, y 좌표 생성

    # tray_top_mask가 1인 위치에서 x, y, z 좌표 추출
    mask_indices = food_mask == 1
    X = x_coords[mask_indices]
    Y = y_coords[mask_indices]
    Z = depth[mask_indices] * scale_factor

    distances = calc_distances_to_plane(X, Y, Z, plane_params)

    food_points = np.column_stack((X, Y, Z))
    food_pcd = o3d.geometry.PointCloud()
    food_pcd.points = o3d.utility.Vector3dVector(food_points)
    food_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    return np.mean(distances), np.std(distances), food_pcd

def calc_distances_to_plane(X, Y, Z, plane_params):
    """
    다수의 3D 점과 피팅된 평면 사이의 거리를 계산합니다.
    
    Parameters:
        X (numpy.ndarray): x 좌표 배열
        Y (numpy.ndarray): y 좌표 배열
        Z (numpy.ndarray): z 좌표 배열
        plane_params (tuple or list): 피팅된 평면의 계수 (a, b, c)
        
    Returns:
        numpy.ndarray: 각 점과 평면 사이의 거리 배열
    """
    a, b, c = plane_params
    d = c  # 평면 방정식을 ax + by - z + d = 0 형태로 변환

    # 평면과 각 (x, y, z) 점 사이의 거리 계산
    distances = a * X + b * Y - Z + d / np.sqrt(a**2 + b**2 + 1)
    return distances

def get_food_masks(results, shape, crop_loc):

    top, bottom, left, right = crop_loc
    food_masks = {}
    food_indices = []
    for i in range(len(results['class_names'])):
        if results['class_names'][i] in ["hand", "spoon", "chopsticks", "cutlery"] :
            pass
        elif results['class_names'][i] == 'tray':
            tray_mask = results['masks'][i]
        else :
            food_indices.append(i)
            zero = np.zeros(shape)
            food_masks[i] =  np.logical_or(zero, results['masks'][i])[top:bottom+1, left:right+1]
    return food_masks, food_indices

def calc_height_of_bottom_from_top(depth, plane_params, tray_mask, food_masks, scale_factor):

    tray_mask = cv2.erode(tray_mask.astype(np.uint8), np.ones((10, 10), np.uint8), iterations=1)
    no_food_mask = np.logical_and(tray_mask, np.logical_not(np.logical_or.reduce(list(food_masks.values())))) 
    
    x_coords, y_coords = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))  

    # tray_top_mask가 1인 위치에서 x, y, z 좌표 추출
    mask_indices = no_food_mask == 1
    X = x_coords[mask_indices]
    Y = y_coords[mask_indices]
    Z = depth[mask_indices] * scale_factor

    inlier_points = statistical_outlier_removal(np.column_stack((X, Y, Z)))

    distances = calc_distances_to_plane(inlier_points[:, 0], inlier_points[:, 1], inlier_points[:, 2], plane_params)

    ## 평면에서 식판쪽 거리가 음값이기에 argmin으로 연산.
    max_distance_index = np.argmin(distances)
    max_x = inlier_points[max_distance_index, 0]
    max_y = inlier_points[max_distance_index, 1]
    # max_distance_index = np.where((inlier_points[:, 0]== max_x) & (inlier_points[:, 1] == max_y))[0][0]
    
    
    
    depth_uint8 = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    # depth_uint8 = (depth*255 ).astype(np.uint8)
    depth_colored = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
    cv2.circle(depth_colored, (int(max_x), int(max_y)), 5, (0, 0, 255), -1)
    cv2.imwrite('vis_depth/depth_with_dist_max_point.png', depth_colored)
    return distances[max_distance_index]

def is_exist_foods_at_all_compart():

    pass



