# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import open3d as o3d
import zlib
import math

# add project directory to python path to enable relative imports
import os
import sys

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools


# visualize lidar point-cloud
def crop_pcl(lidar_pcl, configs, vis=True):

    # remove points outside of detection cube defined in 'configs.lim_*'
    mask = np.where(
        (lidar_pcl[:, 0] >= configs.lim_x[0])
        & (lidar_pcl[:, 0] <= configs.lim_x[1])
        & (lidar_pcl[:, 1] >= configs.lim_y[0])
        & (lidar_pcl[:, 1] <= configs.lim_y[1])
        & (lidar_pcl[:, 2] >= configs.lim_z[0])
        & (lidar_pcl[:, 2] <= configs.lim_z[1])
    )
    lidar_pcl = lidar_pcl[mask]

    # visualize point-cloud
    if vis:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_pcl[:, :3])
        o3d.visualization.draw_geometries([pcd])

    return lidar_pcl

def show_pcl(pcl):

    ####### ID_S1_EX2 START #######
    #######
    print("student task ID_S1_EX2")

    # step 1 : initialize open3d with key callback and create window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud", width=608, height=608, visible=True)

    # step 2 : create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()

    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])

    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    vis.add_geometry(pcd)

    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    should_close = {"flag": False}

    def _close_on_right_arrow(vis_):
        should_close["flag"] = True
        return False  # returning False keeps callback registration behavior simple

    while not should_close["flag"]:
        if not vis.poll_events():  # window closed manually
            break
        vis.update_renderer()

    # while vis.poll_events():
    #     vis.update_renderer()

    vis.destroy_window()

    #######
    ####### ID_S1_EX2 END #######


# visualize range image
def load_range_image(frame, lidar_name):

    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][
        0
    ]  # get laser data structure from frame
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0:  # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    return ri

def show_range_image(frame, lidar_name):
    """
    Build an 8-bit visualization image by stacking range and intensity channels
    from the Waymo range image.

    Returns:
        img_range_intensity (np.ndarray): uint8 image of shape (2H, W)
    """

    ####### ID_S1_EX1 START #######
    #######
    print("student task ID_S1_EX1")

    # 1) Load range image: expected shape (H, W, C)
    ri = load_range_image(frame, lidar_name)
    if ri is None:
        raise ValueError(f"load_range_image returned None for lidar={lidar_name}")
    if ri.ndim != 3 or ri.shape[2] < 2:
        raise ValueError(f"Expected range image (H,W,C>=2), got shape {ri.shape}")

    # 2) Clamp invalid values
    ri = ri.copy()
    ri[ri < 0] = 0.0

    ri_range = ri[:, :, 0]
    ri_intensity = ri[:, :, 1]

    # 3) Convert range to 8-bit (use full nonzero range)
    # Avoid dividing by zero if there are no valid points
    valid_range = ri_range > 0
    if np.any(valid_range):
        r_min = np.min(ri_range[valid_range])
        r_max = np.max(ri_range[valid_range])
        denom = max(r_max - r_min, 1e-6)
        img_range = (ri_range - r_min) / denom
        img_range = np.clip(img_range, 0.0, 1.0)
    else:
        img_range = np.zeros_like(ri_range, dtype=np.float32)

    img_range_u8 = (img_range * 255).astype(np.uint8)

    # 4) Convert intensity to 8-bit using 1–99 percentile (robust to outliers)
    valid_int = ri_intensity > 0
    if np.any(valid_int):
        p1, p99 = np.percentile(ri_intensity[valid_int], [1, 99])
        denom = max(p99 - p1, 1e-6)
        img_int = (ri_intensity - p1) / denom
        img_int = np.clip(img_int, 0.0, 1.0)
    else:
        img_int = np.zeros_like(ri_intensity, dtype=np.float32)

    img_int_u8 = (img_int * 255).astype(np.uint8)

    # 5) Stack vertically (range on top, intensity bottom) and return uint8
    img_range_intensity = np.vstack((img_range_u8, img_int_u8)).astype(np.uint8)

    #######
    ####### ID_S1_EX1 END #######

    return img_range_intensity

import numpy as np
import cv2

def show_counts_grid_20x20(pcl_xy, counts, H, W, win_name="counts_20x20",
                          grid_size=20, colormap=cv2.COLORMAP_JET):
    """
    Visualize point-counts as a coarse (about 20x20) grid heatmap with text labels.

    Parameters
    ----------
    pcl_xy : (M,2) int array
        BEV indices (x=row, y=col) for each unique cell (e.g., pcl_int_unique[:,0:2]).
    counts : (M,) int array
        Point counts per unique BEV cell (aligned with pcl_xy).
    H, W : int
        Original BEV map size.
    grid_size : int
        Number of coarse bins per axis (default 20 -> ~20x20).
    """

    pcl_xy = np.asarray(pcl_xy, dtype=np.int32)
    counts = np.asarray(counts, dtype=np.int32)

    # --- aggregate original cells into coarse bins ---
    bin_x = np.clip((pcl_xy[:, 0] * grid_size) // H, 0, grid_size - 1)  # rows
    bin_y = np.clip((pcl_xy[:, 1] * grid_size) // W, 0, grid_size - 1)  # cols

    coarse = np.zeros((grid_size, grid_size), dtype=np.int32)
    np.add.at(coarse, (bin_x, bin_y), counts)

    # --- log-normalize for visualization ---
    coarse_f = coarse.astype(np.float32)
    coarse_norm = np.log(coarse_f + 1.0)
    denom = np.log(np.max(coarse_f) + 1.0 + 1e-6)
    coarse_norm = coarse_norm / (denom if denom > 0 else 1.0)

    # --- make a bigger image so grid + text is readable ---
    cell_px = 30  # size of each coarse cell in pixels
    vis_h, vis_w = grid_size * cell_px, grid_size * cell_px

    img_gray = (coarse_norm * 255).astype(np.uint8)
    img_gray_big = cv2.resize(img_gray, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
    img_color = cv2.applyColorMap(img_gray_big, colormap)

    # --- draw grid lines ---
    for k in range(grid_size + 1):
        y = k * cell_px
        x = k * cell_px
        cv2.line(img_color, (0, y), (vis_w, y), (0, 0, 0), 1)
        cv2.line(img_color, (x, 0), (x, vis_h), (0, 0, 0), 1)

    # --- overlay counts text (only where count > 0) ---
    for gx in range(grid_size):
        for gy in range(grid_size):
            c = coarse[gx, gy]
            if c <= 0:
                continue

            text = str(int(c))
            # center text in cell
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cx = gy * cell_px + (cell_px - tw) // 2
            cy = gx * cell_px + (cell_px + th) // 2

            # black outline for readability + white fill
            cv2.putText(img_color, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img_color, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # --- show and close on window X ---
    cv2.imshow(win_name, img_color)
    while cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(10)
    cv2.destroyWindow(win_name)


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs, vis=False):
    """
    Create BEV (Bird's Eye View) feature maps (intensity, height, density) from LiDAR point cloud.

    Parameters
    ----------
    lidar_pcl : np.ndarray
        Point cloud in LiDAR/sensor coordinates with shape (N, 4): [x, y, z, intensity].
    configs : object
        Must provide:
          - lim_x = [min_x, max_x]
          - lim_y = [min_y, max_y]
          - lim_z = [min_z, max_z]
          - bev_height (H)
          - bev_width  (W)
          - device (torch device)
    vis : bool
        If True, show intermediate visualizations.

    Returns
    -------
    input_bev_maps : torch.Tensor
        Tensor of shape (1, 3, H, W) with channels [intensity, height, density], dtype=float32, on configs.device.
    """

    # quick guards
    lidar_pcl = np.asarray(lidar_pcl)
    if lidar_pcl.ndim != 2 or lidar_pcl.shape[1] < 4:
        raise ValueError(f"lidar_pcl must be Nx4 (x,y,z,intensity), got shape {lidar_pcl.shape}")

    ####### ID_S2_EX1 START #######
    print("student task ID_S2_EX1")

    # 1) Filter points inside ROI
    # Use +1 sizing for internal grids to match reference discretization
    H, W = int(configs.bev_height), int(configs.bev_width)
    mask = (
        (lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
        (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
        (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1])
    )
    pcl_in = lidar_pcl[mask].copy()

    if pcl_in.shape[0] == 0:
        # return zero tensor of expected shape (1,3,H,W)
        bev_maps = torch.zeros((1, 3, H, W), dtype=torch.float32, device=getattr(configs, "device", "cpu"))
        return bev_maps

    # 2) Shift ground plane (so z is non-negative)
    # Only subtract once; downstream code assumes this already
    pcl_in[:, 2] -= configs.lim_z[0]  # z' in [0, lim_z_range]

    # 3) Discretize to BEV image coords (square cells based on x-range / H)
    x_range = float(configs.lim_x[1] - configs.lim_x[0])
    bev_discret = x_range / float(H)  # size of one pixel in meters (x-direction)

    pcl = pcl_in.copy()
    # x map: floor((x - min_x) / discret)
    pcl[:, 0] = np.floor((pcl[:, 0] - configs.lim_x[0]) / bev_discret)
    # y map using centered discretization: floor(y / discret) + (W/2)
    pcl[:, 1] = np.floor(pcl[:, 1] / bev_discret) + ((W + 1 ) / 2.0)
    pcl[:, 0:2] = pcl[:, 0:2].astype(np.int32)

    # Clip to valid image bounds
    pcl[:, 0] = np.clip(pcl[:, 0], 0, H)
    pcl[:, 1] = np.clip(pcl[:, 1], 0, W)

    # optional visualization of discretized pcl (uses user's show_pcl)
    if vis:
        try:
            show_pcl(pcl[:, :3])
        except Exception:
            pass
    ####### ID_S2_EX1 END #######

    # ------------------------------
    # EX2: Intensity map
    # ------------------------------
    ####### ID_S2_EX2 START #######
    print("student task ID_S2_EX2")

    intensity_map = np.zeros((H+1, W+1), dtype=np.float32)

    # Clip raw intensity (assuming expected range [0,1])
    pcl_int = pcl.copy()
    pcl_int[:, 3] = np.clip(pcl_int[:, 3], 0.0, 1.0)

    # Sort so the strongest intensity per cell comes first (then unique keeps that)
    idx_i = np.lexsort((-pcl_int[:, 3], pcl_int[:, 1], pcl_int[:, 0]))
    pcl_int_sorted = pcl_int[idx_i]

    # unique per cell: get index (first occurrence) and counts (for density)
    unique_xy, unique_idx_i, counts = np.unique(
        pcl_int_sorted[:, 0:2], axis=0, return_index=True, return_counts=True
    )
    pcl_int_unique = pcl_int_sorted[unique_idx_i]

    # Robust normalize via percentiles to mitigate outliers
    i = pcl_int_unique[:, 3].astype(np.float32)
    if i.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = np.percentile(i, [1, 99])
    i_clip = np.clip(i, lo, hi)
    i_norm = (i_clip - lo) / (hi - lo + 1e-6)

    intensity_map[pcl_int_unique[:, 0].astype(np.int32), pcl_int_unique[:, 1].astype(np.int32)] = np.clip(i_norm, 0.0, 1.0)

    # visualize intensity
    if vis:
        img_intensity = (intensity_map * 255).astype(np.uint8)
        cv2.imshow("img_intensity", img_intensity)
        while cv2.getWindowProperty("img_intensity", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.waitKey(10)

    # optional coarse counts visualization
    if vis:
        try:
            show_counts_grid_20x20(pcl_int_unique[:, 0:2], counts, H, W)
        except Exception:
            pass
    ####### ID_S2_EX2 END #######

    # ------------------------------
    # EX3: Height map
    # ------------------------------
    ####### ID_S2_EX3 START #######
    print("student task ID_S2_EX3")

    height_map = np.zeros((H+1, W+1), dtype=np.float32)

    # Sort to keep top-most z per cell
    idx_h = np.lexsort((-pcl[:, 2], pcl[:, 1], pcl[:, 0]))
    pcl_sorted = pcl[idx_h]
    unique_xy_h, unique_idx_h = np.unique(pcl_sorted[:, 0:2], axis=0, return_index=True)
    pcl_top = pcl_sorted[unique_idx_h]

    # Normalize heights to [0,1] using config z-range
    z_range = float(configs.lim_z[1] - configs.lim_z[0])
    z_norm = pcl_top[:, 2] / (z_range + 1e-6)
    height_map[pcl_top[:, 0].astype(np.int32), pcl_top[:, 1].astype(np.int32)] = np.clip(z_norm, 0.0, 1.0)

    # visualize height
    if vis:
        img_height = (height_map * 255).astype(np.uint8)
        cv2.imshow("img_height", img_height)
        while cv2.getWindowProperty("img_height", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.waitKey(10)

    ####### ID_S2_EX3 END #######

    lidar_pcl_cpy = pcl
    lidar_pcl_top = pcl_top

    # ------------------------------
    # Density map (counts) — correct alignment between counts and grid coordinates
    # ------------------------------
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts

    # ------------------------------
    # Assemble BEV map and convert to torch tensor
    # ------------------------------
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()

    return input_bev_maps

