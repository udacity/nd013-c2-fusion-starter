# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Collection of tools for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

##################
## IMPORTS

## general package imports
import cv2
import numpy as np
import math
from shapely.geometry import Polygon

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2



##################
# LIDAR

def compute_beam_inclinations(calibration, height):
    """ Compute the inclination angle for each beam in a range image. """

    if len(calibration.beam_inclinations) > 0:
        return np.array(calibration.beam_inclinations)
    else:
        inclination_min = calibration.beam_inclination_min
        inclination_max = calibration.beam_inclination_max

        return np.linspace(inclination_min, inclination_max, height)


def compute_range_image_polar(range_image, extrinsic, inclination):
    """ Convert a range image to polar coordinates. """

    height = range_image.shape[0]
    width = range_image.shape[1]

    az_correction = math.atan2(extrinsic[1,0], extrinsic[0,0])
    azimuth = np.linspace(np.pi,-np.pi,width) - az_correction

    azimuth_tiled = np.broadcast_to(azimuth[np.newaxis,:], (height,width))
    inclination_tiled = np.broadcast_to(inclination[:,np.newaxis],(height,width))

    return np.stack((azimuth_tiled,inclination_tiled,range_image))


def compute_range_image_cartesian(range_image_polar, extrinsic, pixel_pose, frame_pose):
    """ Convert polar coordinates to cartesian coordinates. """

    azimuth = range_image_polar[0]
    inclination = range_image_polar[1]
    range_image_range = range_image_polar[2]

    cos_azimuth = np.cos(azimuth)
    sin_azimuth = np.sin(azimuth)
    cos_incl = np.cos(inclination)
    sin_incl = np.sin(inclination)

    x = cos_azimuth * cos_incl * range_image_range
    y = sin_azimuth * cos_incl * range_image_range
    z = sin_incl * range_image_range

    range_image_points = np.stack([x,y,z,np.ones_like(z)])
    range_image_points = np.einsum('ij,jkl->ikl', extrinsic,range_image_points)

    return range_image_points


def get_rotation_matrix(roll, pitch, yaw):
    """ Convert Euler angles to a rotation matrix"""

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)

    ones = np.ones_like(yaw)
    zeros = np.zeros_like(yaw)

    r_roll = np.stack([
        [ones,  zeros,     zeros],
        [zeros, cos_roll, -sin_roll],
        [zeros, sin_roll,  cos_roll]])

    r_pitch = np.stack([
        [ cos_pitch, zeros, sin_pitch],
        [ zeros,     ones,  zeros],
        [-sin_pitch, zeros, cos_pitch]])

    r_yaw = np.stack([
        [cos_yaw, -sin_yaw, zeros],
        [sin_yaw,  cos_yaw, zeros],
        [zeros,    zeros,   ones]])

    pose = np.einsum('ijhw,jkhw,klhw->ilhw',r_yaw,r_pitch,r_roll)
    pose = pose.transpose(2,3,0,1)
    return pose


def project_to_pointcloud(frame, ri, camera_projection, range_image_pose, calibration):
    """ Create a pointcloud in vehicle space from LIDAR range image. """
    beam_inclinations = compute_beam_inclinations(calibration, ri.shape[0])
    beam_inclinations = np.flip(beam_inclinations)

    extrinsic = np.array(calibration.extrinsic.transform).reshape(4,4)
    frame_pose = np.array(frame.pose.transform).reshape(4,4)

    ri_polar = compute_range_image_polar(ri[:,:,0], extrinsic, beam_inclinations)

    #    if range_image_pose is None:
    #        pixel_pose = None
    #    else:
    #        pixel_pose = get_rotation_matrix(range_image_pose[:,:,0], range_image_pose[:,:,1], range_image_pose[:,:,2])
    #        translation = range_image_pose[:,:,3:]
    #        pixel_pose = np.block([
    #            [pixel_pose, translation[:,:,:,np.newaxis]],
    #            [np.zeros_like(translation)[:,:,np.newaxis],np.ones_like(translation[:,:,0])[:,:,np.newaxis,np.newaxis]]])

    ri_cartesian = compute_range_image_cartesian(ri_polar, extrinsic, None, frame_pose)
    ri_cartesian = ri_cartesian.transpose(1,2,0)

    mask = ri[:,:,0] > 0

    return ri_cartesian[mask,:3], ri[mask]


def display_laser_on_image(img, pcl, vehicle_to_image):
    # Convert the pointcloud to homogeneous coordinates.
    pcl1 = np.concatenate((pcl,np.ones_like(pcl[:,0:1])),axis=1)

    # Transform the point cloud to image space.
    proj_pcl = np.einsum('ij,bj->bi', vehicle_to_image, pcl1) 

    # Filter LIDAR points which are behind the camera.
    mask = proj_pcl[:,2] > 0
    proj_pcl = proj_pcl[mask]
    proj_pcl_attr = pcl_attr[mask]

    # Project the point cloud onto the image.
    proj_pcl = proj_pcl[:,:2]/proj_pcl[:,2:3]

    # Filter points which are outside the image.
    mask = np.logical_and(
        np.logical_and(proj_pcl[:,0] > 0, proj_pcl[:,0] < img.shape[1]),
        np.logical_and(proj_pcl[:,1] > 0, proj_pcl[:,1] < img.shape[1]))

    proj_pcl = proj_pcl[mask]
    proj_pcl_attr = proj_pcl_attr[mask]

    # Colour code the points based on distance.
    coloured_intensity = 255*cmap(proj_pcl_attr[:,0]/30)

    # Draw a circle for each point.
    for i in range(proj_pcl.shape[0]):
        cv2.circle(img, (int(proj_pcl[i,0]),int(proj_pcl[i,1])), 1, coloured_intensity[i])

# get lidar point cloud from frame
def pcl_from_range_image(frame, lidar_name):

    # extract lidar data and range image
    lidar = waymo_utils.get(frame.lasers, lidar_name)
    range_image, camera_projection, range_image_pose = waymo_utils.parse_range_image_and_camera_projection(lidar)    # Parse the top laser range image and get the associated projection.

    # Convert the range image to a point cloud
    lidar_calib = waymo_utils.get(frame.context.laser_calibrations, lidar_name)
    pcl, pcl_attr = project_to_pointcloud(frame, range_image, camera_projection, range_image_pose, lidar_calib)

    # stack point cloud and lidar intensity
    points_all = np.column_stack((pcl, pcl_attr[:, 1]))

    return points_all




##################
# BIRDS-EYE VIEW

# project detected bounding boxes into birds-eye view
def project_detections_into_bev(bev_map, detections, configs, color=[]):
    for row in detections:
        # extract detection
        _id, _x, _y, _z, _h, _w, _l, _yaw = row

        # convert from metric into pixel coordinates
        x = (_y - configs.lim_y[0]) / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
        y = (_x - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
        z = _z - configs.lim_z[0]
        w = _w / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
        l = _l / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
        yaw = -_yaw

        # draw object bounding box into birds-eye view
        if not color:
            color = configs.obj_colors[int(_id)]
        
        # get object corners within bev image
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw # front left
        bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw 
        bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw # rear left
        bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw
        bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw # rear right
        bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw
        bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw # front right
        bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw
        
        # draw object as box
        corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
        cv2.polylines(bev_map, [corners_int], True, color, 2)

        # draw colored line to identify object front
        corners_int = bev_corners.reshape(-1, 2)
        cv2.line(bev_map, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)




##################
# LABELS AND OBJECTS

# extract object labels from frame
def validate_object_labels(object_labels, pcl, configs, min_num_points):

    ## Create initial list of flags where every object is set to `valid`
    valid_flags = np.ones(len(object_labels)).astype(bool)

    ## Mark labels as invalid that do not enclose a sufficient number of lidar points
    vehicle_to_labels = [np.linalg.inv(waymo_utils.get_box_transformation_matrix(label.box)) for label in object_labels] # for each label, compute transformation matrix from vehicle space to box space
    vehicle_to_labels = np.stack(vehicle_to_labels)

    pcl_no_int = pcl[:, :3] # strip away intensity information from point cloud 
    pcl1 = np.concatenate((pcl_no_int, np.ones_like(pcl_no_int[:, 0:1])), axis=1) # convert pointcloud to homogeneous coordinates    
    proj_pcl = np.einsum('lij,bj->lbi', vehicle_to_labels, pcl1) # transform point cloud to label space for each label (proj_pcl shape is [label, LIDAR point, coordinates])
    mask = np.logical_and.reduce(np.logical_and(proj_pcl >= -1, proj_pcl <= 1), axis=2) # for each pair of LIDAR point & label, check if point is inside the label's box (mask shape is [label, LIDAR point])

    counts = mask.sum(1) # count points inside each label's box and keep boxes which contain min. no of points
    valid_flags = counts >= min_num_points

    ## Mark labels as invalid which are ...
    for index, label in enumerate(object_labels):

        ## ... outside the object detection range
        label_obj = [label.type, label.box.center_x, label.box.center_y, label.box.center_z,
                     label.box.height, label.box.width, label.box.length, label.box.heading]
        valid_flags[index] = valid_flags[index] and is_label_inside_detection_area(label_obj, configs)                     

        ## ... flagged as "difficult to detect" or not of type "vehicle" 
        if(label.detection_difficulty_level > 0 or label.type != label_pb2.Label.Type.TYPE_VEHICLE):
            valid_flags[index] = False
        
    
    return valid_flags


# convert ground truth labels into 3D objects
def convert_labels_into_objects(object_labels, configs):
    
    detections = []
    for label in object_labels:
        # transform label into a candidate object
        if label.type==1 : # only use vehicles
            candidate = [label.type, label.box.center_x, label.box.center_y, label.box.center_z,
                         label.box.height, label.box.width, label.box.length, label.box.heading]

            # only add to object list if candidate is within detection area    
            if(is_label_inside_detection_area(candidate, configs)):
                detections.append(candidate)

    return detections


# compute location of each corner of a box and returns [front_left, rear_left, rear_right, front_right]
def compute_box_corners(x,y,w,l,yaw):
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    fl = (x - w / 2 * cos_yaw - l / 2 * sin_yaw,  # front left
          y - w / 2 * sin_yaw + l / 2 * cos_yaw)

    rl = (x - w / 2 * cos_yaw + l / 2 * sin_yaw,  # rear left
          y - w / 2 * sin_yaw - l / 2 * cos_yaw)

    rr = (x + w / 2 * cos_yaw + l / 2 * sin_yaw,  # rear right
          y + w / 2 * sin_yaw - l / 2 * cos_yaw)

    fr = (x + w / 2 * cos_yaw - l / 2 * sin_yaw,  # front right
          y + w / 2 * sin_yaw + l / 2 * cos_yaw)

    return [fl,rl,rr,fr]


# checks whether label is inside detection area
def is_label_inside_detection_area(label, configs, min_overlap=0.5):

    # convert current label object into Polygon object
    _, x, y, _, _, w, l, yaw = label
    label_obj_corners = compute_box_corners(x,y,w,l,yaw)
    label_obj_poly = Polygon(label_obj_corners)   

    # convert detection are into polygon
    da_w = (configs.lim_x[1] - configs.lim_x[0])  # width
    da_l = (configs.lim_y[1] - configs.lim_y[0])  # length
    da_x = configs.lim_x[0] + da_w/2  # center in x
    da_y = configs.lim_y[0] + da_l/2  # center in y
    da_corners = compute_box_corners(da_x,da_y,da_w,da_l,0)
    da_poly = Polygon(da_corners)   

    # check if detection area contains label object
    intersection = da_poly.intersection(label_obj_poly)
    overlap = intersection.area / label_obj_poly.area

    return False if(overlap <= min_overlap) else True



##################
# VISUALIZATION

# extract RGB front camera image and camera calibration
def extract_front_camera_image(frame):
    # extract camera and calibration from frame
    camera_name = dataset_pb2.CameraName.FRONT
    camera = waymo_utils.get(frame.images, camera_name)

    # get image and convert tom RGB
    image = waymo_utils.decode_image(camera)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def show_bev(bev_maps, configs):

    bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bev_map = cv2.resize(bev_map, (configs.bev_width, configs.bev_height))
    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
    cv2.imshow('BEV map', bev_map)


# visualize ground-truth labels as overlay in birds-eye view
def show_objects_labels_in_bev(detections, object_labels, bev_maps, configs):

    # project detections and labels into birds-eye view
    bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bev_map = cv2.resize(bev_map, (configs.bev_width, configs.bev_height))
    
    label_detections = convert_labels_into_objects(object_labels, configs)
    project_detections_into_bev(bev_map, label_detections, configs, [0,255,0])
    project_detections_into_bev(bev_map, detections, configs, [0,0,255])
    

    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
    cv2.imshow('labels (green) vs. detected objects (red)', bev_map)


# visualize detection results as overlay in birds-eye view and ground-truth labels in camera image
def show_objects_in_bev_labels_in_camera(detections, bev_maps, image, object_labels, object_labels_valid, camera_calibration, configs):

    # project detections into birds-eye view
    bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bev_map = cv2.resize(bev_map, (configs.bev_width, configs.bev_height))
    project_detections_into_bev(bev_map, detections, configs)
    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

    # project ground-truth labels into camera image
    img_rgb = project_labels_into_camera(camera_calibration, image, object_labels, object_labels_valid)

    # merge camera image and bev image into a combined view
    img_rgb_h, img_rgb_w = img_rgb.shape[:2]
    ratio_rgb = configs.output_width / img_rgb_w
    output_rgb_h = int(ratio_rgb * img_rgb_h)
    ret_img_rgb = cv2.resize(img_rgb, (configs.output_width, output_rgb_h))

    img_bev_h, img_bev_w = bev_map.shape[:2]
    ratio_bev = configs.output_width / img_bev_w
    output_bev_h = int(ratio_bev * img_bev_h)
    ret_img_bev = cv2.resize(bev_map, (configs.output_width, output_bev_h))

    out_img = np.zeros((output_rgb_h + output_bev_h, configs.output_width, 3), dtype=np.uint8)
    out_img[:output_rgb_h, ...] = ret_img_rgb
    out_img[output_rgb_h:, ...] = ret_img_bev

    # show combined view
    cv2.imshow('labels vs. detected objects', out_img)


# visualize object labels in camera image
def project_labels_into_camera(camera_calibration, image, labels, labels_valid, img_resize_factor=1.0):

    # get transformation matrix from vehicle frame to image
    vehicle_to_image = waymo_utils.get_image_transform(camera_calibration)

    # draw all valid labels
    for label, vis in zip(labels, labels_valid):
        if vis:
            colour = (0, 255, 0)
        else:
            colour = (255, 0, 0)

        # only show labels of type "vehicle"
        if(label.type == label_pb2.Label.Type.TYPE_VEHICLE):
            waymo_utils.draw_3d_box(image, vehicle_to_image, label, colour=colour)

    # resize image
    if (img_resize_factor < 1.0):
        width = int(image.shape[1] * img_resize_factor)
        height = int(image.shape[0] * img_resize_factor)
        dim = (width, height)
        img_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return img_resized
    else:
        return image

