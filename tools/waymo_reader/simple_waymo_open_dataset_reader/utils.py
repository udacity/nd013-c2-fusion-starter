# Copyright (c) 2019, Grégoire Payen de La Garanderie, Durham University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import zlib
import math
import io

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2



def get_box_transformation_matrix(box):
    """Create a transformation matrix for a given label box pose."""

    tx,ty,tz = box.center_x,box.center_y,box.center_z
    c = math.cos(box.heading)
    s = math.sin(box.heading)

    sl, sh, sw = box.length, box.height, box.width

    return np.array([
        [ sl*c,-sw*s,  0,tx],
        [ sl*s, sw*c,  0,ty],
        [    0,    0, sh,tz],
        [    0,    0,  0, 1]])

def get_3d_box_projected_corners(vehicle_to_image, label):
    """Get the 2D coordinates of the 8 corners of a label's 3D bounding box.

    vehicle_to_image: Transformation matrix from the vehicle frame to the image frame.
    label: The object label
    """

    box = label.box

    # Get the vehicle pose
    box_to_vehicle = get_box_transformation_matrix(box)

    # Calculate the projection from the box space to the image space.
    box_to_image = np.matmul(vehicle_to_image, box_to_vehicle)


    # Loop through the 8 corners constituting the 3D box
    # and project them onto the image
    vertices = np.empty([2,2,2,2])
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k-0.5), (l-0.5), (m-0.5), 1.])

                # Project the point onto the image
                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return None

                vertices[k,l,m,:] = [v[0]/v[2], v[1]/v[2]]

    vertices = vertices.astype(np.int32)

    return vertices

def compute_2d_bounding_box(img_or_shape,points):
    """Compute the 2D bounding box for a set of 2D points.
    
    img_or_shape: Either an image or the shape of an image.
                  img_or_shape is used to clamp the bounding box coordinates.
    
    points: The set of 2D points to use
    """

    if isinstance(img_or_shape,tuple):
        shape = img_or_shape
    else:
        shape = img_or_shape.shape

    # Compute the 2D bounding box and draw a rectangle
    x1 = np.amin(points[...,0])
    x2 = np.amax(points[...,0])
    y1 = np.amin(points[...,1])
    y2 = np.amax(points[...,1])

    x1 = min(max(0,x1),shape[1])
    x2 = min(max(0,x2),shape[1])
    y1 = min(max(0,y1),shape[0])
    y2 = min(max(0,y2),shape[0])

    return (x1,y1,x2,y2)

def draw_3d_box(img, vehicle_to_image, label, colour=(255,128,128), draw_2d_bounding_box=False):
    """Draw a 3D bounding from a given 3D label on a given "img". "vehicle_to_image" must be a projection matrix from the vehicle reference frame to the image space.

    draw_2d_bounding_box: If set a 2D bounding box encompassing the 3D box will be drawn
    """
    import cv2

    vertices = get_3d_box_projected_corners(vehicle_to_image, label)

    if vertices is None:
        # The box is not visible in this image
        return

    if draw_2d_bounding_box:
        x1,y1,x2,y2 = compute_2d_bounding_box(img.shape, vertices)

        if (x1 != x2 and y1 != y2):
            cv2.rectangle(img, (x1,y1), (x2,y2), colour, thickness = 2)
    else:
        # Draw the edges of the 3D bounding box
        for k in [0, 1]:
            for l in [0, 1]:
                for idx1,idx2 in [((0,k,l),(1,k,l)), ((k,0,l),(k,1,l)), ((k,l,0),(k,l,1))]:
                    cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=2)
        # Draw a cross on the front face to identify front & back.
        for idx1,idx2 in [((1,0,0),(1,1,1)), ((1,1,0),(1,0,1))]:
            cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=2)

def draw_2d_box(img, label, colour=(255,128,128)):
    """Draw a 2D bounding from a given 2D label on a given "img".
    """
    import cv2

    box = label.box

    # Extract the 2D coordinates
    # It seems that "length" is the actual width and "width" is the actual height of the bounding box. Most peculiar.
    x1 = int(box.center_x - box.length/2)
    x2 = int(box.center_x + box.length/2)
    y1 = int(box.center_y - box.width/2)
    y2 = int(box.center_y + box.width/2)

    # Draw the rectangle
    cv2.rectangle(img, (x1,y1), (x2,y2), colour, thickness = 1)


def decode_image(camera):
    """ Decode the JPEG image. """

    from PIL import Image
    return np.array(Image.open(io.BytesIO(camera.image)))

def get_image_transform(camera_calibration):
    """ For a given camera calibration, compute the transformation matrix
        from the vehicle reference frame to the image space.
    """

    # TODO: Handle the camera distortions
    extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4,4)
    intrinsic = camera_calibration.intrinsic

    # Camera model:
    # | fx  0 cx 0 |
    # |  0 fy cy 0 |
    # |  0  0  1 0 |
    camera_model = np.array([
        [intrinsic[0], 0, intrinsic[2], 0],
        [0, intrinsic[1], intrinsic[3], 0],
        [0, 0,                       1, 0]])

    # Swap the axes around
    axes_transformation = np.array([
        [0,-1,0,0],
        [0,0,-1,0],
        [1,0,0,0],
        [0,0,0,1]])

    # Compute the projection matrix from the vehicle space to image space.
    vehicle_to_image = np.matmul(camera_model, np.matmul(axes_transformation, np.linalg.inv(extrinsic)))
    return vehicle_to_image

def parse_range_image_and_camera_projection(laser, second_response=False):
    """ Parse the range image for a given laser.

    second_response: If true, return the second strongest response instead of the primary response.
                     The second_response might be useful to detect the edge of objects
    """

    range_image_pose = None
    camera_projection = None

    if not second_response:
        # Return the strongest response if available
        if len(laser.ri_return1.range_image_compressed) > 0:
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(
                zlib.decompress(laser.ri_return1.range_image_compressed))
            ri = np.array(ri.data).reshape(ri.shape.dims)

            if laser.name == dataset_pb2.LaserName.TOP:
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    zlib.decompress(laser.ri_return1.range_image_pose_compressed))
                range_image_pose = np.array(range_image_top_pose.data).reshape(range_image_top_pose.shape.dims)
                
            camera_projection = dataset_pb2.MatrixInt32()
            camera_projection.ParseFromString(
                    zlib.decompress(laser.ri_return1.camera_projection_compressed))
            camera_projection = np.array(camera_projection.data).reshape(camera_projection.shape.dims)

    else:
        # Return the second strongest response if available

        if len(laser.ri_return2.range_image_compressed) > 0:
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(
                zlib.decompress(laser.ri_return2.range_image_compressed))
            ri = np.array(ri.data).reshape(ri.shape.dims)
                
            camera_projection = dataset_pb2.MatrixInt32()
            camera_projection.ParseFromString(
                    zlib.decompress(laser.ri_return2.camera_projection_compressed))
            camera_projection = np.array(camera_projection.data).reshape(camera_projection.shape.dims)

    return ri, camera_projection, range_image_pose


def get(object_list, name):
    """ Search for an object by name in an object list. """

    object_list = [obj for obj in object_list if obj.name == name]
    return object_list[0]

