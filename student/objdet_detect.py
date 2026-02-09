# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Detect 3D objects in lidar point clouds using deep learning
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import torch
from easydict import EasyDict as edict
from tools.objdet_models.resnet.utils.misc import make_folder, time_synchronized
from tools.objdet_models.resnet.utils.torch_utils import _sigmoid
from collections import defaultdict

# add project directory to python path to enable relative imports
import os
import sys

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# model-related
from tools.objdet_models.resnet.models import fpn_resnet
from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing

from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2


# load model-related parameters into an edict
def load_configs_model(model_name="darknet", configs=None):

    # init config file, if none has been passed
    if configs == None:
        configs = edict()

    # get parent directory of this file to enable relative paths
    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = configs.model_path = os.path.abspath(
        os.path.join(curr_path, os.pardir)
    )

    # set parameters according to model type
    if model_name == "darknet":
        configs.model_path = os.path.join(
            parent_path, "tools", "objdet_models", "darknet"
        )
        configs.pretrained_filename = os.path.join(
            configs.model_path, "pretrained", "complex_yolov4_mse_loss.pth"
        )
        configs.arch = "darknet"
        configs.batch_size = 4
        configs.cfgfile = os.path.join(
            configs.model_path, "config", "complex_yolov4.cfg"
        )
        configs.conf_thresh = 0.5
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False
        configs.min_iou = 0.5

    elif model_name == "fpn_resnet":
        ####### ID_S3_EX1-3 START #######
        #######
        print("student task ID_S3_EX1-3")
        configs.model_path = os.path.join(
            parent_path, "tools", "objdet_models", "resnet"
        )
        configs.pretrained_path = os.path.join(
            configs.model_path, "pretrained", "fpn_resnet_18_epoch_300.pth"
        )
        configs.pretrained_filename = os.path.join(
            configs.model_path, "pretrained", "fpn_resnet_18_epoch_300.pth"
        )
        configs.foldername = r"C:\Users\mlee\katana\udacity\nd013-c2-fusion-starter"
        configs.arch = "fpn_resnet_18"
        configs.saved_fn = "fpn_resnet_18"
        configs.K = 50
        configs.no_cuda = 'store_true'
        configs.gpu_idx = 0
        configs.peak_thresh = 0.2
        configs.output_format = "image"  # could be video
        configs.output_width = 608
        configs.pin_memory = True
        configs.distributed = False  # For testing on 1 GPU only

        configs.input_size = (608, 608)
        configs.hm_size = (152, 152)
        configs.down_ratio = 4
        configs.max_objects = 50

        configs.imagenet_pretrained = False
        configs.head_conv = 64
        configs.num_classes = 3
        configs.num_center_offset = 2
        configs.num_z = 1
        configs.num_dim = 3
        configs.num_direction = 2  # sin, cos
        configs.conf_thresh = 0.5

        configs.heads = {
            'hm_cen': configs.num_classes,
            'cen_offset': configs.num_center_offset,
            'direction': configs.num_direction,
            'z_coor': configs.num_z,
            'dim': configs.num_dim
        }

        ####################################################################
        ##############Dataset, Checkpoints, and results dir configs#########
        ####################################################################
        configs.root_dir = '../'
        configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti', 'demo')
        configs.calib_path = os.path.join(configs.root_dir, 'dataset', 'kitti', 'demo', 'calib.txt')
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

        #######
        ####### ID_S3_EX1-3 END #######

    else:
        raise ValueError("Error: Invalid model name")

    # GPU vs. CPU
    configs.no_cuda = True  # if true, cuda is not used
    configs.gpu_idx = 0  # GPU index to use.
    configs.device = torch.device(
        "cpu" if configs.no_cuda else "cuda:{}".format(configs.gpu_idx)
    )

    return configs


# load all object-detection parameters into an edict
def load_configs(model_name="fpn_resnet", configs=None):

    # init config file, if none has been passed
    if configs == None:
        configs = edict()

    # birds-eye view (bev) parameters
    configs.lim_x = [0, 50]  # detection range in m
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0]  # reflected lidar intensity
    configs.bev_width = 608  # pixel resolution of bev image
    configs.bev_height = 608

    # add model-dependent parameters
    configs = load_configs_model(model_name, configs)

    # visualization parameters
    configs.output_width = 608  # width of result image (height may vary)
    configs.obj_colors = [
        [0, 255, 255],
        [0, 0, 255],
        [255, 0, 0],
    ]  # 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2

    return configs


# create model according to selected model type
def create_model(configs):

    # check for availability of model file
    assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(
        configs.pretrained_filename
    )

    # create model depending on architecture name
    if (configs.arch == "darknet") and (configs.cfgfile is not None):
        print("using darknet")
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)

    elif "fpn_resnet" in configs.arch:
        print("using ResNet architecture with feature pyramid")

        ####### ID_S3_EX1-4 START #######
        #######
        print("student task ID_S3_EX1-4")
        try:
            arch_parts = configs.arch.split('_')
            num_layers = int(arch_parts[-1])
        except:
            raise ValueError

        model = fpn_resnet.get_pose_net(num_layers=num_layers, heads=configs.heads, head_conv=configs.head_conv,
                                        imagenet_pretrained=configs.imagenet_pretrained)


        #######
        ####### ID_S3_EX1-4 END #######

    else:
        assert False, "Undefined model backbone"

    # load model weights
    model.load_state_dict(torch.load(configs.pretrained_filename, map_location="cpu"))
    print("Loaded weights from {}\n".format(configs.pretrained_filename))

    # set model to evaluation state
    configs.device = torch.device(
        "cpu" if configs.no_cuda else "cuda:{}".format(configs.gpu_idx)
    )
    model = model.to(device=configs.device)  # load model to either cpu or gpu
    model.eval()

    return model

# print detected objects
def print_detections(detections):
    """
    Nicely print detection results.

    Expected format per detection:
    [cls, x, y, z, h, l, w, yaw]
    """

    def to_float(v):
        try:
            return float(v)
        except Exception:
            return v

    header = (
        f"{'ID':>3} | {'CLS':>3} | {'X':>8} | {'Y':>8} | {'Z':>6} | "
        f"{'H':>6} | {'L':>8} | {'W':>8} | {'YAW':>8}"
    )
    print(header)
    print("-" * len(header))

    for i, det in enumerate(detections):
        cls, x, y, z, h, l, w, yaw = map(to_float, det)

        print(
            f"{i:3d} | {int(cls):3d} | "
            f"{x:8.2f} | {y:8.2f} | {z:6.2f} | "
            f"{h:6.2f} | {l:8.2f} | {w:8.2f} | {yaw:8.4f}"
        )


# detect trained objects in birds-eye view
def detect_objects(input_bev_maps, model, configs):

    # deactivate autograd engine during test to reduce memory usage and speed up computations
    with torch.no_grad():

        # perform inference
        outputs = model(input_bev_maps)

        # decode model output into target object format
        if "darknet" in configs.arch:

            # perform post-processing
            output_post = post_processing_v2(
                outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh
            )
            detections = []
            for sample_i in range(len(output_post)):
                if output_post[sample_i] is None:
                    continue
                detection = output_post[sample_i]
                for obj in detection:
                    x, y, w, l, im, re, _, _, _ = obj
                    yaw = np.arctan2(im, re)
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])

            # Review detections
            print_detections(detections)

        elif "fpn_resnet" in configs.arch:
            # decode output and perform post-processing

            ####### ID_S3_EX1-5 START #######
            #######
            print("student task ID_S3_EX1-5")
            # input_bev_maps = input_bev_maps.unsqueeze(0).to(configs.device, non_blocking=True).float()
            t1 = time_synchronized()
            outputs = model(input_bev_maps)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            # detections size (batch_size, K, 10)
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs)
            t2 = time_synchronized()
            # Inference speed
            # fps = 1 / (t2 - t1)

            # show detections
            describe_detections_fpn(detections)

            #######
            ####### ID_S3_EX1-5 END #######

    ####### ID_S3_EX2 START #######
    #######
    # Extract 3d bounding boxes from model response
    print("student task ID_S3_EX2")
    objects = []

    ## step 1 : check whether there are any detections
    if (detections is None) or (len(detections) == 0) or (detections[0] is None):
        return objects

    if "fpn_resnet" in configs.arch:
        raw_dets = detections[0]  # batch size = 1 expected
        det_batch = []
        for cls_id, dets in raw_dets.items():
            if dets is None or len(dets) == 0:
                continue

            # dets: shape (N, 8) => [score, x_px, y_px, z, h, w_px, l_px, yaw]
            for d in dets:
                det_batch.append([int(cls_id), *d.tolist()])
    elif "darknet" in configs.arch:
        det_batch = detections

    ## step 2 : loop over all detections
    for det in det_batch:
        if det is None or len(det) == 0:
            continue

        cls_id = int(det[0])
        if "fpn_resnet" in configs.arch:
            score, x_px, y_px, z, h, w_px, l_px, yaw = map(float, det[1:])
        else:
            x_px, y_px, z, h, w_px, l_px, yaw = map(float, det[1:])

        # (optional) skip very low confidence
        # if score < configs.min_confidence: continue

        ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure
        # BEV discretization (meters per pixel)
        dx = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
        dy = (configs.lim_y[1] - configs.lim_y[0]) / configs.bev_width

        # --- center position: pixel → meter ---
        x_m = (y_px + 0.5) * dx
        y_m = (x_px + 0.5) * dy - (configs.lim_y[1] - configs.lim_y[0])/2.0

        x = float(np.clip(x_m, configs.lim_x[0], configs.lim_x[1]))
        y = float(np.clip(y_m, configs.lim_y[0], configs.lim_y[1]))
        z = float(np.clip(z, configs.lim_z[0], configs.lim_z[1]))

        # --- box size: pixel → meter ---
        w_m = float(w_px * dy)  # width
        l_m = float(l_px * dx)  # length
        h_m = float(h)  # height already in meters (DO NOT scale)

        # # --- yaw ---
        yaw_world = float(yaw)  # float(-yaw + np.pi / 2)

        ## step 4 : append the current object to the 'objects' array
        obj = [int(cls_id), x, y, z, h_m, w_m, l_m, yaw_world]

        objects.append(obj)

    #######
    ####### ID_S3_EX2 END #######

    return objects

def describe_detections_fpn(detections, class_names=None, score_thresh=0.0):
    """
    Pretty-print detections from an FPN-ResNet-18 model.

    Parameters
    ----------
    detections : list[dict]
        Output of detector: list with one dict per batch item.
        dict[class_id] -> array of shape (N, 8)
    class_names : dict or list, optional
        Mapping from class_id to class name.
        Example: {0: "Pedestrian", 1: "Car", 2: "Cyclist"}
    score_thresh : float
        Minimum confidence score to display.
    """

    if class_names is None:
        class_names = {
            0: "Pedestrian",
            1: "Car",
            2: "Cyclist"
        }

    batch = detections[0]  # usually batch size = 1

    print("\n=== FPN-ResNet-18 Detection Results ===")

    det_id = 0
    for cls_id, dets in batch.items():
        if dets.shape[0] == 0:
            continue

        cls_name = class_names.get(cls_id, f"Class {cls_id}")

        for d in dets:
            score, x, y, z, h, l, w, yaw = map(float, d)

            if score < score_thresh:
                continue

            print(
                f"[{det_id:02d}] {cls_name:<12} | "
                f"score={score:.3f} | "
                f"pos=(x={x:7.2f}, y={y:7.2f}, z={z:5.2f}) | "
                f"size=(h={h:4.2f}, l={l:6.2f}, w={w:6.2f}) | "
                f"yaw={yaw:7.4f}"
            )
            det_id += 1

    if det_id == 0:
        print("No detections above threshold.")


