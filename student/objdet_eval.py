# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import matplotlib

matplotlib.use("wxagg")  # change backend so that figure maximizing works on Mac as well
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon

# add project directory to python path to enable relative imports
import os
import sys

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools

from operator import itemgetter
import math

# def _rect_corners(cx, cy, w, l, yaw):
#     """4 corners of oriented rectangle in BEV."""
#     hl = l / 2.0
#     hw = w / 2.0
#     # local corners (x forward, y left) in clockwise order
#     local = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
#     c = math.cos(yaw)
#     s = math.sin(yaw)
#     return [(cx + lx * c - ly * s, cy + lx * s + ly * c) for lx, ly in local]

def _poly_area(poly):
    """Shoelace formula area."""
    if not poly:
        return 0.0
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0

def _inside(p, a, b):
    """Point p is inside half-plane to the right of directed edge a->b."""
    (x, y) = p
    (x1, y1) = a
    (x2, y2) = b
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0.0

def _line_intersection(s, e, a, b):
    """Intersection of segment s->e with infinite line a->b."""
    x1, y1 = s
    x2, y2 = e
    x3, y3 = a
    x4, y4 = b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return (px, py)

def _polygon_clip(subject, clip):
    """Sutherland–Hodgman clip subject polygon by convex clip polygon."""
    if not subject or not clip:
        return []
    out = subject
    a = clip[-1]
    for b in clip:
        inp = out
        out = []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if _inside(e, a, b):
                if not _inside(s, a, b):
                    out.append(_line_intersection(s, e, a, b))
                out.append(e)
            elif _inside(s, a, b):
                out.append(_line_intersection(s, e, a, b))
            s = e
        a = b
    return out

def _iou_bev(c1, c2):
    """IoU between two oriented rectangles given corners."""
    a1 = _poly_area(c1)
    a2 = _poly_area(c2)
    if a1 <= 0.0 or a2 <= 0.0:
        return 0.0
    inter = _polygon_clip(c1, c2)
    ai = _poly_area(inter)
    union = a1 + a2 - ai
    return 0.0 if union <= 0.0 else (ai / union)


# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):

    # find best detection for each valid label
    true_positives = 0  # no. of correctly detected objects
    center_devs = []
    ious = []
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if not valid:  # exclude all labels from statistics which are not considered valid
            continue

        # compute intersection over union (iou) and distance between centers

        ####### ID_S4_EX1 START #######
        #######
        print("student task ID_S4_EX1 ")

        ## step 1 : extract the four corners of the current label bounding-box
        box = label.box
        lx = float(box.center_x)
        ly = float(box.center_y)
        lz = float(box.center_z)
        lw = float(box.width)
        ll = float(box.length)
        lyaw = float(box.heading)
        label_corners = tools.compute_box_corners(lx, ly, lw, ll, lyaw)

        ## step 2 : loop over all detected objects
        matches_lab_det = []
        for det in detections:
            if det is None or len(det) != 8:
                continue

            ## step 3 : extract the four corners of the current detection
            cls_id = int(det[0])
            cx, cy, cz, h, w, l, yaw = map(float, det[1:])

            ## step 4 : computer the center distance between label and detection bounding-box in x, y, and z
            det_corners = tools.compute_box_corners(cx, cy, w, l, yaw)

            ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box
            iou = _iou_bev(label_corners, det_corners)

            ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
            dist_x = abs(lx - cx)
            dist_y = abs(ly - cy)
            dist_z = abs(lz - cz)
            if iou >= min_iou:
                # store as (iou, dx, dy, dz); best iou wins
                matches_lab_det.append((iou, dist_x, dist_y, dist_z))

        #######
        ####### ID_S4_EX1 END #######

        # find best match and compute metrics
        if matches_lab_det:
            best_match = max(
                matches_lab_det, key=itemgetter(0)
            )  # retrieve entry with max iou in case of multiple candidates
            ious.append(best_match[0])
            center_devs.append(best_match[1:])
            true_positives += 1

    ####### ID_S4_EX2 START #######
    #######
    print("student task ID_S4_EX2")

    # compute positives and negatives for precision/recall

    ## step 1 : compute the total number of positives present in the scene
    all_positives = int(sum(labels_valid))

    ## step 2 : compute the number of false negatives
    false_negatives =  all_positives - true_positives

    ## step 3 : compute the number of false positives
    false_positives = len(detections) - true_positives

    #######
    ####### ID_S4_EX2 END #######

    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]

    return det_performance


# evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all):

    # extract elements
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])

    ####### ID_S4_EX3 START #######
    #######
    print("student task ID_S4_EX3")

    ## step 1 : extract the total number of positives, true positives, false negatives and false positives
    all_positives = 0
    true_positives = 0
    false_negatives = 0
    false_positives = 0

    for pn in pos_negs:
        # pn expected as (all_pos, tp, fn, fp)
        all_positives += int(pn[0])
        true_positives += int(pn[1])
        false_negatives += int(pn[2])
        false_positives += int(pn[3])

    ## step 2 : compute precision
    denom_p = true_positives + false_positives
    precision = (true_positives / denom_p) if denom_p > 0 else 0.0

    ## step 3 : compute recall
    denom_r = true_positives + false_negatives
    recall = (true_positives / denom_r) if denom_r > 0 else 0.0

    #######
    ####### ID_S4_EX3 END #######
    print("precision = " + str(precision) + ", recall = " + str(recall))

    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)

    # plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = [
        "detection precision",
        "detection recall",
        "intersection over union",
        "position errors in X",
        "position errors in Y",
        "position error in Z",
    ]
    textboxes = [
        "",
        "",
        "",
        "\n".join(
            (
                r"$\mathrm{mean}=%.4f$" % (np.mean(devs_x_all),),
                r"$\mathrm{sigma}=%.4f$" % (np.std(devs_x_all),),
                r"$\mathrm{n}=%.0f$" % (len(devs_x_all),),
            )
        ),
        "\n".join(
            (
                r"$\mathrm{mean}=%.4f$" % (np.mean(devs_y_all),),
                r"$\mathrm{sigma}=%.4f$" % (np.std(devs_y_all),),
                r"$\mathrm{n}=%.0f$" % (len(devs_x_all),),
            )
        ),
        "\n".join(
            (
                r"$\mathrm{mean}=%.4f$" % (np.mean(devs_z_all),),
                r"$\mathrm{sigma}=%.4f$" % (np.std(devs_z_all),),
                r"$\mathrm{n}=%.0f$" % (len(devs_x_all),),
            )
        ),
    ]

    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(
                0.05,
                0.95,
                textboxes[idx],
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=props,
            )
    plt.tight_layout()
    plt.show()
    pass
    pass

