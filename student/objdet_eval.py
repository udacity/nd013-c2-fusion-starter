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
from operator import itemgetter

# add project directory to python path to enable relative imports
import os
import sys

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools


# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
    # find best detection for each valid label
    true_positives = 0  # no. of correctly detected objects
    center_devs = []
    ious = []
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid:  # exclude all labels from statistics which are not considered valid
            # compute intersection over union (iou) and distance between centers

            ####### ID_S4_EX1 START #######
            #######
            print("student task ID_S4_EX1 ")

            ## step 1 : extract the four corners of the current label bounding-box
            bbox_corners_label = tools.compute_box_corners(
                x=label.box.center_x,
                y=label.box.center_y,
                w=label.box.width,
                l=label.box.length,
                yaw=label.box.heading,
            )
            bb_label = Polygon(bbox_corners_label)

            ## step 2 : loop over all detected objects
            for detection in detections:
                ## step 3 : extract the four corners of the current detection
                bbox_corners_detection = tools.compute_box_corners(
                    x=detection[1], y=detection[2], w=detection[5], l=detection[6], yaw=detection[7]
                )
                bb_detection = Polygon(bbox_corners_detection)

                ## step 4 : computer the center distance between label and detection bounding-box in x, y, and z
                dist_x = float(np.sqrt((detection[1] - label.box.center_x) ** 2))
                dist_y = float(np.sqrt((detection[2] - label.box.center_y) ** 2))
                dist_z = float(np.sqrt((detection[3] - label.box.center_z) ** 2))

                ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box
                polygon_intersection = bb_label.intersection(bb_detection).area
                polygon_union = bb_label.union(bb_detection).area
                iou = polygon_intersection / polygon_union

                ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
                if iou > min_iou:
                    # print(f"IoU {iou} exceeds threshold {min_iou}")
                    matches_lab_det.append([iou, dist_x, dist_y, dist_z])
            #######
            ####### ID_S4_EX1 END #######

        # find best match and compute metrics
        if matches_lab_det:
            best_match = max(
                matches_lab_det, key=itemgetter(1)
            )  # retrieve entry with max iou in case of multiple candidates
            ious.append(best_match[0])
            center_devs.append(best_match[1:])
            true_positives += 1

    ####### ID_S4_EX2 START #######
    #######
    print("student task ID_S4_EX2")

    # compute positives and negatives for precision/recall

    ## step 1 : compute the total number of positives present in the scene
    all_positives = len(detections)

    ## step 2 : compute the number of false negatives
    false_negatives = labels_valid.sum() - len(ious)

    ## step 3 : compute the number of false positives
    false_positives = all_positives - len(ious)

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

    ## step 2 : compute precision
    precision = 0.0

    ## step 3 : compute recall
    recall = 0.0

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

    # compute statistics
    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)

    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)

    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)
    # std_dev_x = np.std(devs_x)

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
                0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props
            )
    plt.tight_layout()
    plt.show()
