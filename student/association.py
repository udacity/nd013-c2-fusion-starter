# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params


class Association:
    """Data association class with single nearest neighbor association and gating based on Mahalanobis distance"""

    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []

    def associate(self, track_list, meas_list, KF):

        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############

        self.association_matrix = np.matrix([])  # reset matrix
        self.unassigned_tracks = []  # reset lists
        self.unassigned_meas = []

        n_tracks = len(track_list)
        n_meas = len(meas_list)
        if n_tracks == 0 or n_meas == 0:
            self.unassigned_tracks = list(range(n_tracks))
            self.unassigned_meas = list(range(n_meas))
            return

        self.unassigned_tracks = list(range(n_tracks))
        self.unassigned_meas = list(range(n_meas))

        self.association_matrix = np.matrix(
            np.full((n_tracks, n_meas), np.inf, dtype=float)
        )

        for i, track in enumerate(track_list):
            for j, meas in enumerate(meas_list):
                mhd = self.MHD(track, meas, KF)
                if self.gating(mhd, meas.sensor):
                    self.association_matrix[i, j] = mhd

        # debug: summarize association matrix stats
        sensor_name = meas_list[0].sensor.name if n_meas > 0 else "none"
        finite_mask = np.isfinite(self.association_matrix)
        finite_count = int(np.sum(finite_mask))
        if finite_count > 0:
            min_mhd = float(np.min(self.association_matrix[finite_mask]))
            print(
                f"[assoc:{sensor_name}] tracks={n_tracks} meas={n_meas} "
                f"finite={finite_count} min_MHD={min_mhd:.3f}"
            )
        else:
            print(
                f"[assoc:{sensor_name}] tracks={n_tracks} meas={n_meas} "
                "finite=0 (all gated)"
            )

        ############
        # END student code
        ############

    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        if self.association_matrix.size == 0:
            return np.nan, np.nan

        min_index = np.unravel_index(
            np.argmin(self.association_matrix, axis=None),
            self.association_matrix.shape,
        )
        min_value = self.association_matrix[min_index]
        if np.isinf(min_value):
            return np.nan, np.nan

        update_track = self.unassigned_tracks[min_index[0]]
        update_meas = self.unassigned_meas[min_index[1]]

        # remove from list
        self.unassigned_tracks.remove(update_track)
        self.unassigned_meas.remove(update_meas)
        self.association_matrix = np.delete(self.association_matrix, min_index[0], 0)
        self.association_matrix = np.delete(self.association_matrix, min_index[1], 1)

        ############
        # END student code
        ############
        return update_track, update_meas

    def gating(self, MHD, sensor):
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############

        threshold = chi2.ppf(params.gating_threshold, sensor.dim_meas)
        return MHD < threshold

        ############
        # END student code
        ############

    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############

        H = meas.sensor.get_H(track.x)
        S = KF.S(track, meas, H)
        gamma = KF.gamma(track, meas)
        return float(gamma.transpose() * np.linalg.inv(S) * gamma)

        ############
        # END student code
        ############

    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)

        # update associated tracks with measurements
        while (
            self.association_matrix.shape[0] > 0
            and self.association_matrix.shape[1] > 0
        ):

            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print("---no more associations---")
                break
            track = manager.track_list[ind_track]

            # check visibility, only update tracks in fov
            if not meas_list[0].sensor.in_fov(track.x):
                continue

            # Kalman update
            print(
                "update track",
                track.id,
                "with",
                meas_list[ind_meas].sensor.name,
                "measurement",
                ind_meas,
            )
            KF.update(track, meas_list[ind_meas])

            # update score and track state
            manager.handle_updated_track(track)

            # save updated track
            manager.track_list[ind_track] = track

        # run track management
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)

        for track in manager.track_list:
            print("track", track.id, "score =", track.score)
