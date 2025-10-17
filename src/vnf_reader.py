"""
author: Antonyo Musabini

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import argparse
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from pypcd4 import PointCloud

from typing import List, Dict, Tuple, Optional, Sequence, Any


SCANNER_LIST = [1, 2, 3, 4]

CAM_NAME_DICT = {
    'FV': 'Front',
    'RV': 'Rear',
    'MVL': 'Left',
    'MVR': 'Right'
}

class VNF_Reader:
    def __init__(self,
                 rec_folder: Path,
                 cam_side_names: Optional[list] = list(CAM_NAME_DICT.keys()),
                 scanner_list: Optional[list] = SCANNER_LIST,
                 ):
        """
        Initializes the VNF_Reader for a specific recording folder.

        Args:
            rec_folder: Path to the recording folder.
            cam_side_names: A list of camera side names to load (e.g., ['FV', 'RV']).
                            Defaults to all cameras.
            scanner_list: A list of scanner IDs to load. Defaults to all scanners.
        """

        self.rec_folder = Path(rec_folder)
        if not self.rec_folder.is_dir():
            raise FileNotFoundError(f"Recording folder not found: {self.rec_folder}")

        self.frame_readers = {}
        for _side in cam_side_names:
            video_path_files = list(self.rec_folder.glob(f"*{_side}.mp4"))
            video_path = self._check_len(video_path_files)
            if video_path:
                self.frame_readers[_side] = cv2.VideoCapture(str(video_path))

        master_synchronization_path = list(self.rec_folder.glob("*_master_synchronization.csv"))
        ts_path = self._check_len(master_synchronization_path)
        if ts_path:
            self.df_timestamp = pd.read_csv(ts_path, sep=",", index_col=0)

        self.dict_scanner = dict()
        self.scanner_list = scanner_list
        pcd_path = list(self.rec_folder.glob("*_pointcloud.pcd"))
        if len(pcd_path) == 0:
            print("No PCD file found.")
        elif len(pcd_path) != 1:
            print("Multiple PCD file found.")
        else:
            self._load_scanners(str(pcd_path[0]))

    @staticmethod
    def _check_len(path_to_check):
        """
        Checks if a list of paths contains exactly one item.

        Args:
            path_to_check: A list of file paths.

        Returns:
            The path if the list contains one item, otherwise None.
        """
        if len(path_to_check) > 1:
            print("Multiple rec file in folder. Aborting")
            return None
        elif len(path_to_check) == 0:
            print("No rec file in folder. Aborting")
            return None
        return path_to_check[0]

    def get_video_sizes(self) -> List[Tuple[float, float]]:
        """
        Gets the (width, height) for each loaded video stream.

        Returns:
            A list of tuples, where each tuple is the (width, height) of a camera.
        """
        sizes = []
        for _, reader in self.frame_readers.items():
            width = reader.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
            sizes.append((width, height))
        return sizes

    def get_mean_fps(self) -> int:
        """
        Calculates the mean frames per second based on timestamps.

        Returns:
            The rounded mean FPS value.
        """
        mean_frame_time = ((self.df_timestamp['master_chunktime'].max() -
                            self.df_timestamp['master_chunktime'].min()) /
                            self.df_timestamp['FV_frame'].max())
        return round(1e6 / mean_frame_time)

    def get_cameras_names(self, get_full_namers=True) -> List[str]:
        """
        Returns the list of camera names.

        Args:
            get_full_namers: If True, returns full names (e.g., 'Front').
                             If False, returns short names (e.g., 'FV').

        Returns:
            A list of camera names.
        """
        if not get_full_namers:
            return self.frame_readers.keys()
        return [CAM_NAME_DICT[short_name] for short_name in self.frame_readers.keys()]

    def get_next_frames(self) -> Tuple[bool, List[Optional[np.ndarray]], List[Any], List[str], List[Optional[int]]]:
        """
        Reads the next synchronized frame from all camera streams.

        Returns:
            A tuple containing:
            - ret (bool): False if frames could not be read, True otherwise.
            - frames (List[Optional[np.ndarray]]): A list of image frames.
            - current_ts (List[Any]): A list of timestamps for each frame.
            - cam_name_list (List[str]): A list of camera names.
            - frame_idx_list (List[Optional[int]]): A list of frame indices.
        """
        frames = [None] * len(self.frame_readers)
        current_ts = [None] * len(self.frame_readers)
        cam_name_list = [None] * len(self.frame_readers)
        frame_idx_list = [None] * len(self.frame_readers)

        for idx, cam_name in enumerate(self.frame_readers.keys()):
            cam_name_list[idx] = cam_name
            # Capture frame-by-frame
            ret, frames[idx] = self.frame_readers[cam_name].read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame for cam %s. Exiting ..." % cam_name)
                break
            frame_idx = (
                int(self.frame_readers[cam_name].get(cv2.CAP_PROP_POS_FRAMES)) - 1
            )
            if frame_idx < len(self.df_timestamp["master_UTC"]):
                current_ts[idx] = self.df_timestamp["master_UTC"][frame_idx]
                frame_idx_list[idx] = frame_idx

            else:
                # Error while reading timestamp, ignore frame
                ret = False
                break
        return ret, frames, current_ts, cam_name_list, frame_idx_list

    def get_scan_dict_for_cam_frame(self, camera_frame: int = 0, ts_row: Optional[pd.Series] = None) -> Dict[int, pd.DataFrame]:
        """
        Gets a dictionary of scanner point clouds for a given camera frame index.

        Args:
            camera_frame: The index of the camera frame.
            ts_row: A pre-fetched row from the timestamp DataFrame. If None, it will be
                    fetched using the camera_frame index.

        Returns:
            A dictionary where keys are scanner IDs and values are DataFrames of points.
        """
        dict_scan = dict()
        if ts_row == None:
            ts_row = self.df_timestamp.loc[camera_frame]
        for scanner_id in self.scanner_list:
            n_scanner_frame = ts_row["LR_" + str(scanner_id) + "_frame"]
            dict_scan[scanner_id] = self.dict_scanner[scanner_id].loc[self.dict_scanner[scanner_id]['scan_id'].values == n_scanner_frame]
        return dict_scan

    def _load_scanners(self, pcd_path: str):
        """
        Loads and preprocesses scanner data from a PCD file.

        The point cloud data is loaded into a DataFrame and then split into a
        dictionary of DataFrames, one for each scanner ID.

        Args:
            pcd_path: The file path to the .pcd file.
        """
        pc: PointCloud = PointCloud.from_path(pcd_path)
        df: pd.DataFrame = pd.DataFrame(pc.numpy(), columns=pc.fields)

        for scanner_id in self.scanner_list:
            self.dict_scanner[scanner_id] = df.loc[df["scanner_id"].values == float(scanner_id)]

    def get_scanner_dict(self):
        """
        Returns the dictionary of all loaded scanner data.

        Returns:
            A dictionary where keys are scanner IDs and values are DataFrames
            containing all points for that scanner.
        """
        return self.dict_scanner

    def stack_points(self, dict_scan: pd.DataFrame):
        """
        Stacks points from a dictionary of scans into single numpy arrays.

        Args:
            dict_scan: A dictionary where keys are scanner IDs and values are
                       DataFrames containing the point cloud for a single frame.

        Returns:
            A tuple of numpy arrays (x, y, z, intensity).
        """
        x, y, z, i = [], [], [], []
        for scanner_id in self.scanner_list:
            x = np.hstack([x, dict_scan[scanner_id]['x'].values])
            y = np.hstack([y, dict_scan[scanner_id]['y'].values])
            z = np.hstack([z, dict_scan[scanner_id]['z'].values])
            i = np.hstack([i, dict_scan[scanner_id]['intensity'].values])

        return x, y, z, i

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Valeo Near-Field (VNF) dataset reader."
    )
    parser.add_argument(
        "rec_folder",
        type=str,
        help="Path to the recording folder.",
    )
    parser.add_argument(
        "--cam_side_names",
        nargs='+',
        default=list(CAM_NAME_DICT.keys()),
        choices=list(CAM_NAME_DICT.keys()),
        help=f"List of camera sides to process. Defaults to all: {list(CAM_NAME_DICT.keys())}."
    )
    parser.add_argument(
        "--scanner_list",
        nargs='+',
        type=int,
        default=SCANNER_LIST,
        choices=SCANNER_LIST,
        help=f"List of scanner IDs to process. Defaults to all: {SCANNER_LIST}."
    )
    args = parser.parse_args()

    data_reader = VNF_Reader(
        rec_folder=args.rec_folder,
        cam_side_names=args.cam_side_names,
        scanner_list=args.scanner_list
    )

    print(f"Mean FPS: {data_reader.get_mean_fps()}")
    print(f"Camera names: {data_reader.get_cameras_names()}")

    while True:
        ret, frames, current_ts, cam_name_list, frame_idx_list = data_reader.get_next_frames()
        if not ret:
            break

        if frame_idx_list and frame_idx_list[0] is not None:
            dict_scan = data_reader.get_scan_dict_for_cam_frame(frame_idx_list[0])
            x, y, z, i = data_reader.stack_points(dict_scan)
            print(f"Frame {frame_idx_list[0]}: Found {len(x)} points.")
