import os
import random
import trimesh
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R



class FractureAssemblyDataset(Dataset):
    """
    Custom dataset loader designed for multi-part fracture assembly: it handles
    variable nr of pieces per object and uses area-based point sampling.
    """
    def __init__(self, dataset_dir, split, additional_data, num_points=1000, min_num_points=30, min_parts=2, max_parts=20, rot_range=-1, shuffle_parts=False, overfit=-1, length=-1, fracture_label_threshold=0.025):
        """
        Input:
            dataset_dir: path to the dataset directory
            split: 'train', 'val' or 'test' split
            additional_data: additional data keys to load
            num_points: number of points to sample per object
            min_num_points_part: minimum number of points per piece
            min_parts: minimum number of pieces per object
            max_parts: maximum number of pieces per object
            rot_range: rotation augmentation range
            shuffle_parts: whether to shuffle the order of parts
            overfit: if >0, limits dataset size for overfitting tests
            length: if >0, sets fixed dataset length
            fracture_label_threshold: threshold for fracture labeling
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.additional_data = additional_data
        self.num_points = num_points
        self.min_num_points = min_num_points
        self.min_parts = min_parts
        self.max_parts = max_parts # ignore shapes with more parts
        self.rot_range = rot_range # rotation range in degree
        self.shuffle_parts = shuffle_parts
        self.overfit = overfit
        self.fracture_label_threshold = fracture_label_threshold # threshold for determining if a point it on the fracture surface

        # list of fracture folder path
        self.data_list = self._read_data(split)
        print(f"Dataset length: {len(self.data_list)}")

        if self.overfit > 0:
            # restrict dataset size for overfitting
            self.data_list = self.data_list[:self.overfit]

        if 0 < length < len(self.data_list):
            # if we need to contract the dataset, we are doing random sampling
            self.length = length
            if self.shuffle_parts:
                print("Shuffling dataset indices.")
                pos = list(range(len(self.data_list)))
                random.shuffle(pos)
                self.data_list = [self.data_list[i] for i in pos]
        else:
            self.length = len(self.data_list)

    def _read_data(self, split):
        """
        Read the dataset file list and filter based on number of parts.
        It shall consider all categories.
        """
        # load pre-generated data_list if it exists
        pre_computed_filename = f"fracture_assembly_metadata_{self.min_parts}_{self.max_parts}_" + split
        if os.path.exists(os.path.join(self.dataset_dir, pre_computed_filename)):
            with open(os.path.join(self.dataset_dir, pre_computed_filename), 'rb') as meta_table:
                meta_dict = pickle.load(meta_table)
                data_list = meta_dict['data_list']
            return data_list

        # read the file: each line contains a relative path to a mesh folder
        with open(os.path.join(self.dataset_dir, split), 'r') as f:
            mesh_list = [line.strip() for line in f.readlines()]

        data_list = []
        for mesh in mesh_list:
            mesh_dir = os.path.join(self.dataset_dir, mesh)
            if not os.path.isdir(mesh_dir):
                print(f"Warning: {mesh_dir} is not a valid directory, skipping.")
                continue


            fractures = os.listdir(mesh_dir)
            fractures.sort()

            for fracture in fractures:
                if "fractured" not in fracture and "mode" not in fracture:
                    continue
                fracture = os.path.join(mesh_dir, fracture)
                pieces = os.listdir(os.path.join(self.dataset_dir, fracture))

                # filter based on number of parts
                if self.min_parts <= len(pieces) <= self.max_parts:
                    data_list.append(fracture)

        meta_dict = {
            'data_list': data_list,
        }
        # save pre-computed metadata for future use
        with open(os.path.join(self.dataset_dir, pre_computed_filename), 'wb') as meta_table:
            pickle.dump(meta_dict, meta_table, protocol=pickle.HIGHEST_PROTOCOL)

        return data_list

    def __len__(self):
        """
        Return the length of the dataset, how many samples we are allowed to query from.
        """
        return self.length

    @staticmethod
    def _recenter_point_cloud(point_cloud):
        """
        Center point cloud at the origin by subtracting the centroid.
        The centroid is returned to track the translation applied to each piece.

        Input:
            point_cloud: (N, 3) numpy array of point cloud

        Output:
            centered_point_cloud: (N, 3) numpy array of centered point cloud
            centroid: (3,) numpy array of the centroid
        """
        centroid = np.mean(point_cloud, axis=0)
        centered_point_cloud = point_cloud - centroid
        return centered_point_cloud, centroid

    def _rotate_point_cloud(self, point_cloud):
        """
        Apply random rotation augmentation to the point cloud within the specified range.
        Each piece is randomly rotated to simulate arbitrary initial orientations.
        This is the data augmentation strategy described in the paper for training
        the model to handle pieces in arbitrary poses.

        Input:
            point_cloud: (N, 3) numpy array of point cloud

        Output:
            rotated_point_cloud: (N, 3) numpy array of rotated point cloud
            quat_gt: (4,) numpy array of ground truth quaternion representing the rotation applied
        """
        if self.rot_range > 0: 
            # gradually increase rotation range during training
            # in early epochs, we have limited rotation range for easier learning
            rot_euler = (np.random.rand(3) - 0.5) * 2.0 * self.rot_range
            rot_mat = R.from_euler('xyz', rot_euler, degrees=True).as_matrix()
        else:
            # full rotation used for final training stages
            rot_mat = R.random().as_matrix()

        # apply rotation to point cloud
        point_cloud = (rot_mat @ point_cloud.T).T

        # store inverse rotation as ground truth
        # it will help to recover original poses
        quat_gt = R.from_matrix(rot_mat.T).as_quat()

        # convert to scalar-first quat format (w, x, y, z)
        # because this is the format used throughout the codebase
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return point_cloud, quat_gt

    @staticmethod
    def _shuffle_point_cloud(point_cloud, gt_point_cloud):
        """
        Shuffle the order of points in the point cloud.
        This ensures that the model does not rely on any implicit ordering of points
        and its permutation invariant.

        Input:
            point_cloud: (N, 3) numpy array of point cloud
            gt_point_cloud: (N, 3) untransformed point cloud

        Output:
            shuffled_point_cloud: (N, 3) numpy array of shuffled point cloud
            shuffled_gt_point_cloud: (N, 3) numpy array of shuffled ground truth point cloud
        """
        order = np.arange(point_cloud.shape[0])
        random.shuffle(order)

        point_cloud = point_cloud[order]
        point_cloud_gt = gt_point_cloud[order]

        return point_cloud, point_cloud_gt

    def _pad_data(self, data, padded_size=None):
        """
        Pad data to shape [self.max_parts, data.shape[1], ...].
        Since objects have varying numbers of pieces (2-20), we pad all data to 
        a fixed size (max_parts) for batching. The valid pieces are tracked
        using the valids mask.

        Input:
            data: data array to pad (quaternions, translations ...)
            pad_size: target size (default: self.max_parts)

        Output:
            padded_data: zero padded data array
        """
        if padded_size is None:
            padded_size = self.max_parts # default

        data = np.array(data)
        if len(data.shape) > 1: # check if data is multi-dimensional
            # drop the first dimension
            # create the final padded shape
            padded_shape = (padded_size,) + tuple(data.shape[1:])
        else:
            padded_shape = (padded_size,)

        # allocate a zero-filled array
        padded_data = np.zeros(padded_shape, dtype=data.dtype)
        padded_data[:data.shape[0]] = data # copy data into padded array
        return padded_data

    @staticmethod
    def sample_points_by_area(areas, total_points):
        """
        Distribute points among pieces proportional to their surface areas.
        It simulates how a 3D scanner would capture more points from larger surfaces.

        Input:
            areas: surface area for each piece
            total_points: total number of points to distribute

        Output:
            nr_points_per_piece: list of number of points assigned to each piece
        """
        total_area = np.sum(areas)

        # proportionally allocate points based on area
        nr_points_per_piece = np.ceil(areas * total_points / total_area).astype(np.int32)

        # adjust largest piece to ensure exact total point count
        diff = np.sum(nr_points_per_piece) - total_points
        nr_points_per_piece[np.argmax(nr_points_per_piece)] -= diff

        return np.array(nr_points_per_piece, dtype=np.int64)

    def sample_reweighted_points_by_areas(self, areas):
        """
        Sample points by areas, ensuring each part has at least min_num_points points.
        Steps:
        1. First distribute points proportionally by area.
        2. Ensure every piece has at least min_num_points points.
        3. Take extra points from the largest pieces to maintain total count.

        This prevents tiny fracture pieces from having too few points for  reliable feature
        extraction, while maintaining realistic area-based distribution.

        Input:
            areas: list of areas for each piece

        Returns:
            nr_points_per_piece: list of number of points assigned to each piece
        """
        nr_points_per_piece = self.sample_points_by_area(areas, self.num_points)
        if self.min_num_points <= 1:
            return nr_points_per_piece

        # count how many extra points we need to add to small pieces
        delta = 0
        for i in range(len(nr_points_per_piece)):
            if nr_points_per_piece[i] < self.min_num_points:
                delta += self.min_num_points - nr_points_per_piece[i]
                nr_points_per_piece[i] = self.min_num_points

        # take points from the largest piece to maintain total point count
        while delta > 0:
            k = np.argmax(nr_points_per_piece) # largest piece index
            if nr_points_per_piece[k] - delta >= self.min_num_points:
                nr_points_per_piece[k] -= delta
                delta = 0
            else:
                # we reduce the largest piece to min_num_points and try again with the next largest piece
                delta -= nr_points_per_piece[k] - self.min_num_points 
                nr_points_per_piece[k] = self.min_num_points


        return np.array(nr_points_per_piece, dtype=np.int64)

    def _load_point_clouds(self, data_folder):
        """
        Read mesh files and sample point clouds from a folder.
        Steps:
        1. Load all mesh pieces from the object folder.
        2. Compute surface area of each piece.
        3. Samples point proportionally to surface area.
        4. Tracks which piece each point belongs to.

        Input:
            data_folder: path to the folder containing mesh pieces

        Output:
            point_clouds: list of (N_i, 3) numpy arrays of sampled points
            piece_id: (N, 1) numpy array indicating which piece each point belongs to
            nr_points_per_piece: list of number of points assigned to each piece
            areas: list of surface areas for each piece
        """
        data_folder = os.path.join(self.dataset_dir, data_folder)
        
        mesh_files = os.listdir(data_folder) # list all mesh files from the folder
        mesh_files.sort()
        if not self.min_parts <= len(mesh_files) <= self.max_parts:
            raise ValueError(f"Number of parts {len(mesh_files)} not in range [{self.min_parts}, {self.max_parts}]")

        # load all mesh pieces
        meshes = []
        for mesh_file in mesh_files:
            mesh = trimesh.load(os.path.join(data_folder, mesh_file), force='mesh')
            if isinstance(mesh, trimesh.Scene): # handle case where trimesh returns a Scene instead of a Trimesh
                mesh = trimesh.util.concatenate(list(mesh.geometry.values())) # concatenate all geometries in the scene into one mesh
            meshes.append(mesh)

        point_clouds = [] 
        piece_id = [] 
        nr_points_per_piece = [] 

        # compute surface areas
        areas = [mesh.area for mesh in meshes]
        areas = np.array(areas)
        nr_points_per_piece = self.sample_reweighted_points_by_areas(areas)

        for i, (mesh) in enumerate(meshes):
            # sample points from each mesh
            num_points = nr_points_per_piece[i]
            sampled_points, _ = mesh.sample(num_points, return_index=True)
            point_clouds.append(sampled_points)
            piece_id.append([i] * num_points) # track which piece each point belongs to

        piece_id = np.concatenate(piece_id).astype(np.int64).reshape((-1, 1))
        return point_clouds, piece_id, nr_points_per_piece, areas

    def __getitem__(self, index):
        """
        Get a single data sample.
        Complete data loading pipeline:
        1. Load and sample points from all mesh pieces.
        2. Apply transformations (center, rotation) to each piece.
        3. Shuffle point orders for permutation invariance.
        4. Prepare ground truth labels and metadata.

        The data preparation simulates the fracture assembly problem where each piece
        is in an arbitrary pose and must be aligned to reconstruct the object.

        Input:
            index: index of the data sample to retrieve

        Output:

        """
        point_clouds, piece_ids, nr_points_per_piece, areas = self._load_point_clouds(self.data_list[index])
        num_parts = len(point_clouds)

        assembled_pcs = [] # list of transformed point clouds for each piece
        gt_assembled_pcs = [] # list of ground truth point clouds for each piece
        gt_translations = [] # list of ground truth translations for each piece
        gt_rotations = [] # list of ground truth rotations for each piece

        # process each piece: center, rotate and shuffle
        for point_cloud, nr_points in zip(point_clouds, nr_points_per_piece):
            # keep original point cloud as ground truth (for segmentation labels)
            gt_point_cloud = point_cloud.copy()

            # center the piece at origin
            point_cloud, gt_trans = self._recenter_point_cloud(point_cloud)

            # apply random rotation augmentation
            point_cloud, gt_quat = self._rotate_point_cloud(point_cloud)

            # shuffle point order
            point_cloud, gt_shuffle = self._shuffle_point_cloud(point_cloud, gt_point_cloud)

            assembled_pcs.append(point_cloud)
            gt_assembled_pcs.append(gt_shuffle)
            gt_translations.append(gt_trans)
            gt_rotations.append(gt_quat)

        # concatenate all pieces into single point cloud with piece ids tracked for each point
        assembled_pcs = np.concatenate(assembled_pcs).astype(np.float32) # [N_total, 3]
        gt_assembled_pcs = np.concatenate(gt_assembled_pcs).astype(np.float32) # [N_total, 3]
        
        gt_translations = self._pad_data(np.stack(gt_translations, axis=0), self.max_parts).astype(np.float32) # [max_parts, 3]
        gt_rotations = self._pad_data(np.stack(gt_rotations, axis=0), self.max_parts).astype(np.float32) # [max_parts, 4]
        points_per_part = self._pad_data(np.array(nr_points_per_piece), self.max_parts).astype(np.int64) # [max_parts]

        # validity mask: 1 for valid pieces, 0 for padded pieces
        valids_mask = np.zeros(self.max_parts, dtype=np.float32)
        valids_mask[:num_parts] = 1.0

        # distance threshold for fracture surface labeling: every sampled point gets the same distance threshold
        label_thresholds = np.ones([self.num_points], dtype=np.float32) * self.fracture_label_threshold

        data_dict = {
            "part_pcs": assembled_pcs,
            "gt_pcs": gt_assembled_pcs,
            "part_valids": valids_mask,
            "part_quat": gt_rotations,
            "part_trans": gt_translations,
            "n_pcs": points_per_part,
            "data_id": index,
            "critical_label_thresholds": label_thresholds,
        }

        return data_dict

def build_data_loaders(config):
    """
    Build train and validation dataloader for FractureAssemblyDataset.

    Input:
        config: EasyDict configuration containing dataset and training parameters
    """
    data_dict = {
        'dataset_dir': config.DATA.DATA_DIR,
        'split': config.DATA.DATA_FN.format('train'),
        'additional_data': config.DATA.DATA_KEYS,
        'num_points': config.DATA.NUM_PC_POINTS,
        'min_num_points': config.DATA.MIN_PART_POINT,
        'min_parts': config.DATA.MIN_NUM_PART,
        'max_parts': config.DATA.MAX_NUM_PART,
        'rot_range': config.DATA.ROT_RANGE,
        'shuffle_parts': config.DATA.SHUFFLE_PARTS,
        'overfit': config.DATA.OVERFIT,
        'length': config.DATA.LENGTH,
        'fracture_label_threshold': config.DATA.FRACTURE_LABEL_THRESHOLD,
    }

    train_dataset = FractureAssemblyDataset(**data_dict)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # shuffle for training
        num_workers=config.NUM_WORKERS,
        pin_memory=True,  # faster GPU transfer
        drop_last=True,  # drop last incomplete batch to ensure consistent batch size
        persistent_workers=(config.NUM_WORKERS > 0),  # keep workers alive between epochs
    )

    # create validation dataset and loader
    data_dict['split'] = config.DATA.DATA_FN.format('val')
    data_dict['shuffle_parts'] = False  # no shuffling during validation
    data_dict['length'] = config.DATA.TEST_LENGTH

    val_dataset = FractureAssemblyDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE * 2,  # can use larger batch for validation
        shuffle=False,  # no shuffling for validation, we need consistent validation order
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,  # do not drop last batch for validation, we need all validation samples
        persistent_workers=(config.NUM_WORKERS > 0),
    )
    return train_loader, val_loader