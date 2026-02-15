import os
import random
import csv
import trimesh
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R


class FracturePairsDataset(Dataset):
    """
    Dataset loader for flat fracture pairs format (artifact_all_2 / everyday_all_2).
    
    Data structure:
        - All .obj files in a flat directory
        - CSV file with pairs: pc1,pc2 columns matching piece_0 and piece_1
        - Always exactly 2 pieces per pair
    
    Requires a pre-split CSV file (e.g. everyday.train.txt created by
    create_pair_splits.py) with pc1,pc2 headers. All pairs are loaded directly.
    """
    def __init__(
        self,
        dataset_dir,
        split_file,
        split='train',
        num_points=5000,
        rot_range=-1,
        overfit=-1,
        length=-1,
        fracture_label_threshold=0.025,
        seed=42,
    ):
        """
        Input:
            dataset_dir: path to directory containing .obj files
            split_file: path to a pre-split CSV file (e.g. everyday.train.txt)
                        with pc1,pc2 headers (created by create_pair_splits.py)
            split: 'train', 'val', or 'test' (used for logging only)
            num_points: total number of points to sample per pair
            rot_range: rotation augmentation range (-1 for full SO(3))
            overfit: if >0, limits dataset size for overfitting tests
            length: if >0, sets fixed dataset length
            fracture_label_threshold: threshold for fracture labeling
            seed: random seed
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.num_points = num_points
        self.rot_range = rot_range
        self.overfit = overfit
        self.fracture_label_threshold = fracture_label_threshold
        
        # Always 2 pieces for this dataset format
        self.max_parts = 2
        self.min_num_points = 30  # minimum points per piece
        
        # Load pairs from the split file
        self.pairs = self._load_from_split_file(split_file)
        print(f"[{split}] Loaded {len(self.pairs)} pairs from {split_file}")
        
        if self.overfit > 0:
            self.pairs = self.pairs[:self.overfit]
        
        if 0 < length < len(self.pairs):
            self.length = length
            random.seed(seed)
            indices = list(range(len(self.pairs)))
            random.shuffle(indices)
            self.pairs = [self.pairs[i] for i in indices[:length]]
        else:
            self.length = len(self.pairs)
    
    @staticmethod
    def _load_from_split_file(split_file):
        """
        Load all pairs directly from a pre-split CSV file.
        The file should have pc1,pc2 headers (created by create_pair_splits.py).
        """
        pairs = []
        with open(split_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((row['pc1'].strip(), row['pc2'].strip()))
        return pairs
    
    def __len__(self):
        return self.length
    
    @staticmethod
    def _recenter_point_cloud(point_cloud):
        """Center point cloud at origin, return centered cloud and centroid."""
        centroid = np.mean(point_cloud, axis=0)
        centered = point_cloud - centroid
        return centered, centroid
    
    def _rotate_point_cloud(self, point_cloud):
        """Apply random rotation augmentation."""
        if self.rot_range > 0:
            rot_euler = (np.random.rand(3) - 0.5) * 2.0 * self.rot_range
            rot_mat = R.from_euler('xyz', rot_euler, degrees=True).as_matrix()
        else:
            rot_mat = R.random().as_matrix()
        
        point_cloud = (rot_mat @ point_cloud.T).T
        
        # Inverse rotation as ground truth (scalar-first quaternion: w, x, y, z)
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        quat_gt = quat_gt[[3, 0, 1, 2]]
        
        return point_cloud, quat_gt
    
    @staticmethod
    def _shuffle_point_cloud(point_cloud, gt_point_cloud):
        """Shuffle point order for permutation invariance."""
        order = np.arange(point_cloud.shape[0])
        np.random.shuffle(order)
        return point_cloud[order], gt_point_cloud[order]
    
    def _sample_points_by_area(self, areas):
        """Distribute points among pieces proportional to surface areas."""
        total_area = np.sum(areas)
        nr_points = np.ceil(areas * self.num_points / total_area).astype(np.int32)
        
        # Adjust to exact total
        diff = np.sum(nr_points) - self.num_points
        nr_points[np.argmax(nr_points)] -= diff
        
        # Ensure minimum points per piece
        for i in range(len(nr_points)):
            if nr_points[i] < self.min_num_points:
                delta = self.min_num_points - nr_points[i]
                nr_points[i] = self.min_num_points
                # Take from largest piece
                largest = np.argmax(nr_points)
                if largest != i:
                    nr_points[largest] -= delta
        
        return nr_points.astype(np.int64)
    
    def _load_mesh(self, filename):
        """Load a mesh file and handle edge cases."""
        mesh_path = os.path.join(self.dataset_dir, filename)
        mesh = trimesh.load(mesh_path, force='mesh')
        
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        if isinstance(mesh, list):
            if len(mesh) == 0:
                raise ValueError(f"Corrupted mesh file: {mesh_path}")
            mesh = trimesh.util.concatenate(mesh)
        
        return mesh
    
    def _load_pair(self, pc1_file, pc2_file):
        """
        Load a pair of mesh pieces and sample point clouds.
        
        Returns:
            point_clouds: list of (N_i, 3) arrays
            piece_ids: (N, 1) array of piece indices
            nr_points_per_piece: points per piece
            areas: surface areas
        """
        mesh1 = self._load_mesh(pc1_file)
        mesh2 = self._load_mesh(pc2_file)
        meshes = [mesh1, mesh2]
        
        # Compute areas and distribute points
        areas = np.array([m.area for m in meshes])
        nr_points_per_piece = self._sample_points_by_area(areas)
        
        point_clouds = []
        piece_ids = []
        
        for i, mesh in enumerate(meshes):
            n_pts = nr_points_per_piece[i]
            sampled_points, _ = mesh.sample(n_pts, return_index=True)
            point_clouds.append(sampled_points)
            piece_ids.append([i] * n_pts)
        
        piece_ids = np.concatenate(piece_ids).astype(np.int64).reshape(-1, 1)
        return point_clouds, piece_ids, nr_points_per_piece, areas
    
    def __getitem__(self, index):
        """
        Get a single pair sample.
        
        Returns data dict with same format as FractureAssemblyDataset.
        """
        pc1_file, pc2_file = self.pairs[index]
        
        try:
            point_clouds, piece_ids, nr_points_per_piece, areas = self._load_pair(pc1_file, pc2_file)
        except (ValueError, Exception) as e:
            print(f"Skipping corrupted sample {index}: {e}")
            return self.__getitem__((index + 1) % len(self.pairs))
        
        num_parts = 2
        
        assembled_pcs = []
        gt_assembled_pcs = []
        gt_translations = []
        gt_rotations = []
        
        for point_cloud, nr_points in zip(point_clouds, nr_points_per_piece):
            gt_point_cloud = point_cloud.copy()
            
            # Center at origin
            point_cloud, gt_trans = self._recenter_point_cloud(point_cloud)
            
            # Random rotation
            point_cloud, gt_quat = self._rotate_point_cloud(point_cloud)
            
            # Shuffle points
            point_cloud, gt_shuffle = self._shuffle_point_cloud(point_cloud, gt_point_cloud)
            
            assembled_pcs.append(point_cloud)
            gt_assembled_pcs.append(gt_shuffle)
            gt_translations.append(gt_trans)
            gt_rotations.append(gt_quat)
        
        # Concatenate all pieces
        assembled_pcs = np.concatenate(assembled_pcs).astype(np.float32)
        gt_assembled_pcs = np.concatenate(gt_assembled_pcs).astype(np.float32)
        
        gt_translations = np.stack(gt_translations, axis=0).astype(np.float32)
        gt_rotations = np.stack(gt_rotations, axis=0).astype(np.float32)
        points_per_part = np.array(nr_points_per_piece, dtype=np.int64)
        
        # Validity mask (always both pieces valid)
        valids_mask = np.ones(self.max_parts, dtype=np.float32)
        
        # Fracture label thresholds
        label_thresholds = np.ones(self.num_points, dtype=np.float32) * self.fracture_label_threshold
        
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


def _resolve_split_file(data_dir, data_fn, split_name):
    """
    Build the path to a split file (e.g. everyday.train.txt) in data_dir.
    Raises FileNotFoundError if the file doesn't exist.
    
    Input:
        data_dir: base data directory
        data_fn: filename pattern like 'everyday.{}.txt'
        split_name: 'train', 'val', or 'test'
    """
    split_filename = data_fn.format(split_name)
    split_path = os.path.join(data_dir, split_filename)
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Split file not found: {split_path}\n"
            f"Run create_pair_splits.py first to generate train/val/test splits."
        )
    return split_path


def build_pairs_data_loaders(config):
    """
    Build train and validation dataloaders for FracturePairsDataset.
    
    Expects pre-split files (e.g. everyday.train.txt, everyday.val.txt)
    in DATA.DATA_DIR, named according to DATA.DATA_FN pattern.
    
    Input:
        config: EasyDict configuration with:
            - DATA.DATA_DIR: directory containing .obj files and split files
            - DATA.DATA_FN: filename pattern (e.g. 'everyday.{}.txt')
            - DATA.NUM_PC_POINTS: total points per sample
            - DATA.ROT_RANGE: rotation augmentation range
            - DATA.OVERFIT: overfit sample count (-1 for all)
            - DATA.LENGTH: dataset length limit (-1 for all)
            - DATA.TEST_LENGTH: validation length limit
            - DATA.FRACTURE_LABEL_THRESHOLD: fracture labeling threshold
            - BATCH_SIZE: training batch size
            - NUM_WORKERS: dataloader workers
    """
    data_dir = config.DATA.DATA_DIR
    data_fn = config.DATA.DATA_FN

    train_split_file = _resolve_split_file(data_dir, data_fn, 'train')
    val_split_file = _resolve_split_file(data_dir, data_fn, 'val')

    train_dataset = FracturePairsDataset(
        dataset_dir=data_dir,
        split_file=train_split_file,
        split='train',
        num_points=config.DATA.NUM_PC_POINTS,
        rot_range=config.DATA.ROT_RANGE,
        overfit=config.DATA.OVERFIT,
        length=config.DATA.LENGTH,
        fracture_label_threshold=config.DATA.FRACTURE_LABEL_THRESHOLD,
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )
    
    val_dataset = FracturePairsDataset(
        dataset_dir=data_dir,
        split_file=val_split_file,
        split='val',
        num_points=config.DATA.NUM_PC_POINTS,
        rot_range=config.DATA.ROT_RANGE,
        overfit=-1,
        length=config.DATA.TEST_LENGTH,
        fracture_label_threshold=config.DATA.FRACTURE_LABEL_THRESHOLD,
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(config.NUM_WORKERS > 0),
    )
    
    return train_loader, val_loader


def build_pairs_test_loader(config):
    """
    Build test dataloader for FracturePairsDataset.
    
    Expects a pre-split test file (e.g. everyday.test.txt)
    in DATA.DATA_DIR, named according to DATA.DATA_FN pattern.
    """
    data_dir = config.DATA.DATA_DIR
    data_fn = config.DATA.DATA_FN

    test_split_file = _resolve_split_file(data_dir, data_fn, 'test')

    test_dataset = FracturePairsDataset(
        dataset_dir=data_dir,
        split_file=test_split_file,
        split='test',
        num_points=config.DATA.NUM_PC_POINTS,
        rot_range=config.DATA.ROT_RANGE,
        overfit=-1,
        length=config.DATA.TEST_LENGTH,
        fracture_label_threshold=config.DATA.FRACTURE_LABEL_THRESHOLD,
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(config.NUM_WORKERS > 0),
    )
    
    return test_loader
