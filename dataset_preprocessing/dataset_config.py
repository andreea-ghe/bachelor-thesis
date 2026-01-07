from easydict import EasyDict as edict

__C = edict()

dataset_cfg = __C

# Breaking Bad geometry assembly dataset configuration
# Used in Jigsaw paper experiments (Section 4.1)

__C.BREAKING_BAD = edict()

# Directory containing the Breaking Bad dataset files
__C.BREAKING_BAD.DATA_DIR = "C:\\Users\\AndreeaGheorghe\\ndrea"

# File naming pattern for train/val/test splits
# Will be formatted as "everyday.train.txt", "everyday.val.txt", etc.
__C.BREAKING_BAD.DATA_FN = (
    "everyday.{}.txt"
)

# Additional data keys to load with each sample (e.g., part_ids for tracking pieces)
__C.BREAKING_BAD.DATA_KEYS = ("part_ids",)

# Dataset subset selection (Paper Section 4.1: "everyday" and "artifact" subsets)
# Empty string means no specific subset filtering
__C.BREAKING_BAD.SUBSET = ""  # must in ['artifact', 'everyday', 'other']

# Category filtering - allows training/testing on specific object categories
# Empty string means all categories will be used
__C.BREAKING_BAD.CATEGORY = ""  # empty means all categories

# All available object categories in the everyday subset (Paper Table 1, Section 4.1)
# Breaking Bad dataset contains 20 everyday object categories
__C.BREAKING_BAD.ALL_CATEGORY = [
    "BeerBottle",
    "Bowl",
    "Cup",
    "DrinkingUtensil",
    "Mug",
    "Plate",
    "Spoon",
    "Teacup",
    "ToyFigure",
    "WineBottle",
    "Bottle",
    "Cookie",
    "DrinkBottle",
    "Mirror",
    "PillBottle",
    "Ring",
    "Statue",
    "Teapot",
    "Vase",
    "WineGlass",
]  # Only used for everyday

# Rotation range for data augmentation during training
# -1.0 means random full rotation (SO(3)), otherwise specifies degree range
__C.BREAKING_BAD.ROT_RANGE = -1.0

# Number of points to sample per object (not per part)
# Paper mentions "at least 30 points per fragment for multi-part assembly"
# Total points are distributed among parts based on their surface area
__C.BREAKING_BAD.NUM_PC_POINTS = 5000  # points per part

# Minimum number of points guaranteed for each piece when sampling by area
# Ensures even small fracture pieces have sufficient points for feature extraction ( > 30)
__C.BREAKING_BAD.MIN_PART_POINT = 30

# Range of number of parts per object (Paper Section 4.1, Table 1)
# Objects with fewer than 2 or more than 20 parts are filtered out
__C.BREAKING_BAD.MIN_NUM_PART = 2
__C.BREAKING_BAD.MAX_NUM_PART = 20

# Whether to shuffle the order of parts in each batch
# Useful for ensuring the model doesn't rely on part ordering
__C.BREAKING_BAD.SHUFFLE_PARTS = False

# Sampling strategy: "area" means points are sampled proportional to surface area
# This is more realistic than uniform point sampling across all pieces
__C.BREAKING_BAD.SAMPLE_BY = "area"

# Dataset length controls (-1 means use full dataset)
__C.BREAKING_BAD.LENGTH = -1
__C.BREAKING_BAD.TEST_LENGTH = -1
__C.BREAKING_BAD.OVERFIT = -1  # For debugging: use only N samples if > 0

# Threshold for determining fracture surface labels (in meters)
# Points within this distance to nearest neighbor in another piece are labeled as fracture points
__C.BREAKING_BAD.FRACTURE_LABEL_THRESHOLD = 0.025

# Color palette for visualizing different parts in point clouds
__C.BREAKING_BAD.COLORS = [
    [0, 204, 0],
    [204, 0, 0],
    [0, 0, 204],
    [127, 127, 0],
    [127, 0, 127],
    [0, 127, 127],
    [76, 153, 0],
    [153, 0, 76],
    [76, 0, 153],
    [153, 76, 0],
    [76, 0, 153],
    [153, 0, 76],
    [204, 51, 127],
    [204, 51, 127],
    [51, 204, 127],
    [51, 127, 204],
    [127, 51, 204],
    [127, 204, 51],
    [76, 76, 178],
    [76, 178, 76],
    [178, 76, 76],
]
