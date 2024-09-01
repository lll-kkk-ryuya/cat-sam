import json
import os
from os.path import join
from cat_sam.datasets.base import BinaryCATSAMDataset
import random

class CustomDataset(BinaryCATSAMDataset):
    def __init__(
            self,
            data_dir: str,
            train_flag: bool,
            shot_num: int = None,
            label_threshold: int = 254,
            object_connectivity: int = 8,
            area_threshold: int = 20,
            relative_threshold: bool = True,
            relative_threshold_ratio: float = 0.001,
            ann_scale_factor: int = 1,
            noisy_mask_threshold: float = 0.0,
            **super_args
    ):
        json_path = join(data_dir, 'train.json' if train_flag else 'test.json')
        with open(json_path, 'r') as j_f:
            json_config = json.load(j_f)
        
        # Update image and mask paths
        for key in json_config.keys():
            json_config[key]['image_path'] = join(data_dir, json_config[key]['image_path'])
            json_config[key]['mask_path'] = join(data_dir, json_config[key]['mask_path'])

        # Handle few-shot learning
        if shot_num is not None:
            assert 1 <= shot_num <= 16, f"Invalid shot_num: {shot_num}! Must be between 1 and 16!"
            all_keys = list(json_config.keys())
            selected_keys = random.sample(all_keys, min(shot_num, len(all_keys)))
            json_config = {key: json_config[key] for key in selected_keys}

        super(CustomDataset, self).__init__(
            dataset_config=json_config, train_flag=train_flag,
            label_threshold=label_threshold, object_connectivity=object_connectivity,
            area_threshold=area_threshold, relative_threshold=relative_threshold,
            relative_threshold_ratio=relative_threshold_ratio,
            ann_scale_factor=ann_scale_factor, noisy_mask_threshold=noisy_mask_threshold,
            **super_args
        )

# Function to get dataset parameters from command line arguments
def get_dataset_params():
    import argparse
    parser = argparse.ArgumentParser(description='Custom Dataset Parameters')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--label_threshold', type=int, default=254)
    parser.add_argument('--object_connectivity', type=int, default=8)
    parser.add_argument('--area_threshold', type=int, default=20)
    parser.add_argument('--relative_threshold', type=bool, default=True)
    parser.add_argument('--relative_threshold_ratio', type=float, default=0.001)
    parser.add_argument('--ann_scale_factor', type=int, default=1)
    parser.add_argument('--noisy_mask_threshold', type=float, default=0.0)
    return parser.parse_args()