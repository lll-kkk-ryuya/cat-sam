import json
from os.path import join

from cat_sam.datasets.base import BinaryCATSAMDataset

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
            ann_scale_factor: float = 1.0,
            noisy_mask_threshold: float = 0.0,
            **super_args
    ):
        json_path = join(data_dir, 'train.json' if train_flag else 'test.json')
        with open(json_path, 'r') as j_f:
            json_config = json.load(j_f)
        
        for key in json_config.keys():
            json_config[key]['image_path'] = join(data_dir, json_config[key]['image_path'])
            json_config[key]['mask_path'] = join(data_dir, json_config[key]['mask_path'])

        if shot_num is not None:
            json_config = self._sample_few_shot(json_config, shot_num)

        super(CustomDataset, self).__init__(
            dataset_config=json_config,
            train_flag=train_flag,
            label_threshold=label_threshold,
            object_connectivity=object_connectivity,
            area_threshold=area_threshold,
            relative_threshold=relative_threshold,
            ann_scale_factor=ann_scale_factor,
            noisy_mask_threshold=noisy_mask_threshold,
            **super_args
        )

    def _sample_few_shot(self, json_config, shot_num):
        # Few-shotサンプリングのロジックをここに実装
        import random
        keys = list(json_config.keys())
        sampled_keys = random.sample(keys, min(shot_num, len(keys)))
        return {key: json_config[key] for key in sampled_keys}
