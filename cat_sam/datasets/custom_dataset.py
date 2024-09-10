import json
from os.path import join
from typing import List

from cat_sam.datasets.base import BinaryCATSAMDataset

class CustomDataset(BinaryCATSAMDataset):
    def __init__(
            self,
            data_dir: str,
            json_config: dict,
            train_flag: bool,
            shot_num: int = None,
            label_threshold: int = 254,
            object_connectivity: int = 8,
            area_threshold: int = 20,
            relative_threshold: bool = True,
            ann_scale_factor: float = 1.0,
            noisy_mask_threshold: float = 0.0,
            class_names: List[str] = None,
            **super_args
    ):
        # JSONコンフィグを直接使用
        dataset_config = json_config

        if shot_num is not None:
            dataset_config = self._sample_few_shot(dataset_config, shot_num)

        # クラス名が指定されていない場合はデフォルト値を使用
        self.class_names = class_names if class_names is not None else ['Background', 'Foreground']

        super(CustomDataset, self).__init__(
            dataset_config=dataset_config,
            train_flag=train_flag,
            label_threshold=label_threshold,
            object_connectivity=object_connectivity,
            area_threshold=area_threshold,
            relative_threshold=relative_threshold,
            ann_scale_factor=ann_scale_factor,
            noisy_mask_threshold=noisy_mask_threshold,
            **super_args
        )

    def _sample_few_shot(self, dataset_config, shot_num):
        # Few-shotサンプリングのロジックをここに実装
        import random
        keys = list(dataset_config.keys())
        sampled_keys = random.sample(keys, min(shot_num, len(keys)))
        return {key: dataset_config[key] for key in sampled_keys}
