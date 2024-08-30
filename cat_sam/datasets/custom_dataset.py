import json
from os.path import join
from cat_sam.datasets.base import BinaryCATSAMDataset

few_shot_img_dict = {
    1: ['image1'],
    16: ['image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'image8', 'image9',
         'image10', 'image11', 'image12', 'image13', 'image14', 'image15', 'image16', 'image17']
}

class CustamDataset(BinaryCATSAMDataset):
    
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
        for key in json_config.keys():
            json_config[key]['image_path'] = join(data_dir, json_config[key]['image_path'])
            json_config[key]['mask_path'] = join(data_dir, json_config[key]['mask_path'])

        if shot_num is not None:
            assert shot_num in [1, 16], f"Invalid shot_num: {shot_num}! Must be either 1 or 16!"
            json_config = {key: value for key, value in json_config.items() if key in few_shot_img_dict[shot_num]}

        super(CustamDataset, self).__init__(
            dataset_config=json_config, train_flag=train_flag,
            label_threshold=label_threshold, object_connectivity=object_connectivity,
            area_threshold=area_threshold, relative_threshold=relative_threshold,
            relative_threshold_ratio=relative_threshold_ratio,
            ann_scale_factor=ann_scale_factor, noisy_mask_threshold=noisy_mask_threshold,
            **super_args
        )
