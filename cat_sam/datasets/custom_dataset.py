from .base import BaseDataset
import os
from PIL import Image
import numpy as np

class CustomDataset(BaseDataset):
    def __init__(self, split='train', data_root=None, transform=None, **kwargs):
        super().__init__(split, data_root, transform, **kwargs)
        # カスタムデータセット固有の初期化
        self.img_dir = os.path.join(self.data_root, 'images')
        self.mask_dir = os.path.join(self.data_root, 'masks')
        self.img_list = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # データセット固有の前処理をここに追加

        # BaseDatasetの共通処理を使用
        return self.preparation_for_train(image, mask)