import argparse
import os
import numpy as np
from os.path import join
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from misc import get_idle_gpu, set_randomness
from train import batch_to_cuda

from cat_sam.datasets.whu import WHUDataset
from cat_sam.datasets.kvasir import KvasirDataset
from cat_sam.datasets.sbu import SBUDataset
from cat_sam.datasets.custom_dataset import CustomDataset  # カスタムデータセットのインポートを追加
from cat_sam.models.modeling import CATSAMT, CATSAMA
from cat_sam.utils.evaluators import SamHQIoU, StreamSegMetrics


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', default='./data', type=str,
        help="The directory that the datasets are placed. Default to be ./data"
    )
    parser.add_argument(
        '--num_workers', default=4, type=int,
        help="The num_workers argument used for the testing dataloaders. Default to be 4."
    )
    parser.add_argument(
        '--batch_size', default=2, type=int,
        help="The batch size for the testing dataloader. Default to be 2."
    )
    parser.add_argument(
        '--dataset', required=True, type=str, choices=['whu', 'sbu', 'kvasir', 'custom'],
        help="Your target dataset. This argument is required."
    )
    parser.add_argument(
        '--ckpt_path', required=True, type=str,
        help="The absolute path to your target checkpoint file. This argument is required."
    )
    parser.add_argument(
        '--sam_type', default='vit_l', type=str, choices=['vit_b', 'vit_l', 'vit_h'],
        help='The type of the backbone SAM model. Default to be vit_l.'
    )
    parser.add_argument(
        '--cat_type', required=True, type=str, choices=['cat-a', 'cat-t'],
        help='The type of the CAT-SAM model. This argument is required.'
    )
    # カスタムデータセット用の引数を追加
    parser.add_argument(
        '--json_config', type=str,
        help='Path to the JSON configuration file for custom dataset.'
    )
    return parser.parse_args()


def run_test(test_args):
    set_randomness()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    if test_args.dataset == 'whu':
        dataset_class = WHUDataset
    elif test_args.dataset == 'kvasir':
        dataset_class = KvasirDataset
    elif test_args.dataset == 'sbu':
        dataset_class = SBUDataset
    elif test_args.dataset == 'custom':
        dataset_class = CustomDataset
        # カスタムデータセット用の設定を読み込む
        with open(test_args.json_config, 'r') as f:
            custom_config = json.load(f)
    else:
        raise ValueError(f'invalid dataset name: {test_args.dataset}!')

    if test_args.dataset == 'custom':
        test_dataset = dataset_class(
            data_dir=test_args.data_dir,  # join()を使用しない
            json_config=custom_config,  # json_configを直接渡す
            train_flag=False
        )
    else:
        test_dataset = dataset_class(
            data_dir=join(test_args.data_dir, test_args.dataset), train_flag=False
        )

    test_dataloader = DataLoader(
        dataset=test_dataset, shuffle=False, drop_last=False,
        batch_size=test_args.batch_size, num_workers=test_args.num_workers,
        collate_fn=test_dataset.collate_fn
    )

    # 以下は変更なし
    if test_args.cat_type == 'cat-t':
        model_class = CATSAMT
    elif test_args.cat_type == 'cat-a':
        model_class = CATSAMA
    else:
        raise ValueError(f'invalid cat_type: {test_args.cat_type}!')
    model = model_class(model_type=test_args.sam_type).to(device=device)
    model_state_dict = torch.load(test_args.ckpt_path, map_location=device)
    if 'model' in model_state_dict.keys():
        model_state_dict = model_state_dict['model']
    model.load_state_dict(model_state_dict)
    model.eval()

    if test_args.dataset == 'hqseg44k':
        iou_eval = SamHQIoU()
    else:
        if test_args.dataset == 'custom':
            # カスタムデータセット用のクラス名を設定
            class_names = test_dataset.class_names  # または適切なクラス名のリスト
        elif test_args.dataset in ['jsrt', 'fls']:
            class_names = test_dataset.class_names
        else:
            class_names = ['Background', 'Foreground']
        iou_eval = StreamSegMetrics(class_names=class_names)

    os.makedirs('test_results', exist_ok=True)

    for test_step, batch in enumerate(tqdm(test_dataloader)):
        print(f"Processing batch {test_step + 1}")

        # バッチデータの処理
        images = batch['images'][0] if isinstance(batch['images'], list) else batch['images']
        box_coords = batch['box_coords'][0] if isinstance(batch['box_coords'], list) else batch['box_coords']
        images = images.to(device)
        box_coords = box_coords.to(device)

        print(f"Input image shape: {images.shape}")
        print(f"Input box_coords shape: {box_coords.shape}")

        # モデルの推論
        with torch.no_grad():
            model.set_infer_img(img=images)
            masks_pred = model.infer(box_coords=box_coords)

        # モデル出力の処理
        if isinstance(masks_pred, list):
            masks_pred = torch.stack(masks_pred, dim=0)
        masks_pred = masks_pred.cpu().numpy()

        masks_gt = batch['gt_masks'][0] if isinstance(batch['gt_masks'], list) else batch['gt_masks']
        masks_gt = masks_gt.cpu().numpy() if isinstance(masks_gt, torch.Tensor) else masks_gt

        print(f"Shape of masks_pred: {masks_pred.shape}")
        print(f"Shape of masks_gt: {masks_gt.shape}")
        print(f"Unique values in masks_pred: {np.unique(masks_pred)}")
        print(f"Unique values in masks_gt: {np.unique(masks_gt)}")

        # IoU の計算
        try:
            iou_eval.update(masks_gt, masks_pred, batch['index_name'])
        except Exception as e:
            print(f"Error in iou_eval.update: {str(e)}")
            print(f"masks_gt shape: {masks_gt.shape}, dtype: {masks_gt.dtype}")
            print(f"masks_pred shape: {masks_pred.shape}, dtype: {masks_pred.dtype}")
            print(f"batch['index_name']: {batch['index_name']}")

    # 最終的な mIoU の計算と出力
    miou = iou_eval.compute()[0]['Mean Foreground IoU']
    print(f'Final mIoU: {miou:.2%}')


if __name__ == '__main__':
    args = parse()
    used_gpu = get_idle_gpu(gpu_num=1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(used_gpu[0])
    args.used_gpu, args.gpu_num = used_gpu, 1
    run_test(test_args=args)
