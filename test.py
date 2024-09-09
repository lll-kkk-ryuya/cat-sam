import argparse
import os
from os.path import join
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from misc import set_randomness
from train import batch_to_cuda

from cat_sam.datasets.custom_dataset import CustomDataset
from cat_sam.models.modeling import CATSAMT, CATSAMA
from cat_sam.utils.evaluators import StreamSegMetrics

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='.', type=str,
                        help="The directory that the datasets are placed. Default to current directory.")
    parser.add_argument('--num_workers', default=2, type=int,
                        help="The num_workers argument used for the testing dataloaders. Default to be 2.")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="The batch size for the testing dataloader. Default to be 1.")
    parser.add_argument('--dataset', default='custom', type=str, choices=['custom'],
                        help="Your target dataset. Default is custom.")
    parser.add_argument('--ckpt_path', default='/content/drive/MyDrive/trained_models/best_model.pth', type=str,
                        help="The path to your target checkpoint file. Default is the trained model in Google Drive.")
    parser.add_argument('--sam_type', default='vit_b', type=str, choices=['vit_b', 'vit_l', 'vit_h'],
                        help='The type of the backbone SAM model. Default to be vit_b.')
    parser.add_argument('--cat_type', default='cat-a', type=str, choices=['cat-a', 'cat-t'],
                        help='The type of the CAT-SAM model. Default to be cat-a.')
    parser.add_argument('--json_config', default='test_dataset_config.json', type=str,
                        help='Path to the JSON configuration file for custom dataset.')
    return parser.parse_args()

def run_test(test_args):
    set_randomness()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with open(test_args.json_config, 'r') as f:
        custom_config = json.load(f)

    test_dataset = CustomDataset(
        data_dir=test_args.data_dir,
        train_flag=False,
        json_config=custom_config
    )

    test_dataloader = DataLoader(
        dataset=test_dataset, shuffle=False, drop_last=False,
        batch_size=test_args.batch_size, num_workers=test_args.num_workers,
        collate_fn=test_dataset.collate_fn
    )

    model_class = CATSAMA if test_args.cat_type == 'cat-a' else CATSAMT
    model = model_class(model_type=test_args.sam_type).to(device=device)
    model_state_dict = torch.load(test_args.ckpt_path, map_location=device)
    if 'model' in model_state_dict.keys():
        model_state_dict = model_state_dict['model']
    model.load_state_dict(model_state_dict)
    model.eval()

    class_names = test_dataset.class_names if hasattr(test_dataset, 'class_names') else ['Background', 'Foreground']
    iou_eval = StreamSegMetrics(class_names=class_names)

    for test_step, batch in enumerate(tqdm(test_dataloader)):
        batch = batch_to_cuda(batch, device)
        with torch.no_grad():
            model.set_infer_img(img=batch['images'])
            masks_pred = model.infer(box_coords=batch['box_coords'])

        masks_gt = batch['gt_masks']
        for masks in [masks_pred, masks_gt]:
            for i in range(len(masks)):
                if len(masks[i].shape) == 2:
                    masks[i] = masks[i][None, None, :]
                if len(masks[i].shape) == 3:
                    masks[i] = masks[i][None, :]
                if len(masks[i].shape) != 4:
                    raise RuntimeError

        iou_eval.update(masks_gt, masks_pred, batch['index_name'])

    miou = iou_eval.compute()[0]['Mean Foreground IoU']
    print(f'mIoU: {miou:.2%}')

if __name__ == '__main__':
    args = parse()
    run_test(test_args=args)
