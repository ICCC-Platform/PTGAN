import os
import argparse
from tqdm import tqdm
from config import cfg
from dataset import Gallery, get_transform
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from models import TransReIDBase_Inference

def parse_arguments():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()



def main(DATASET, save_root:Path):

    device = torch.device(f"cuda:{cfg.MODEL.DEVICE_ID}") if torch.cuda.is_available() else torch.device("cpu")

    dset = Gallery(
        root = Path("VehicleData")/DATASET,
        img_transform = get_transform(
            target_size=(256,256),
            normalize_args={
                'mean':cfg.INPUT.PIXEL_MEAN,
                'std':cfg.INPUT.PIXEL_STD
            }
        )
    )
    gallery_loader = DataLoader(dataset=dset, batch_size=1)
    reid_model = TransReIDBase_Inference(cfg=cfg)
    reid_model.eval()
    reid_model.to(device=device)

  
    features = None
    fnames = []
    for imgs, names in tqdm(gallery_loader):
        if features is not None:
            features = torch.vstack((features, reid_model(imgs.to(device))))
        else:
            features = reid_model(imgs.to(device))
        fnames += list(names)
    
    features = features.cpu()
    print(features.size())

    torch.save(features, save_root/DATASET/f"{DATASET}_reid_features.pt")
    
    with open(save_root/DATASET/f"{DATASET}_items.json", "w+") as f:
        json.dump(fnames, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    
    parse_arguments()
    save_root = Path("precalculated")
    if not os.path.exists(save_root ):
        os.mkdir(save_root)
        os.mkdir(save_root/"query")
        os.mkdir(save_root/"gallery")
    
    main(DATASET="query", save_root=save_root)
    main(DATASET="gallery", save_root=save_root)