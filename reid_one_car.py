"""
The sample of getting the reid features of one car
and then reid
"""
import argparse
import torch
from tools.jsonio import read_json
from metrics import metrices_map
from models import TransReIDBase_Inference
from PIL import Image
from pathlib import Path
from dataset import get_transform
from config import cfg

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", type=str, help="path of query car",  default=Path("VehicleData")/"query"/"0002_c002_00030600_0.jpg")
    parser.add_argument("--topk",type=int, default=5)
    parser.add_argument("--precalculated", type=str, default=Path("precalculated"))
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    cfg.merge_from_file(Path("config")/"transreid_256_veri_gan.yml")
    cfg.freeze()
    
    device = torch.device(args.device) if args.device == "cpu" else torch.device(f"cuda:{args.device}")

    reid_model = TransReIDBase_Inference(cfg = cfg)
    reid_model.eval()
    # If having cuda device:
    reid_model = reid_model.to(device = device)
    
    # Get a image
    preprocessor =  get_transform(
        target_size = cfg.INPUT.SIZE_TEST, 
        normalize_args = {
            'mean':cfg.INPUT.PIXEL_MEAN, 
            'std':cfg.INPUT.PIXEL_STD
        }
    )
    car_img = preprocessor( Image.open(args.query_path).convert("RGB") ).unsqueeze(0)
    car_img = car_img.to(device = device)
    
    # Car features extraction :
    # output : The features of car_img, a tensor with shape (1, 768)
    reid_features = reid_model(car_img)
    
    # Calculate the distance based on the characteristics of all the cars in the gallery.
    # precalculate gallery features and their coressponding file names : 
    gallery_car_ids_list = read_json(Path(args.precalculated)/"gallery"/"gallery_items.json")
    gallery = torch.load(Path(args.precalculated)/"gallery"/"gallery_reid_features.pt", map_location = "cpu")
    gallery = gallery.to(device)

    # distance :
    distances_matrix:torch.Tensor = metrices_map['cosine'](reid_features, gallery)
    distances_matrix = distances_matrix.cpu()
    # Sorting the distances from smallest to largest and get index
    sim = distances_matrix.argsort(dim=1)
    # get top k 
    topk_indices = sim[0, :args.topk]
    
    # get path of top k cars 
    most_similarity = [
        Path("VehicleData")/"gallery"/gallery_car_ids_list[i] for i in topk_indices
    ]
    for topi, p in enumerate(most_similarity):
        print(f"top{topi+1} : {str(p)}, distance : {distances_matrix[0, topk_indices[topi]]:.6f}")
    
