import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class Gallery(Dataset):
    
    def __init__(self, root:os.PathLike, img_transform:transforms = None) -> None:

        super().__init__()

        self.root = Path(root)
        self.imgs:list[Path] = [i for i in self.root.glob("*.jpg")]
        self._l:int = len(self.imgs)
        self.transform:transforms = img_transform

    def __len__(self) -> int:
        return self._l
    
    def __getitem__(self, index) -> tuple[torch.Tensor, str]:
        
        img_path:Path = self.imgs[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            return self.transform(img), img_path.name

        return img, img_path.name