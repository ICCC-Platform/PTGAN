from torchvision import transforms

class RectScale(object):
    def __init__(self, height, width, interpolation=3):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)

def get_transform(target_size:tuple[int], normalize_args:dict[str, list[float]]) -> transforms:
    return transforms.Compose(
        [
            RectScale(target_size[0], target_size[1]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = normalize_args['mean'], 
                std = normalize_args['std']
            )
        ]
    )
