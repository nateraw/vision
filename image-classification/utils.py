from PIL import Image

import torch


class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def __call__(self, batch):
        encodings = self.feature_extractor([x['image'] for x in batch], return_tensors='pt')
        encodings['labels'] = torch.tensor([x['labels'] for x in batch], dtype=torch.long)
        return encodings


def pil_loader(path: str):
    with open(path, 'rb') as f:
        im = Image.open(f)
        return im.convert('RGB')


def image_loader(example_batch):
    example_batch['image'] = [pil_loader(f) for f in example_batch['image_file_path']]
    return example_batch
