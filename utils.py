import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import json

# Custom data loader for Flickr8k dataset
class Loadf8k(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        filename = root + '/flickr8k.json'
        self.dataset = json.load(open(filename, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        path = 'images/' + self.dataset[img_id]['filename']

        try:
            image = Image.open(os.path.join(root, path)).convert('RGB')
        except IOError:
            print(f"Error reading image {path}")
            return None, None

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    # Filter out any failed image loads
    data = [item for item in data if item is not None]

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(data, 0)

    return images
