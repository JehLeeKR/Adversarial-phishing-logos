import torch.utils.data as data
from PIL import Image, ImageOps
import os


class GetLoaderV2(data.Dataset):
    def __init__(self, file_path, data_root, transform=None, grayscale=False):
        self.file_path = file_path
        self.data_root = data_root
        self.grayscale = grayscale
        self.transform = transform
        self.classes = 181
        with open(file_path, encoding='utf-8') as f:
            data_lines = f.readlines()
            self.annotation_lines = data_lines

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
        img_path_full = os.path.join(self.data_root, annotation_path)
        image = Image.open(img_path_full)
        if self.grayscale:
            img = image.convert('L').convert('RGB')
        else:
            img = image.convert('RGB')

        img = ImageOps.expand(img, (
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=(255, 255, 255))

        if self.transform is not None:
            img = self.transform(img)

        label = int(self.annotation_lines[index].split(';')[0])
        return img, label
