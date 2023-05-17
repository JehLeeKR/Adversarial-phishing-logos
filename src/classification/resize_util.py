import torch.utils.data as data
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os
import imageio
from pathlib import Path


class DataSet(data.Dataset):
    def __init__(self, img_dir, resize):
        super(DataSet, self).__init__()
        self.img_paths = glob('{:s}/*'.format(img_dir))
        self.transform = transforms.Compose([transforms.Resize(size=(resize, resize))])

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item]).convert('RGB')
        img = self.transform(img)

        return img, self.img_paths[item]

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='train')
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--save_dir', type=str, default='train_resized')
    args = parser.parse_args()
    img_dir = args.img_dir
    for dir_name in os.listdir(args.img_dir):
        img_dir = args.img_dir + '/' + dir_name
        save_dir = args.save_dir + '/' + dir_name.capitalize()
        if os.path.exists(save_dir):
            continue
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        dataset = DataSet(img_dir, args.resize)
        print('Processing brand: ', dir_name)

        for i in range(len(dataset) // 2):
            try:
                img, path = dataset[i]
            except:
                print('Processing error')
            else:
                path = str(i) + ".jpg"
                imageio.imwrite(save_dir+'/{:s}'.format(path), img)

