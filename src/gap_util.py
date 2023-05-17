

import os
from os.path import join
from os import listdir


import gzip
import pickle
import sys
import numpy as np

from datetime import datetime, timezone, timedelta
from imageio import imread, imsave
from PIL import Image

import torch
import torch.utils.data as data

sgt_timezone_offset = 8.0  # SGT UTC+8
sgt_tzinfo = timezone(timedelta(hours=sgt_timezone_offset))

str_util_abs_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(str_util_abs_path, '.'))
str_project_root_path = os.path.abspath(os.path.join(str_util_abs_path, '../'))


g_gpu_available = torch.cuda.is_available()

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def pickle_dump(obj, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        
# Shared functions
def console_log(*args, par_end='\n', str_tag='MalPage'):
    dt_now = datetime.now(sgt_tzinfo)
    str_now = dt_now.strftime("%Y-%m-%d/%H:%M:%S")    
    print(f'[{str_now}] [{str_tag}]', *args, end=par_end)   

def console_error(*args, par_end='\n', str_tag='MalPage'):
    dt_now = datetime.now(sgt_tzinfo)
    str_now = dt_now.strftime("%Y-%m-%d/%H:%M:%S")    
    print(f'[{str_now}] [Error]', *args, end=par_end)
    
def gz_pickle_load(path):
    #log.debug(f"Loading {path}")
    with gzip.open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except BaseException as e:            
            raise e

def gz_pickle_dump(obj, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with gzip.open(path, 'wb') as f:
        pickle.dump(obj, f)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = imread(filepath)
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
    img = np.array(Image.fromarray(img).resize((256, 256)))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    img = preprocess_img(img)
    return img


def save_img(img, filename):
    img = deprocess_img(img)
    img = img.numpy()
    img *= 255.0
    img = img.clip(0, 255)
    img = np.transpose(img, (1, 2, 0))
    img = np.array(Image.fromarray(img).resize((250, 200, 3)))
    img = img.astype(np.uint8)
    imsave(filename, img)
    print("Image saved as {}".format(filename))


def preprocess_img(img):
    # [0,255] image to [0,1]
    min = img.min()
    max = img.max()
    img = torch.FloatTensor(img.size()).copy_(img)
    img.add_(-min).mul_(1.0 / (max - min))

    # RGB to BGR
    idx = torch.LongTensor([2, 1, 0])
    img = torch.index_select(img, 0, idx)

    # [0,1] to [-1,1]
    img = img.mul_(2).add_(-1)

    # check that input is in expected range
    assert img.max() <= 1, 'badly scaled inputs'
    assert img.min() >= -1, "badly scaled inputs"

    return img


def custom_pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return cvtColor(img)

def cvtColor(image):
    if image.mode != 'RGBA':
       image = image.convert('RGBA')
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image
    
def deprocess_img(img):
    # BGR to RGB
    idx = torch.LongTensor([2, 1, 0])
    img = torch.index_select(img, 0, idx)

    # [-1,1] to [0,1]
    img = img.add_(1).div_(2)

    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

    def __getitem__(self, index):
        # Load Image
        input = load_img(join(self.a_path, self.image_filenames[index]))
        target = load_img(join(self.b_path, self.image_filenames[index]))

        return input, target

    def __len__(self):
        return len(self.image_filenames)

def get_training_set(root_dir) -> DatasetFromFolder:
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir)


def get_test_set(root_dir) -> DatasetFromFolder:
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir)