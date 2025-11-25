import os
from os import getcwd
from typing import List, Tuple
from utils.utils import get_classes
import config

# -------------------------------------------------------------------#
#   classes_path    Points to the txt file under model_data, related to your own training dataset.
#                   Be sure to modify classes_path before training to correspond to your dataset.
#                   The txt file contains the categories you want to distinguish.
#                   It should be consistent with the classes_path used for training and prediction.
# -------------------------------------------------------------------#
classes_path: str = config.CLASSES_PATH
# -------------------------------------------------------#
#   datasets_path   Points to the path where the dataset is located.
# -------------------------------------------------------#
datasets_path: str = config.DATASET_DIR

sets: List[str] = ["train", "test"]
classes: List[str]
_: int
classes, _ = get_classes(classes_path)

def main() -> None:
    """
    Generates 'train_data.txt' and 'test_data.txt' from image files in the dataset directory.
    Each line in the output file contains a class ID and the path to an image, separated by a semicolon.
    """
    for se in sets:
        # Open the output file for writing (e.g., 'train_data.txt')
        list_file = open(se + '_data.txt', 'w')
        datasets_path_t: str = os.path.join(datasets_path, se)
        types_name: List[str] = os.listdir(datasets_path_t)
        # Iterate through each class directory (e.g., 'adidas', 'apple')
        for type_name in types_name:
            if type_name not in classes:
                continue
            cls_id: int = classes.index(type_name)
            photos_path: str = os.path.join(datasets_path_t, type_name)
            photos_name: List[str] = os.listdir(photos_path)
            # Iterate through each image in the class directory
            for photo_name in photos_name:
                _: str
                postfix: str
                _, postfix = os.path.splitext(photo_name)
                if postfix not in ['.jpg', '.png', '.jpeg']:
                    continue
                list_file.write(str(cls_id) + ";" + '%s' % (os.path.join(photos_path, photo_name)))
                list_file.write('\n') # Add a newline character after each entry
        list_file.close()

if __name__ == "__main__":
    main()
