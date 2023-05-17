from PIL import Image
from classification import Discriminator

discriminator = Discriminator()


def predict():
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except (PIL.UnidentifiedImageError, FileNotFoundError):
        print('Error opening image')
        return

    class_name = discriminator.detect_image(image)
    print(class_name)


if __name__ == "__main__":
    predict()
