from PIL import Image
from classification import Discriminator


def predict():
    discriminator = Discriminator()
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except (Image.UnidentifiedImageError, FileNotFoundError):
        print('Error opening image')
        return

    class_name = discriminator.detect_image(image)
    print(class_name)

def main():
    predict()

if __name__ == "__main__":
    main()
