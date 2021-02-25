import os


def imread(filename, grayscale=False):
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    img = Image.open(filename)

    if grayscale:
        img = img.convert('L')

    return to_tensor(img)

