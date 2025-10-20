from PIL import Image
def load_image(path: str):
    return Image.open(path)
