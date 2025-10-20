import os
from PIL import Image
import matplotlib.pyplot as plt

def list_images(data_path="data/deepfashion_samples"):
    return [os.path.join(data_path, f) for f in os.listdir(data_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))]

def show_samples(n=3, data_path="data/deepfashion_samples"):
    images = list_images(data_path)[:n]
    for p in images:
        img = Image.open(p)
        plt.imshow(img)
        plt.title(os.path.basename(p))
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    show_samples()
