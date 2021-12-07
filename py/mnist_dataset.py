from mnist import MNIST
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import cv2
import numpy as np

# mndata = MNIST('samples')
train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
for curr in train:
    print(curr)

# images, labels = mndata.load_training()

# print(images[0])
# img = np.asarray(images[0])
# img = img.astype(np.uint8)
# plt.imshow(img.view(28, 28), interpolation='nearest')
# plt.show()

# cv2.imshow("image", img)