
import torch
#from torchvision import datasets, transforms

# 定义数据预处理变换
#transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.1307,), (0.3081,))
#])

# 下载训练数据集
#train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 下载测试数据集
#test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
import struct

import struct
import os

root="../data"

def read_magic_number_and_dimensions(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        print(f"Magic number: {magic}")
        print(f"Number of images: {num_images}")
        print(f"Image size: {rows} x {cols}")

# 检查 train-images.idx3-ubyte 文件
train_images_path = os.path.join(root, 'MNIST', 'raw', 'train-images-idx3-ubyte')
read_magic_number_and_dimensions(train_images_path)

# 检查 train-labels.idx1-ubyte 文件
train_labels_path = os.path.join(root, 'MNIST', 'raw', 'train-labels-idx1-ubyte')
read_magic_number_and_dimensions(train_labels_path)

# 检查 t10k-images.idx3-ubyte 文件
test_images_path = os.path.join(root, 'MNIST', 'raw', 't10k-images-idx3-ubyte')
read_magic_number_and_dimensions(test_images_path)

# 检查 t10k-labels.idx1-ubyte 文件
test_labels_path = os.path.join(root, 'MNIST', 'raw', 't10k-labels-idx1-ubyte')
read_magic_number_and_dimensions(test_labels_path)