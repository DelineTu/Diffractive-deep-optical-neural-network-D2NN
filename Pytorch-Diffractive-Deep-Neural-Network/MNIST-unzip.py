import os
import torch
import gzip
import struct
import numpy as np
from torch.utils.data import TensorDataset
import shutil

# 定义数据集路径
root = "data/MNIST"

# 解压文件
def decompress_files(root):
    for file_name in os.listdir(os.path.join(root, 'raw')):
        if file_name.endswith('.gz'):
            with gzip.open(os.path.join(root, 'raw', file_name), 'rb') as f_in:
                with open(os.path.join(root, 'raw', file_name[:-3]), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

# 解压原始文件
decompress_files(root)

# 读取图像文件
def read_image_file(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return torch.tensor(images, dtype=torch.float32)

# 读取标签文件
def read_label_file(file_path):
    with open(file_path, 'rb') as f:
        magic, num_items = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return torch.tensor(labels, dtype=torch.int64)

# 处理训练集
train_images_path = os.path.join(root, 'raw', 'train-images-idx3-ubyte')
train_labels_path = os.path.join(root, 'raw', 'train-labels-idx1-ubyte')
train_images = read_image_file(train_images_path)
train_labels = read_label_file(train_labels_path)
torch.save((train_images, train_labels), os.path.join(root, 'processed', 'training.pt'))

# 处理测试集
test_images_path = os.path.join(root, 'raw', 't10k-images-idx3-ubyte')
test_labels_path = os.path.join(root, 'raw', 't10k-labels-idx1-ubyte')
test_images = read_image_file(test_images_path)
test_labels = read_label_file(test_labels_path)
torch.save((test_images, test_labels), os.path.join(root, 'processed', 'test.pt'))