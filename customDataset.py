import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

# Credit to: https://www.youtube.com/watch?v=ZoZHd0Zm3RY


dataset_creation = False


if dataset_creation:
    images = []

    for file_name in glob.iglob('xrays/*/*/*', recursive=True):
        if 'NORMAL' in file_name:
            images.append((os.path.basename(file_name), 0))
        else:
            images.append((os.path.basename(file_name), 1))

    filenames = pd.DataFrame(images, columns=['File', 'Label'])

    filenames.to_csv('pneumonia.csv', index=False)


class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

            return (image, y_label)
