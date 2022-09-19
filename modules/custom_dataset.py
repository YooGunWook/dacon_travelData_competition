import cv2
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self, img_path_list, text_vectors, label_list, transforms, infer=False
    ):
        self.img_path_list = img_path_list
        self.text_vectors = text_vectors
        self.label_list = label_list
        self.transforms = transforms
        self.infer = infer

    def __getitem__(self, index):
        # NLP
        text_vector = self.text_vectors[index]

        # Image
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        # Label
        if self.infer:
            return image, torch.Tensor(text_vector).view(-1)
        else:
            label = self.label_list[index]
            return image, torch.Tensor(text_vector).view(-1), label

    def __len__(self):
        return len(self.img_path_list)
