from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(
        self, tokenizer, img_path_list, text_list, transforms, config, label_list=None
    ):
        self.tokenizer = tokenizer
        self.img_path_list = img_path_list
        self.text_list = text_list
        self.label_list = label_list
        self.transforms = transforms
        self.config = config

    def __getitem__(self, index):

        # NLP
        text = self.text_list[index]
        text_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Image
        img_path = "./data/" + self.img_path_list[index][1:]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transforms(image).unsqueeze(0)

        # Label
        if self.label_list:
            label = self.label_list[index]
            return image_tensor, text_dict, label
        else:
            return image, text_dict

    def __len__(self):
        return len(self.img_path_list)
