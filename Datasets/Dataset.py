import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import numpy as np
import re

class GarbageDataset(ImageFolder):

    def __init__(self, root, tokenizer, max_len, transform=None, target_transform=None):
        super(GarbageDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index): # Modified from source code for DatasetFolder
        path, target = self.samples[index]

        sample = self.loader(path) # PIL Image
        if self.transform is not None: # Apply transformations if appropriate
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Added: text information and encoding
        file_name_no_ext, _ = os.path.splitext(path)
        text = file_name_no_ext.split('/')[-1].replace('_', ' ')
        text_without_digits = re.sub(r'\d+', '', text)

        encoding = self.tokenizer.encode_plus(
                        text_without_digits,
                        add_special_tokens=True,
                        max_length=self.max_len,
                        return_token_type_ids=False,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                    )

        return {
            'image': sample,
            'text': text_without_digits,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(target, dtype = torch.long)
        }