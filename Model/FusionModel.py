import torch
import torch.nn as nn
from Model.TextModel import TextModel
from Model.ImageModel import ImageModel


class FusionModel(nn.Module):
    def __init__(self, num_classes, image_input_shape, transfer=False):
        super(FusionModel, self).__init__()
        self.image_model = ImageModel(num_classes, image_input_shape, transfer)
        self.text_model = TextModel(num_classes)
        
        self.fc = nn.Sequential(
            nn.Linear(2 * num_classes, 2 * num_classes),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2 * num_classes, num_classes)
        )
    
    def forward(self, image, input_ids, attention_mask):
        img_features = self.image_model(image)  # (batch_size, img_feature_dim)
        text_features = self.text_model(input_ids, attention_mask)  # (batch_size, text_feature_dim)
        fused_features = torch.cat((img_features, text_features), dim=1)  # (batch_size, img_feature_dim + text_feature_dim)
        return self.fc(fused_features)  # (batch_size, num_classes)