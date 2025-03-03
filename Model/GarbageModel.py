import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel

class GarbageModel(nn.Module):
    def __init__(self, num_classes, embed_dim = 256):
        super(GarbageModel, self).__init__()

        # Text model
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.distilbert.config.hidden_size, embed_dim) # Project to embedding size

        # Image model
        self.model_ft = models.resnet18(weights = "IMAGENET1K_V1")
        for param in self.model_ft.parameters(): # Make it fixed feature extractor
            param.requires_grad = False
        self.num_fftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.num_fftrs, embed_dim) # Project to embedding size

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2 * embed_dim, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):

        # Image model
        image_output = self.model_ft(image)

        # Text model
        pooled_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)[0]
        output = self.drop(pooled_output[:,0])
        text_output = self.out(output)
    
        # Fusion layer
        fused_features = torch.cat((image_output, text_output), dim=1)
        return self.fusion(fused_features)