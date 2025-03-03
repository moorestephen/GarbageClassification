import torch
import torch.nn as nn
from torchvision import models

class ImageModel(nn.Module):
    def __init__(self, input_shape, embed_dim = 512, transfer=False):
        super().__init__()

        self.transfer = transfer
        # self.num_classes = num_classes
        self.input_shape = input_shape
        
        # transfer learning if weights=True
        self.feature_extractor = models.resnet18(weights = "IMAGENET1K_V1" if transfer else None)

        if self.transfer:
            self.feature_extractor.eval() # Set to evaluation mode
            for param in self.feature_extractor.parameters():
                param.requires_grad = False # Freeze parameters

    #     n_features = self._get_conv_output(self.input_shape)
    #     self.classifier = nn.Linear(n_features, num_classes)

    # def _get_conv_output(self, shape):
    #     batch_size = 1
    #     tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

    #     output_feat = self.feature_extractor(tmp_input) 
    #     n_size = output_feat.data.view(batch_size, -1).size(1)
    #     return n_size

        self.feature_extractor.fc = nn.Identity() # Remove final layer
        self.feature_projector = nn.Linear(512, embed_dim) # Number of embeddings

    def forward(self, x):
        x = self.feature_extractor(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        x = self.feature_projector(x)
        return x
