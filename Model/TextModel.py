import torch
import torch.nn as nn
from transformers import DistilBertModel

# class TextModel(nn.Module):
#     def __init__(self, num_classes):
#         super(TextModel, self).__init__()
#         self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
#         self.drop = nn.Dropout(0.3)
#         self.out = nn.Linear(self.distilbert.config.hidden_size, num_classes)

#     def forward(self, input_ids, attention_mask):
#         pooled_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)[0]
#         output = self.drop(pooled_output[:,0])
#         return self.out(output)

class TextModel(nn.Module):
    def __init__(self, embed_dim = 512):
        super(TextModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(0.3)

        self.feature_projector = nn.Linear(self.distilbert.config.hidden_size, embed_dim) # Project to embedding size

    def forward(self, input_ids, attention_mask):
        pooled_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)[0]
        output = self.drop(pooled_output)
        
        # Project to embedding size
        return self.feature_projector(output)