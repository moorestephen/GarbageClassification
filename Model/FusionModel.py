import torch
import torch.nn as nn
from Model.TextModel import TextModel
from Model.ImageModel import ImageModel

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)  # Project fused features

    def forward(self, text_embed, image_embed):
        # Reshape image embedding
        image_embed = image_embed.unsqueeze(1)  # (batch, 1, embed_dim)

        # print(f"text_embed shape: {text_embed.shape}")
        # print(f"image_embed shape: {image_embed.shape}")

        # Apply cross-attention
        fused, _ = self.cross_attention(query = text_embed, key = image_embed, value = image_embed)

        return self.fc(fused.squeeze(1))  # (batch, embed_dim)


# class FusionModel(nn.Module):
#     def __init__(self, num_classes, image_input_shape, transfer=False):
#         super(FusionModel, self).__init__()
#         self.image_model = ImageModel(num_classes, image_input_shape, transfer)
#         self.text_model = TextModel(num_classes)
        
#         self.fc = nn.Sequential(
#             nn.Linear(2 * num_classes, 2 * num_classes),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(2 * num_classes, num_classes)
#         )
    
#     def forward(self, image, input_ids, attention_mask):
#         img_features = self.image_model(image)  # (batch_size, img_feature_dim)
#         text_features = self.text_model(input_ids, attention_mask)  # (batch_size, text_feature_dim)
#         fused_features = torch.cat((img_features, text_features), dim=1)  # (batch_size, img_feature_dim + text_feature_dim)
#         return self.fc(fused_features)  # (batch_size, num_classes)

class FusionModel(nn.Module):
    def __init__(self, num_classes, image_input_shape, embed_dim=512, transfer=False):
        super(FusionModel, self).__init__()
        self.image_model = ImageModel(image_input_shape, embed_dim, transfer)
        self.text_model = TextModel(embed_dim)
        self.cross_attention = CrossAttentionFusion(embed_dim = embed_dim)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, image, input_ids, attention_mask):
        img_features = self.image_model(image)  # (batch, embed_dim)
        text_features = self.text_model(input_ids, attention_mask)  # (batch, embed_dim)

        fused_features = self.cross_attention(text_features, img_features)  # (batch, embed_dim)

        fused_features = fused_features.mean(dim=1)  # (batch, embed_dim)

        return self.fc(fused_features)  # (batch, num_classes)