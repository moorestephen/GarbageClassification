import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from Model.GarbageModel import GarbageModel
from Datasets.Dataset import GarbageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device; check if GPU is available

print(f'Device: {device}', flush = True) 

TEST_PATH = f"/work/souza_lab/garbage_data/CVPR_2024_dataset_Test"

# Define transformation, datasets, and batch size / num workers - from tutorials
torchvision_transform_test = transforms.Compose([transforms.Resize((224,224)),\
    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225])])


# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_len = 24

# Load Datasets
test_dataset = GarbageDataset(
    root = TEST_PATH, tokenizer = tokenizer, max_len = max_len, transform = torchvision_transform_test
)

# Define batch size and number of workers (adjust as needed)
batch_size = 16
num_workers = 4

# Create data loaders
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

PATH = './garbage_net.pth'

model = GarbageModel(
    num_classes = 4
)
model.to(device)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))  # Load weights
model.eval()  # Set model to evaluation mode

correct = 0
total = 0

labels_tracker = []
predictions = []

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data['image'].to(device), data['label'].to(device)
        input_ids, attention_mask = data['input_ids'].to(device), data['attention_mask'].to(device)

        outputs = model(inputs, input_ids, attention_mask)

        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        labels_tracker.extend(labels.cpu().numpy())
        predictions.extend(predicted.cpu().numpy())

print(f'Accuracy of the network on the test images: {100 * correct / total} %', flush = True)

cm = confusion_matrix(labels_tracker, predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Save confusion matrix
plt.savefig('confusion_matrix.png')
# Save predictions
torch.save(predictions, 'predictions.pth')
# Save labels
torch.save(labels_tracker, 'labels.pth')