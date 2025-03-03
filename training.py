import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from transformers import DistilBertTokenizer

from Datasets.Dataset import GarbageDataset
from Model.GarbageModel import GarbageModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device; check if GPU is available

print(f'Device: {device}', flush = True) 

TRAIN_PATH = f"/work/souza_lab/garbage_data/CVPR_2024_dataset_Train"
VAL_PATH = f"/work/souza_lab/garbage_data/CVPR_2024_dataset_Val"
TEST_PATH = f"/work/souza_lab/garbage_data/CVPR_2024_dataset_Test"

# Define transformation, datasets, and batch size / num workers - from tutorials

# Transforms 
torchvision_transform = transforms.Compose([transforms.Resize((224,224)),\
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225] )])

torchvision_transform_test = transforms.Compose([transforms.Resize((224,224)),\
    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225])])

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_len = 24

# Load Datasets
train_dataset = GarbageDataset(
    root = TRAIN_PATH, tokenizer = tokenizer, max_len = max_len, transform = torchvision_transform
)
val_dataset = GarbageDataset(
    root = VAL_PATH, tokenizer = tokenizer, max_len = max_len, transform = torchvision_transform
)
test_dataset = GarbageDataset(
    root = TEST_PATH, tokenizer = tokenizer, max_len = max_len, transform = torchvision_transform_test
)

# Define batch size and number of workers (adjust as needed)
batch_size = 32
num_workers = 4

# Create data loaders
trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
valloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)
testloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

class_names = train_dataset.classes
print(f"Label names: {class_names}", flush = True)
print(f"Train set: {len(trainloader)*batch_size}", flush = True)
print(f"Val set: {len(valloader)*batch_size}", flush = True)
print(f"Test set: {len(testloader)*batch_size}", flush = True)

model = GarbageModel(
    num_classes = 4
)
model.to(device)

criterion = nn.CrossEntropyLoss() # Loss function
optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5)
scheduler = ExponentialLR(optimizer, gamma=0.9)

nepochs = 30
PATH = './garbage_net.pth'

best_lost = 1e+20

print()

for epoch in range(nepochs):
    print(f"Epoch: {epoch+1}", flush = True)
    print('-'*10, flush = True)
    # Training Loop
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_num_total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data['image'].to(device), data['label'].to(device)
        input_ids, attention_mask = data['input_ids'].to(device), data['attention_mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, input_ids, attention_mask)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_correct += torch.sum(preds == labels.data)
        running_num_total += len(labels)
        
    print(f"Training loss: {running_loss / len(trainloader)}; Acc: {running_correct / running_num_total}", flush = True)
    
    # Validation Loop
    model.eval()
    val_loss = 0.0
    running_correct = 0
    running_num_total = 0
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            input_ids, attention_mask = data['input_ids'].to(device), data['attention_mask'].to(device)
            
            outputs = model(inputs, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            running_correct += torch.sum(preds == labels.data)
            running_num_total += len(labels)
            
    print(f"Validation loss: {val_loss / len(valloader)}; Acc: {running_correct / running_num_total}", flush = True)
    
    if val_loss < best_lost:
        best_lost = val_loss
        torch.save(model.state_dict(), PATH)

    print()

print('Finished Training', flush = True)