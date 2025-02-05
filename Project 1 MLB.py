import torch
from torchvision import transforms

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(nn.ReLU()(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x

from tqdm import tqdm
from sklearn.metrics import f1_score

def process_forward_pass(model, batch, criterion):
    images, labels = batch
    labels = labels.float()

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs.squeeze(), labels)
    preds = (outputs > 0.5).float() 
    return loss, preds, labels

def train_and_eval(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs): 
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
    
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            optimizer.zero_grad()
            loss, preds, labels = process_forward_pass(model, batch, criterion)
    
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
    
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
    
        with torch.no_grad(): 
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                loss, preds, labels = process_forward_pass(model, batch, criterion)
    
                val_loss += loss.item()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
    
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
    
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, F1 Score: {train_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, F1 Score: {val_f1:.4f}")
