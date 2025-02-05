import torch
from torchvision import transforms

# For training, we will apply random horizontal flip, random rotation, normalization, and conversion to tensor
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(20),  # Randomly rotate the image by up to 20 degrees
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# For testing, we will only normalize the images and convert them to tensor
transform_test = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Assuming input image size is 128x128
        self.fc2 = nn.Linear(512, 1)  # Binary classification (Pneumonia vs. No Pneumonia)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        
        # Flatten the output of the convolutional layers
        x = x.view(-1, 128 * 8 * 8)  # Flatten to feed into the fully connected layers
        x = self.dropout(nn.ReLU()(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary classification
        return x

from tqdm import tqdm
from sklearn.metrics import f1_score

def process_forward_pass(model, batch, criterion):
    images, labels = batch
    labels = labels.float()

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs.squeeze(), labels)  # Squeeze labels for binary classification
    preds = (outputs > 0.5).float()  # Convert to 0 or 1 based on the threshold 0.5
    return loss, preds, labels

def train_and_eval(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs): 
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
    
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            optimizer.zero_grad()
            loss, preds, labels = process_forward_pass(model, batch, criterion)
    
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
        # Compute training loss, accuracy, and f1 score
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
    
        # Validation phase
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
    
        # Compute validation loss, accuracy, and f1 score
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
    
        # Visualize results
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, F1 Score: {train_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, F1 Score: {val_f1:.4f}")
