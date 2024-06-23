import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Define transformations with data augmentation for the training set
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define transformations for the validation and test sets
val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_dataset = torchvision.datasets.ImageFolder(root='/content/drive/MyDrive/Colab Notebooks/DataSet/train', transform=train_transform)
val_dataset = torchvision.datasets.ImageFolder(root='/content/drive/MyDrive/Colab Notebooks/DataSet/val', transform=val_test_transform)
test_dataset = torchvision.datasets.ImageFolder(root='/content/drive/MyDrive/Colab Notebooks/DataSet/test', transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Increased batch size
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1)  # Additional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.batch_norm5 = nn.BatchNorm2d(512)
        self.batch_norm6 = nn.BatchNorm2d(1024)
        self.batch_norm7 = nn.BatchNorm2d(2048)  # Additional batch norm
        self.fc1 = nn.Linear(2048 * 2 * 2, 512)  # Adjusted for new layer
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 5)  # 5 classes

    def forward(self, x):
        x = self.pool(self.batch_norm1(nn.ReLU()(self.conv1(x))))
        x = self.pool(self.batch_norm2(nn.ReLU()(self.conv2(x))))
        x = self.pool(self.batch_norm3(nn.ReLU()(self.conv3(x))))
        x = self.pool(self.batch_norm4(nn.ReLU()(self.conv4(x))))
        x = self.pool(self.batch_norm5(nn.ReLU()(self.conv5(x))))
        x = self.pool(self.batch_norm6(nn.ReLU()(self.conv6(x))))
        x = self.pool(self.batch_norm7(nn.ReLU()(self.conv7(x))))  # Additional layer forward
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = ImprovedCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to plot the performance metrics
def plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_accuracies, 'b', label='Validation accuracy')
    plt.title('Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_precisions, 'b', label='Validation precision')
    plt.plot(epochs, val_recalls, 'g', label='Validation recall')
    plt.plot(epochs, val_f1_scores, 'r', label='Validation F1 score')
    plt.title('Validation precision, recall, and F1 score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Adjusted training function to track and plot metrics
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25):
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        val_precision = precision_score(all_labels, all_preds, average='weighted')
        val_recall = recall_score(all_labels, all_preds, average='weighted')
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%, Val Precision: {val_precision:.4f}, '
              f'Val Recall: {val_recall:.4f}, Val F1 Score: {val_f1:.4f}')

    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores)

# Use the updated train_model function
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f'Test Accuracy: {accuracy:.2f}%, Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, '
          f'Test F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')

# Evaluate the model on the test set
evaluate_model(model, test_loader)

# Create directory if it doesn't exist
save_dir = '/content/drive/MyDrive/DataSet/CNN_Model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the trained model
torch.save(model.state_dict(), os.path.join(save_dir, 'improved_cnn.pth'))
