import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import time
import timm  # Import timm

from torchvision.transforms import Lambda

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(224),  # Resize to the input size expected by EfficientNet
    transforms.Grayscale(num_output_channels=3),  # Convert 1-channel grayscale to 3-channel grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for three channels
])


# Load data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)  # Adjusted batch size for GPU memory considerations
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet model
model = timm.create_model('efficientnet_b0', pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 10)  # Adjust the classifier

# Wrap the model with DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Move model to the device
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}.. Train loss: {running_loss/len(trainloader):.3f}")

# Evaluate the model
correct = 0
total = 0
model.eval()  # Evaluation mode
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

# Save the model
now = time.strftime("%Y%m%d_%H%M%S")
torch.save(model.module.state_dict(), f'./run/fashion_efficientnet_{now}.pth')
