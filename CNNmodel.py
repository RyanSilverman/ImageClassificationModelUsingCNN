# Import the necessary libraries
import os

import torch
import torchvision.transforms as transforms
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Defining the CNN architecture
class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(8),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = Sequential(
            Linear(8 * 7 * 7, 6) # Spatial dimensions of output feature maps reduced by 2 every convolution (28/4=7)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
# End

# # Train, Test, and Validation data directories
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# Load the train, test, and validation data
train_dataset = ImageFolder(train_dir, transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
]))
val_dataset = ImageFolder(val_dir, transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
]))
test_dataset = ImageFolder(test_dir, transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
]))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# defining the model
model = Net()
print(model)
# checking if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    model = model.cuda()


# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.001)
# defining the loss function
criterion = CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data

            inputs = inputs.to(device)  # Move inputs to the desired device
            labels = labels.to(device)  # Move labels to the desired device

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

    return running_loss/len(loader), correct/total

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

running_loss = 0.0
total = 0.0
MAX_EPOCHS = 15

for epoch in range(MAX_EPOCHS):
    print("Starting Epoch: {}".format(epoch+1))
    for i, data in enumerate(train_loader, 0):
        model.train()

        inputs, labels = data
        inputs = inputs.to(device)  # Move inputs to the desired device
        labels = labels.to(device)  # Move labels to the desired device
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if(i % 10 == 9):
            mean_loss = running_loss/10

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            train_acc = correct/labels.size(0)

            history['train_loss'].append(mean_loss)
            history['train_acc'].append(train_acc)

            print('# mini-batch {}\ntrain loss: {} train accuracy: {}'.format(i+1, mean_loss, train_acc))
            running_loss = 0.0

            mean_loss, val_acc = evaluate(model, val_loader)
            history['val_loss'].append(mean_loss)
            history['val_acc'].append(val_acc)

            print("validation loss: {} validation accuracy: {}\n".format(mean_loss, val_acc))
print("Finished training")


# Save the model with the current date and time as the filename
torch.save(model.state_dict(), 'model.pth')

# ---------------LOSS--------------
fig = plt.figure(figsize=(8,8))
plt.plot(history['train_loss'], label = 'train_loss')
plt.plot(history['val_loss'], label = 'val_loss')
plt.xlabel("Logging iterations")
plt.ylabel("Cross-entropy loss")
plt.legend()
plt.savefig("LossFunction.png")
#---------------ACC------------------
fig = plt.figure(figsize=(8,8))
plt.plot(history['train_acc'], label = 'train_acc')
plt.plot(history['val_acc'], label = 'val_acc')
plt.xlabel("Logging iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("Accuracy.png")
#-------------------------------------

_, test_acc = evaluate(model, test_loader)
print("Test accuracy = {}".format(test_acc))

x = torch.stack([image for image, _ in test_dataset])
y = torch.tensor([label for _, label in test_dataset])

outputs = model(x.to(device))
_, y_pred = torch.max(outputs, 1)

cm = confusion_matrix(y.numpy(), (y_pred.cpu()).numpy(), normalize="true")

plt.figure(figsize=(8, 6))
class_names = ['circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle']
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('Confusion_Matrix.png')


def predict_class(image_path, model):
    # Load the image and apply the same transforms as used for training data
    image = Image.open(image_path)

    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image_tensor = data_transforms(image).unsqueeze(0)
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the desired device
    model.to(device)
    # Pass the transformed image through the trained model to get the predicted class

    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

def test_own_images(folder_path):
    # Create an instance of the model and load the saved model state dictionary
    model = Net()
    model.load_state_dict(torch.load('model.pth'))
    class_names = ['circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle']

    # Get a list of image file paths in the folder
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]

    # Predict the class for each image
    predicted_classes = []
    for image_file in image_files:
        predicted_class = predict_class(image_file, model)
        predicted_class_name = class_names[predicted_class]
        predicted_classes.append(predicted_class_name)

    return predicted_classes, [os.path.basename(file) for file in image_files]


# Test the script with images in the personal_drawing folder
folder_path = "dataset/personal_drawing"
predicted_classes, file_names = test_own_images(folder_path)

# Print the predicted classes with file names
for file_name, predicted_class in zip(file_names, predicted_classes):
    print(f"File Name: {file_name}\t\tPredicted Class: {predicted_class}")


