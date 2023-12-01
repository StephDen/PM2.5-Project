import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import Dataset

class  TrainingDataset (Dataset):
    def __init__(self, train=True):
        self.data = ...  # Your data loading logic here
        self.targets = ...  # Your target loading logic here
        self.train = train

    def __len__(self):
        # Return the number of samples indataset
        return len(self.data)

    def __getitem__(self, index):
        # Return a single sample and its corresponding target
        sample = self.data[index]
        target = self.targets[index]

        # You may need to preprocess the data and target here
        # e.g., convert to PyTorch tensors

        return sample, target

# Define U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Mid part
        self.mid = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)

        # Mid part
        x2 = self.mid(x1)

        # Decoder
        x3 = self.decoder(x2)

        return x3

# Example usage
in_channels = 3  # number of input channels
out_channels = 1  # number of output channels 
model = UNet(in_channels, out_channels)

# Print the model architecture
print(model)

# Define dataset and DataLoader
train_dataset =  TrainingDataset(train=True)
test_dataset =  TrainingDataset(train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize U-Net model, loss function, and optimizer
in_channels = 3  # Number of input channels
out_channels = 1  # Number of output channels
model = UNet(in_channels, out_channels)

criterion = nn.CrossEntropyLoss()  # Use appropriate loss function for   task

optimizer = Adam(model.parameters(), lr=0.001)  # Adjust the learning rate as needed

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")

# Evaluation loop
model.eval()
with torch.no_grad():
    test_loss = 0

    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

    average_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {average_test_loss}")



