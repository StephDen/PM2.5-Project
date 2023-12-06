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

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder1 = self.double_conv(in_channels, 64)
        self.encoder2 = self.double_conv(64, 128)
        self.encoder3 = self.double_conv(128, 256)
        self.encoder4 = self.double_conv(256, 512)
        
        # Decoder
        self.decoder1 = self.double_conv(512, 256)
        self.decoder2 = self.double_conv(256, 128)
        self.decoder3 = self.double_conv(128, 64)
        
        # Up-sampling
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Output layer
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Decoder with skip connections
        dec1 = self.upconv1(enc4)
        dec1 = torch.cat([dec1, enc3], dim=1)
        dec1 = self.decoder1(dec1)
        
        dec2 = self.upconv2(dec1)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec3 = self.upconv3(dec2)
        dec3 = torch.cat([dec3, enc1], dim=1)
        dec3 = self.decoder3(dec3)
        
        # Output
        output = self.outconv(dec3)
        return output
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

# Define dataset and DataLoader
train_dataset =  TrainingDataset(train=True)
test_dataset =  TrainingDataset(train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize U-Net model, loss function, and optimizer
in_channels = 4  # Number of input channels
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



