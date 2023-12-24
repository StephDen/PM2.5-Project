#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import Dataset
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import torchvision
import pandas as pd
import os
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss_values):
    """
    Function to plot loss per epoch.

    Parameters:
    - loss_values (list): List of loss values for each epoch.
    """
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.show()

# Set script directory to working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))
#%%
class  TrainingDataset ():
    def __init__(self, tensor_size):
        def custom_min_max_scaling(dataframes, columns = None):
            """
            Normalize specified columns of a DataFrame using Min-Max scaling.

            Parameters:
            - dataframe (pd.DataFrame): The DataFrame containing the data.
            - columns (list): List of columns to be normalized. If None, normalize all numeric columns.

            Returns:
            - pd.DataFrame: The DataFrame with normalized values.
            - MinMaxScaler: The fitted scaler object.
            """
            if columns is None:
                # If columns are not specified, normalize all numeric columns
                numeric_columns = dataframes[0].select_dtypes(include=['float64']).columns
            else:
                numeric_columns = columns

            # Concatenate only the numeric columns of all DataFrames in the list
            combined_numeric_df = pd.concat([df[numeric_columns] for df in dataframes], ignore_index=True)

            # Initialize the scaler and fit it to the combined numeric data
            scaler = MinMaxScaler()
            scaler.fit(combined_numeric_df)

            
            # Apply min-max scaling to each DataFrame in the list
            #scaled_dataframes = [pd.DataFrame(scaler.transform(df[numeric_columns]), columns=numeric_columns) for df in dataframes]
            scaled_dataframes = []
            for df in dataframes:
                # Extract non-numeric columns
                non_numeric_columns = df.drop(columns=numeric_columns)
                
                # Scale numeric columns
                scaled_numeric_columns = pd.DataFrame(scaler.fit_transform(df[numeric_columns]), columns=numeric_columns)
                
                # Concatenate scaled numeric columns with non-numeric columns
                scaled_dataframe = pd.concat([non_numeric_columns, scaled_numeric_columns], axis=1)
                
                # Append the result to the list
                scaled_dataframes.append(scaled_dataframe)
            return scaled_dataframes, scaler
        def read_csvs(file_name, directory_path = "data"):
            """
            Read all csv files in a directory and return a list of dataframes.
            """
            # Get a list of all files in the directory
            files_folders = os.listdir(directory_path)
            # get only folders
            folders = [entry for entry in files_folders if os.path.isdir(os.path.join(directory_path, entry))]
            dataframes = []
            for folder in folders:
                file_path = os.path.join(directory_path, folder, file_name)
                print(file_path)
                df = pd.read_csv(file_path, index_col=0)
                df = df.replace(np.nan, 0)  # Replace NaN values with 0
                dataframes.append(df)
            return dataframes
        
        root_folder = "data"
        self.tensor_size = tensor_size
        self.pm25, self.pm25_scaler = custom_min_max_scaling(read_csvs("PM25.csv",root_folder), columns=['value'])
        self.modisaod, self.modisaod_scaler = custom_min_max_scaling(read_csvs("AODregrid.csv",root_folder)) #size: 47,48
        self.U_windspeed, self.U_windspeed_scaler = custom_min_max_scaling(read_csvs("U_windspeed.csv",root_folder))
        self.V_windspeed, self.V_windspeed_scaler = custom_min_max_scaling(read_csvs("V_windspeed.csv",root_folder))
        self.DewpointTemp, self.DewpointTemp_scaler = custom_min_max_scaling(read_csvs("DewpointTemp.csv",root_folder))
        self.Temp, self.Temp_scaler = custom_min_max_scaling(read_csvs("Temp.csv",root_folder))
        self.SurfPressure, self.SurfPressure_scaler = custom_min_max_scaling(read_csvs("SurfPressure.csv",root_folder))
        self.Precip, self.Precip_scaler = custom_min_max_scaling(read_csvs("Precip.csv",root_folder))
        # self.pm25, self.pm25_scaler = custom_min_max_scaling(pd.read_csv("PM25_final.csv"), columns=['value'])
        # self.modisaod, self.modisaod_scaler = custom_min_max_scaling(pd.read_csv("AODregrid.csv")) #size: 47,48
        # self.U_windspeed, self.U_windspeed_scaler = custom_min_max_scaling(pd.read_csv("U_windspeed.csv"))
        # self.V_windspeed, self.V_windspeed_scaler = custom_min_max_scaling(pd.read_csv("V_windspeed.csv"))
        # self.DewpointTemp, self.DewpointTemp_scaler = custom_min_max_scaling(pd.read_csv("DewpointTemp.csv"))
        # self.Temp, self.Temp_scaler = custom_min_max_scaling(pd.read_csv("Temp.csv"))
        # self.SurfPressure, self.SurfPressure_scaler = custom_min_max_scaling(pd.read_csv("SurfPressure.csv"))
        # self.Precip, self.Precip_scaler = custom_min_max_scaling(pd.read_csv("Precip.csv"))
        
    
    def datasearch(self, df, lat, lon):
        lat_min = int(lat - self.tensor_size/2)
        lon_min = int(lon - self.tensor_size/2)

        result_tensor = torch.zeros((self.tensor_size, self.tensor_size), dtype=torch.float32)
        for i in range(0, tensor_size):
            for j in range(0, tensor_size):
                if (i+lat_min < df.shape[0] and j+lon_min < df.shape[1]) and (i+lat_min >= 0 and j+lon_min >= 0):
                    result_tensor[i, j] = pd.to_numeric(df.iloc[i + lat_min, j + lon_min], errors='coerce')
        #print(result_tensor)
        return result_tensor


    def getdata(self,folder, lat,lon):
        modisaod = self.datasearch(self.modisaod[folder], lat,lon)
        U_windspeed = self.datasearch(self.U_windspeed[folder], lat,lon)
        V_windspeed = self.datasearch(self.V_windspeed[folder], lat,lon)
        DewpointTemp = self.datasearch(self.DewpointTemp[folder], lat,lon)
        Temp = self.datasearch(self.Temp[folder], lat,lon)
        SurfPressure = self.datasearch(self.SurfPressure[folder], lat,lon)
        Precip = self.datasearch(self.Precip[folder], lat,lon)
    
        input_tensor = torch.stack((modisaod, U_windspeed, V_windspeed, DewpointTemp, Temp, SurfPressure, Precip), dim=0)
        return input_tensor


tensor_size = 10  # Size of the input tensor
dataset = TrainingDataset(tensor_size)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels,tensor_size):
        super(UNet, self).__init__()
        # Encoder
        self.encoder1 = self.double_conv(in_channels, tensor_size)
        self.encoder2 = self.double_conv(tensor_size, tensor_size*2)
        self.encoder3 = self.double_conv(tensor_size*2, tensor_size*4)
        self.encoder4 = self.double_conv(tensor_size*4, tensor_size*8)
        
        # Decoder
        self.decoder1 = self.double_conv(tensor_size*8, tensor_size*4)
        self.decoder2 = self.double_conv(tensor_size*4, tensor_size*2)
        self.decoder3 = self.double_conv(tensor_size*2, tensor_size)
        
        # Up-sampling
        self.upconv1 = nn.ConvTranspose2d(tensor_size*8, tensor_size*4, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(tensor_size*4, tensor_size*2, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(tensor_size*2, tensor_size, kernel_size=2, stride=2)
        
        # Output layer
        self.outconv = nn.Conv2d(tensor_size, out_channels, kernel_size=1)
    
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

#%%
# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, in_channels):
        super(SimpleCNN, self).__init__()
        size = 32
        # Define the layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Sequential(
            nn.Linear(64*2*2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 64*2*2)
        x = self.fc(x)
        x = torch.sigmoid(x)
        x = x.squeeze(1)
        return x
class AdvancedCNN(nn.Module):
    def __init__(self, in_channels):
        super(AdvancedCNN, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 1)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #x = x.view(128)  # Flatten before fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x.squeeze(1)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Modified the first convolution layer to accept 7x10x10 input
        self.conv1 = nn.Conv2d(7, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Use global average pooling
        out = F.adaptive_avg_pool2d(out, (1, 1))
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18_7Channels():
    return ResNet(BasicBlock, [2,2,2,2])
# Initialize CNN model, loss function, and optimizer
in_channels = 7  # Number of input channels
num_epochs = 10000
model = SimpleCNN(in_channels)

criterion = nn.MSELoss()

optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Adjust the learning rate as needed

losses = []
for epoch in range(num_epochs):
    random_folder = random.randint(0, len(dataset.pm25) - 1)
    random_index = random.randint(0, len(dataset.pm25[random_folder]) - 1)
    row = dataset.pm25[random_folder].iloc[random_index]

    model.train()

    inputs = dataset.getdata(random_folder,int(row['row']),int(row['col']))
    inputs = torch.unsqueeze(inputs, 0)
    #print(inputs)
    targets = row['value']

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, torch.tensor(targets).float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

    losses.append(loss.item())
    
    print(f"Epoch {epoch+1}, Loss: {loss}, Target: {targets}, Output: {outputs}")

plot_loss(losses)
# %%
### Plot PM2.5 predictions 

outputs = []
for i in range(0,46):
    for j in range(0,46):
        inputs = dataset.getdata(28,i, j)
        tempout = model(inputs).detach().numpy()
        tempout = dataset.pm25_scaler.inverse_transform(tempout.reshape(1, -1))
        outputs.append(tempout)
        
PMoutputs = np.array(outputs).reshape(46,46)        
#%%

# Set specific minimum and maximum contour levels
min_contour_level = 0
max_contour_level = 100

# Create a contour plot with specific levels
levels = np.linspace(min_contour_level, max_contour_level, 101)  # 101 levels from 0 to 100
f, ax = plt.subplots(1, 1, figsize=(4, 4))
img = ax.contourf(PMoutputs, levels=levels, cmap="RdYlBu_r")
f.colorbar(img)

# %%