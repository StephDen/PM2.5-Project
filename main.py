#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import Dataset
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.transforms import functional as TF
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
        def custom_min_max_scaling(dataframes, columns):
            """
            Normalize specified columns of a DataFrame using Min-Max scaling.

            Parameters:
            - dataframe (pd.DataFrame): The DataFrame containing the data.
            - columns (list): List of columns to be normalized. If None, normalize all numeric columns.

            Returns:
            - pd.DataFrame: The DataFrame with normalized values.
            - MinMaxScaler: The fitted scaler object.
            """

            # Initialize the scaler
            scaler = MinMaxScaler()

            # Apply min-max scaling to each DataFrame in the list
            scaled_dataframes = [pd.DataFrame(scaler.fit_transform(df, columns), columns=df.columns) for df in dataframes]

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
                file_path = os.path.join(folder, file_name)
                df = pd.read_csv(file_path)
                dataframes.append(df)
            return dataframes
        
        root_folder = "data"
        self.tensor_size = tensor_size
        self.pm25, self.pm25_scaler = custom_min_max_scaling(read_csvs(root_folder, "PM25.csv"), columns=['value'])
        self.modisaod, self.modisaod_scaler = custom_min_max_scaling(read_csvs(root_folder, "AODregrid.csv")) #size: 47,48
        self.U_windspeed, self.U_windspeed_scaler = custom_min_max_scaling(read_csvs(root_folder, "U_windspeed.csv"))
        self.V_windspeed, self.V_windspeed_scaler = custom_min_max_scaling(read_csvs(root_folder, "V_windspeed.csv"))
        self.DewpointTemp, self.DewpointTemp_scaler = custom_min_max_scaling(read_csvs(root_folder, "DewpointTemp.csv"))
        self.Temp, self.Temp_scaler = custom_min_max_scaling(read_csvs(root_folder, "Temp.csv"))
        self.SurfPressure, self.SurfPressure_scaler = custom_min_max_scaling(read_csvs(root_folder, "SurfPressure.csv"))
        self.Precip, self.Precip_scaler = custom_min_max_scaling(read_csvs(root_folder, "Precip.csv"))
        # self.pm25, self.pm25_scaler = custom_min_max_scaling(pd.read_csv("PM25_final.csv"), columns=['value'])
        # self.modisaod, self.modisaod_scaler = custom_min_max_scaling(pd.read_csv("AODregrid.csv")) #size: 47,48
        # self.U_windspeed, self.U_windspeed_scaler = custom_min_max_scaling(pd.read_csv("U_windspeed.csv"))
        # self.V_windspeed, self.V_windspeed_scaler = custom_min_max_scaling(pd.read_csv("V_windspeed.csv"))
        # self.DewpointTemp, self.DewpointTemp_scaler = custom_min_max_scaling(pd.read_csv("DewpointTemp.csv"))
        # self.Temp, self.Temp_scaler = custom_min_max_scaling(pd.read_csv("Temp.csv"))
        # self.SurfPressure, self.SurfPressure_scaler = custom_min_max_scaling(pd.read_csv("SurfPressure.csv"))
        # self.Precip, self.Precip_scaler = custom_min_max_scaling(pd.read_csv("Precip.csv"))
        
    
    def datasearch(self, df, lat, lon):
        lat_max = int(lat + self.tensor_size/2)
        lat_min = int(lat - self.tensor_size/2)
        lon_max = int(lon + self.tensor_size/2)
        lon_min = int(lon - self.tensor_size/2)

        filtered_data = df.iloc[lat_min:lat_max, lon_min:lon_max] #breaks at this line

        result_tensor = torch.zeros((self.tensor_size, self.tensor_size), dtype=torch.float32)
        
        # Check if filtered_data has at least one row and one column
        if len(filtered_data) > 0 and len(filtered_data.columns) > 0:
            for i in range(lat_min, lat_max):
                if (i > -1 and i < len(filtered_data)):
                    for j in range(lon_min, lon_max):
                        if (j > -1 and j < len(filtered_data.columns)):
                            # Convert the value to a numeric type (float)
                            result_tensor[i - lat_min, j - lon_min] = pd.to_numeric(filtered_data.iloc[i - lat_min, j - lon_min], errors='coerce')
        
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
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Sequential(
            nn.Linear(64*2*2, 128),
            nn.LeakyReLU(negative_slope=0.01),
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
# Initialize CNN model, loss function, and optimizer
in_channels = 7  # Number of input channels
num_epochs = 100
model = SimpleCNN(in_channels)

criterion = nn.MSELoss()

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # Adjust the learning rate as needed

losses = []
for epoch in range(num_epochs):
    random_folder = random.randint(0, len(dataset.pm25) - 1)
    random_index = random.randint(0, len(dataset.pm25) - 1)
    row = dataset.pm25.iloc[random_index]

    model.train()

    inputs = dataset.getdata(random_folder,int(row['row']),int(row['col']))
    targets = row['value']

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, torch.tensor(targets).float())
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

    losses.append(loss.item())
    
    print(f"Epoch {epoch+1}, Loss: {loss}, Target: {targets}, Output: {outputs}")

plot_loss(losses)
# %%
### Plot PM2.5 predictions 

outputs = []
for i in range(0,47):
    for j in range(0,47):
        inputs = dataset.getdata(i, j)
        tempout = model(inputs).detach().numpy()
        tempout = dataset.pm25_scaler.inverse_transform(tempout.reshape(1, -1))
        outputs.append(tempout)
        
PMoutputs = np.array(outputs).reshape(47,47)        

f, ax = plt.subplots(1, 1, figsize=(4, 4))
img = ax.contourf(PMoutputs, cmap="RdYlBu_r")
f.colorbar(img)

# %%