#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import Dataset
import torch.nn.init as init
import pandas as pd
import utm
import os
import random
from sklearn.preprocessing import MinMaxScaler

# Set script directory to working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))
#%%
class  TrainingDataset ():
    def __init__(self, tensor_size):
        def custom_min_max_scaling(dataframe, columns=None):
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
                numeric_columns = dataframe.select_dtypes(include=['float64']).columns
            else:
                numeric_columns = columns

            # Initialize the scaler
            scaler = MinMaxScaler()

            # Fit the scaler on the specified columns and transform the data
            dataframe[numeric_columns] = scaler.fit_transform(dataframe[numeric_columns])

            return dataframe, scaler

        self.tensor_size = tensor_size
        self.pm25, self.pm25_scaler = custom_min_max_scaling(pd.read_csv("PM25_final.csv"))
        self.modisaod, self.modisaod_scaler = custom_min_max_scaling(pd.read_csv("MAIACAOD.csv"))
        self.U_windspeed, self.U_windspeed_scaler = custom_min_max_scaling(pd.read_csv("U_windspeed.csv"))
        self.V_windspeed, self.V_windspeed_scaler = custom_min_max_scaling(pd.read_csv("V_windspeed.csv"))
        self.DewpointTemp, self.DewpointTemp_scaler = custom_min_max_scaling(pd.read_csv("DewpointTemp.csv"))
        self.Temp, self.Temp_scaler = custom_min_max_scaling(pd.read_csv("Temp.csv"))
        self.SurfPressure, self.SurfPressure_scaler = custom_min_max_scaling(pd.read_csv("SurfPressure.csv"))
        self.Precip, self.Precip_scaler = custom_min_max_scaling(pd.read_csv("Precip.csv"))
    
    def datasearch(self, df, lat, lon):
        lat_max = int(lat + self.tensor_size/2)
        lat_min = int(lat - self.tensor_size/2)
        lon_max = int(lon + self.tensor_size/2)
        lon_min = int(lon - self.tensor_size/2)

        filtered_data = df.iloc[lat_min:lat_max, lon_min:lon_max]

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


    def getdata(self, lat,lon):
        modisaod = self.datasearch(self.modisaod, lat,lon)
        U_windspeed = self.datasearch(self.U_windspeed, lat,lon)
        V_windspeed = self.datasearch(self.V_windspeed, lat,lon)
        DewpointTemp = self.datasearch(self.DewpointTemp, lat,lon)
        Temp = self.datasearch(self.Temp, lat,lon)
        SurfPressure = self.datasearch(self.SurfPressure, lat,lon)
        Precip = self.datasearch(self.Precip, lat,lon)
    
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

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        
        # Define the layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*2*2, 128)
        self.fc2 = nn.Linear(128, 1)
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
        #print(x.size())
        x = x.view(-1, 64*2*2)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.squeeze(1)
        return x

#%%
# Initialize U-Net model, loss function, and optimizer
in_channels = 7  # Number of input channels
num_epochs = 100
model = CNN(in_channels)

criterion = nn.MSELoss()

optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Adjust the learning rate as needed


for epoch in range(num_epochs):

    random_index = random.randint(0, len(dataset.pm25) - 1)
    row = dataset.pm25.iloc[random_index]

    model.train()

    inputs = dataset.getdata(int(row['row']),int(row['col']))
    targets = row['value']

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, torch.tensor(targets).float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()

    
    print(f"Epoch {epoch+1}, Loss: {loss}, Target: {targets}, Output: {outputs}")

# %%
