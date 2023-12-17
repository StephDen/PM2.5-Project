#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import Dataset
import pandas as pd
import utm
import os
import random

# Set script directory to working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))
#%%
class  TrainingDataset ():
    def __init__(self, tensor_size):
        
        """ # Function to apply the conversion to each row
        def lat_lon_to_utm(row):
            utm_coords = utm.from_latlon(row["lat"],row["lon"])
            return utm_coords[0], utm_coords[1]
        # Return a new dataframe with the rounded UTM coordinates
        def lat_long_to_utm_df(df):
            df[['utm_easting', 'utm_northing']] = df.apply(lat_lon_to_utm, axis=1, result_type='expand')
            df["utm_easting"] = df["utm_easting"].round(-3)
            df["utm_northing"] = df["utm_northing"].round(-3)
            return df
        #pm25
        self.pm25 = lat_long_to_utm_df(pd.read_csv("PM25.csv")[["value","lat","lon"]])
        #modisaod
        self.modisaod = lat_long_to_utm_df(pd.read_csv("MAIACAOD.csv")[["lat","lon","value"]])
        #era5
        era5 = pd.read_csv("ERA5data.csv")
        self.U_windspeed = lat_long_to_utm_df(era5[["lat","lon","U_windspeed"]])
        self.V_windspeed = lat_long_to_utm_df(era5[["lat","lon","V_windspeed"]])
        self.DewpointTemp = lat_long_to_utm_df(era5[["lat","lon","DewpointTemp"]])
        self.Temp = lat_long_to_utm_df(era5[["lat","lon","Temp"]])
        self.SurfPressure = lat_long_to_utm_df(era5[["lat","lon","SurfPressure"]])
        self.Precip = lat_long_to_utm_df(era5[["lat","lon","Precip"]])
        """
        self.tensor_size = tensor_size
        self.pm25 = pd.read_csv("PM25.csv")
        self.modisaod = pd.read_csv("MAIACAOD.csv")
        self.ERA5data = pd.read_csv("ERA5data.csv")
        self.U_windspeed



    # Convert UTM data to a tensor
"""     def datatotensor(self,df,easting_max,easting_min,northing_max,northing_min):
        # Filter the dataframe to the specified easting and northing ranges
        filtered_data = df[
            (df['utm_easting'] >= easting_min) & (df['utm_easting'] <= easting_max) &
            (df['utm_northing'] >= northing_min) & (df['utm_northing'] <= northing_max)
        ]

        kilometer_spacing = 1  # Assuming a regular grid
        result_tensor = torch.zeros((tensor_size,tensor_size),dtype=torch.float32)
        # Populate the tensor with values from sparse data
        for index, row in filtered_data.iterrows():
            easting_index = int((row['utm_easting'] - easting_min)/kilometer_spacing)
            northing_index = int((row['utm_northing'] - northing_min)/kilometer_spacing)
            
            # Ensure that the indices are within the tensor size
            if 0 <= easting_index < self.tensor_size and 0 <= northing_index < self.tensor_size:
                result_tensor[easting_index, northing_index] = row['value']

        return result_tensor """
    def datasearch(self, df, lat,lon):
        lat_max = lat + self.tensor_size/2
        lat_min = lat - self.tensor_size/2
        lon_max = lon + self.tensor_size/2
        lon_min = lon - self.tensor_size/2

        filtered_data = df.iloc[lat_min:lat_max,lon_min:lon_max,]

        result_tensor = torch.zeros((tensor_size,tensor_size),dtype=torch.float32)
        for lat in range(lat_min, lat_max):
            for lon in range(lon_min, lon_max):
                result_tensor[lat, lon] = filtered_data[lat, lon]
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
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*2*2, 128)
        self.fc2 = nn.Linear(128, 1)

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
num_epochs = 1000
model = CNN(in_channels)

criterion = nn.MSELoss()

optimizer = Adam(model.parameters(), lr=0.001)  # Adjust the learning rate as needed


for epoch in range(num_epochs):

    random_index = random.randint(0, len(dataset.pm25) - 1)
    row = dataset.pm25.iloc[random_index]

    model.train()

    inputs = dataset.getdata(row['utm_northing'],row['utm_easting'])
    targets = row['value']

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, torch.tensor(targets).float())
    loss.backward()
    optimizer.step()

    
    print(f"Epoch {epoch+1}, Loss: {loss}")

# %%
