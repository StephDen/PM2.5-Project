#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import Dataset
import pandas as pd
import utm
import os

# Set script directory to working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))
#%%
class  TrainingDataset ():
    def __init__(self, tensor_size):
        # Function to apply the conversion to each row
        def lat_lon_to_utm(row):
            utm_coords = utm.from_latlon(row["lat"],row["lon"])
            return utm_coords[0], utm_coords[1]
        # Return a new dataframe with the rounded UTM coordinates
        def lat_long_to_utm_df(df):
            df[['utm_easting', 'utm_northing']] = df.apply(lat_lon_to_utm, axis=1, result_type='expand')
            df["utm_easting"] = df["utm_easting"].round(-3)
            df["utm_northing"] = df["utm_northing"].round(-3)
            return df

        self.tensor_size = tensor_size

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

    # Convert UTM data to a tensor
    def datatotensor(self,df,easting_max,easting_min,northing_max,northing_min):
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

        return result_tensor
    def getmodisaod(self, northing,easting):
        modisaod = self.modisaod
        return self.datatotensor(modisaod,easting + self.tensor_size / 2,easting - self.tensor_size / 2,northing + self.tensor_size / 2,northing - self.tensor_size / 2)

    def getU_windspeed(self, northing,easting):
        U_windspeed = self.U_windspeed
        return self.datatotensor(U_windspeed, easting+self.tensor_size/2, easting-self.tensor_size/2, northing+self.tensor_size/2, northing-self.tensor_size/2)
    
    def getV_windspeed(self, northing,easting):
        V_windspeed = self.V_windspeed
        return self.datatotensor(V_windspeed, easting+self.tensor_size/2, easting-self.tensor_size/2, northing+self.tensor_size/2, northing-self.tensor_size/2)
    
    def getDewpointTemp(self, northing,easting):
        DewpointTemp = self.DewpointTemp
        return self.datatotensor(DewpointTemp, easting+self.tensor_size/2, easting-self.tensor_size/2, northing+self.tensor_size/2, northing-self.tensor_size/2)
    
    def getTemp(self, northing,easting):
        Temp = self.Temp
        return self.datatotensor(Temp, easting+self.tensor_size/2, easting-self.tensor_size/2, northing+self.tensor_size/2, northing-self.tensor_size/2)

    def getSurfPressure(self, northing,easting):
        SurfPressure = self.SurfPressure
        return self.datatotensor(SurfPressure, easting+self.tensor_size/2, easting-self.tensor_size/2, northing+self.tensor_size/2, northing-self.tensor_size/2)
    
    def getPrecip(self, northing,easting):
        Precip = self.Precip.copy()
        return  self.datatotensor(Precip, easting+self.tensor_size/2, easting-self.tensor_size/2, northing+self.tensor_size/2, northing-self.tensor_size/2)

    def getdata(self, northing,easting):
        modisaod = self.getmodisaod(northing,easting)
        U_windspeed = self.getU_windspeed(northing,easting)
        V_windspeed = self.getV_windspeed(northing,easting)
        DewpointTemp = self.getDewpointTemp(northing,easting)
        Temp = self.getTemp(northing,easting)
        SurfPressure = self.getSurfPressure(northing,easting)
        Precip = self.getPrecip(northing,easting)
    
        input_tensor = torch.stack((modisaod, U_windspeed, V_windspeed, DewpointTemp, Temp, SurfPressure, Precip), dim=0)
        return input_tensor

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
        self.fc1 = nn.Linear(625, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize U-Net model, loss function, and optimizer
in_channels = 7  # Number of input channels
tensor_size = 100  # Size of the input tensor
dataset = TrainingDataset(tensor_size)
#%%
model = CNN(in_channels)

criterion = nn.MSELoss()

optimizer = Adam(model.parameters(), lr=0.001)  # Adjust the learning rate as needed


for epoch, row in enumerate(dataset.pm25.iterrows()):

    model.train()

    inputs = dataset.getdata(row[1]['utm_northing'],row[1]['utm_easting'])
    targets = row[1]['value']

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")

# %%
