#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import Dataset
import pandas as pd
import os
import utm
# Set script directory to working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

#%%
class  TrainingDataset (Dataset):
    def __init__(self, tensor_size,train=True):
        ## Function to apply the conversion to each row
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
        self.pm25 = lat_long_to_utm_df(pd.read_csv("PM25.csv")[["value","lat","lon"]])
        self.modisaod = lat_long_to_utm_df(pd.read_csv("MAIACAOD.csv")[["lat","lon","value"]])
        #self.era5 = pd.read_csv("ERA5.csv")
        self.train = train

    # Convert UTM data to a tensor
    def datatotensor(df,easting_max,easting_min,northing_max,northing_min):
        # Filter the dataframe to the specified easting and northing ranges
        filtered_data = df[
            (df['utm_easting'] >= easting_min) & (df['utm_easting'] <= easting_max) &
            (df['utm_northing'] >= northing_min) & (df['utm_northing'] <= northing_max)
        ]

        kilometer_spacing = (3000 - 0) / 3000  # Assuming a regular grid
        result_tensor = torch.zeros((tensor_size,tensor_size),dtype=torch.float32)
        # Populate the tensor with values from sparse data
        for index, row in df.iterrows():
            easting_index = int((row['utm_easting'] - easting_min)/kilometer_spacing)
            northing_index = int((row['utm_northing'] - northing_min)/kilometer_spacing)
            
            # Ensure that the indices are within the tensor size
            if 0 <= easting_index < self.tensor_size and 0 <= northing_index < self.tensor_size:
                result_tensor[easting_index, northing_index] = row['value']

        return result_tensor
    def getmodisaod(self, northing,easting):
        return datatotensor(self.modisaod, easting+self.tensor_size/2, easting-self.tensor_size/2, northing+self.tensor_size/2, northing-self.tensor_size/2)
    def getPM2_5(self, northing,easting):
        return datatotensor(self.pm25, easting+self.tensor_size/2, easting-self.tensor_size/2, northing+self.tensor_size/2, northing-self.tensor_size/2)

    def getdata(self, northing,easting):
        modisaod = self.getmodisaod(northing,easting)
        PM2_5 = self.getPM2_5(northing,easting)
        input_tensor = torch.cat([modisaod.unsqueeze(2)], dim=2)
        target_tensor = torch.cat([pm2_5.unsqueeze(0)], dim=0)
        return input_tensor, target_tensor

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels,tensor_size):
        super(UNet, self).__init__()
        self.dim = tensor_size
        # Encoder
        self.encoder1 = self.double_conv(in_channels, dim)
        self.encoder2 = self.double_conv(dim, dim*2)
        self.encoder3 = self.double_conv(dim*2, dim*4)
        self.encoder4 = self.double_conv(dim*4, dim*8)
        
        # Decoder
        self.decoder1 = self.double_conv(dim*8, dim*4)
        self.decoder2 = self.double_conv(dim*4, dim*2)
        self.decoder3 = self.double_conv(dim*2, dim)
        
        # Up-sampling
        self.upconv1 = nn.ConvTranspose2d(dim*8, dim*4, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(dim*4, dim*2, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2)
        
        # Output layer
        self.outconv = nn.Conv2d(dim, out_channels, kernel_size=1)
    
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

# Initialize U-Net model, loss function, and optimizer
in_channels = 4  # Number of input channels
out_channels = 1  # Number of output channels
tensor_size = 3000  # Size of the input tensor
dataset = TrainingDataset(tensor_size)
model = UNet(in_channels, out_channels)

criterion = nn.MSELoss()

optimizer = Adam(model.parameters(), lr=0.001)  # Adjust the learning rate as needed


for epoch, row in enumerate(dataset.pm25.iterrows()):

    model.train()

    inputs, targets = dataset.getdata(row['utm_northing'],row['utm_easting'])

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")
