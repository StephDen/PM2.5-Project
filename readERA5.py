### 555b Final Project
### Read and format ERA5 grib files

### Where to get ERA5 meteorological reanalysis data: 
    ### https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form
    
### Reference material: https://jswhit.github.io/pygrib/api.html
    ### https://docs.xarray.dev/en/stable/examples/ERA5-GRIB-example.html 

### ERA5 is in UTC. MODIS is 18-21 local time. PST = UTC-8, so the time steps we are interested in here are 23-02 next day
### We will get just the time step at hour 23 for this exercise

import numpy as np
import pandas as pd
# from glob import glob
# import cfgrib
# import xarray as xr
import pygrib

file = r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\WA_OR_07182021.grib"
grbs = pygrib.open(file)

# Look for variable names
grbs.seek(0)
for grb in grbs:
    grb

# Select values for the time step of interest for all variables    
u_grb = grbs.select(name='10 metre U wind component')[23].values
v_grb = grbs.select(name='10 metre V wind component')[23].values
Td_grb = grbs.select(name='2 metre dewpoint temperature')[23].values
T_grb = grbs.select(name='2 metre temperature')[23].values
P_grb = grbs.select(name='Surface pressure')[23].values
precip_grb = grbs.select(name='Total precipitation')[23].values

# Get lat lons of the grid 
lats, lons = grb.latlons()
lats.shape, lats.min(), lats.max(), lons.shape, lons.min(), lons.max()

# Cut grid to lat: 42.3 to 46.9, lon: -121.8 to -117.2 (inclusive of these values)
#this is just for reference:
lat_min = 42.3
lat_max = 46.9
lon_min = -121.8
lon_max = -117.2

lats = lats[18:,18:]
lons = lons[18:,18:]
u_grb = u_grb[18:,18:]
v_grb = v_grb[18:,18:]
Td_grb = Td_grb[18:,18:]
T_grb = T_grb[18:,18:]
P_grb = P_grb[18:,18:]
precip_grb = precip_grb[18:,18:]

lats = pd.DataFrame(lats[3:50,4:51])
lons = pd.DataFrame(lons[3:50,4:51])
u_grb = pd.DataFrame(u_grb[3:50,4:51])
v_grb = pd.DataFrame(v_grb[3:50,4:51])
Td_grb = pd.DataFrame(Td_grb[3:50,4:51])
T_grb = pd.DataFrame(T_grb[3:50,4:51])
P_grb = pd.DataFrame(P_grb[3:50,4:51])
precip_grb = pd.DataFrame(precip_grb[3:50,4:51])


lats.to_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\lats.csv")
lons.to_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\lons.csv")
u_grb.to_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\U_windspeed.csv")
v_grb.to_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\V_windspeed.csv")
T_grb.to_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\Temp.csv")
Td_grb.to_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\DewpointTemp.csv")
P_grb.to_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\SurfPressure.csv")
precip_grb.to_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\Precip.csv")



# #cut and flatten data to put in dataframe
# lats = pd.DataFrame(np.ravel(lats[18:,18:]))
# lons = pd.DataFrame(np.ravel(lons[18:,18:]))
# u_grb = pd.DataFrame(np.ravel(u_grb[18:,18:]))
# v_grb = pd.DataFrame(np.ravel(v_grb[18:,18:]))
# Td_grb = pd.DataFrame(np.ravel(Td_grb[18:,18:]))
# T_grb = pd.DataFrame(np.ravel(T_grb[18:,18:]))
# P_grb = pd.DataFrame(np.ravel(P_grb[18:,18:]))
# precip_grb = pd.DataFrame(np.ravel(precip_grb[18:,18:]))

# df = pd.concat([lats,lons,u_grb,v_grb,Td_grb,T_grb,P_grb,precip_grb],axis=1)
# df.columns=['lat','lon','U_windspeed','V_windspeed','DewpointTemp','Temp','SurfPressure','Precip']


#df.to_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\ERA5data.csv")



