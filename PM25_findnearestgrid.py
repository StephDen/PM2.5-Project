#Edit PM25 csv with nearest grid location

import numpy as np
import pandas as pd

pmdf = pd.read_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\PM25_update.csv")
lat = pd.read_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\lats.csv")
lon = pd.read_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\lons.csv")


pmdf['lat_round'] = pmdf['lat'].astype(float).round(1)
pmdf['lon_round'] = pmdf['lon'].astype(float).round(1)

#ERA5 grid info
lat = np.array(lat)[:,2]
lon = np.array(lon)[2,1:]
lat = np.round(lat,1)
lon = np.round(lon,1)

#Remove repeats in same lat/lon (idx = 3 and 7)
pmdf = pmdf.drop([7])
pmdf = pmdf.drop([3])
pmdf.reset_index(inplace=True)

#Find corresponding row/col values from ERA5 grid
col = []
row = []
for i in range(len(pmdf.lat_round)):
    temp_row = np.where(lat == pmdf.lat_round[i]) #lats are rows
    temp_col = np.where(lon == pmdf.lon_round[i]) #lons are cols
    temp_row = temp_row[0]
    temp_col = temp_col[0]
    row.append(temp_row)
    col.append(temp_col)
    
row = np.array(row)
col = np.array(col)

pmdf['row'] = row.astype(int)
pmdf['col'] = col.astype(int)    

pmdf.to_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\PM25_final.csv")   
    
    