### Regrid AOD data to coarser ERA5-compatible grid

# ref: https://earthscience.stackexchange.com/questions/12057/how-to-interpolate-scattered-data-to-a-regular-grid-in-python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import xarray as xr
from mpl_toolkits import basemap

#aoddata = pd.read_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\MAIACAOD.csv")
aoddata = pd.read_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\MAIACAOD.csv")
lat = pd.read_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\lats.csv")
lon = pd.read_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\lons.csv")
press = pd.read_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\SurfPressure.csv")

#coarser ERA5 grid
lat = np.array(lat)[:,2]
lon = np.array(lon)[2,1:]
lat = np.round(lat,1)
lon = np.round(lon,1)
#x,y = np.meshgrid(lon,lat)

aodlats = aoddata.lat
aodlons = aoddata.lon
aod = aoddata.value
orbit = np.array(aoddata.orbit)

# X,Y = np.meshgrid(aodlons,aodlats)

#idx = np.where(aodlats[:-1] != aodlats[1:])[0]

#Get unique lat lons for AOD grid:
aodlats_cut = np.unique(aodlats)
aodlons_cut = np.unique(aodlons)

# aod_interp = basemap.interp(aod,aodlons,aodlats,lon,lat,order=1)


# Round lat lons to era5 lat lon values, find indices of each rounded coordinate, and average at that location 
# aodlons_round = np.round(aodlons,1)
# aodlats_round = np.round(aodlats,1)
aoddata['lat'] = aoddata['lat'].astype(float).round(1)
aoddata['lon'] = aoddata['lon'].astype(float).round(1)


# loop through era5 lats and lons, for each unique lat/lon pair, find corresponding aod lat, lon idxs 
# average values of those idxs, save in same df order as era5 lat, lon
AOD_new = []
lat_new = []
lon_new = []
for i in range(len(lat)):
    for j in range(len(lon)):
        temp = aoddata.loc[(aoddata.lat == lat[i]) & (aoddata.lon == lon[j])]
        aod_avg = temp.loc[:,'value'].mean()
        AOD_new.append(aod_avg) 
        lat_new.append(lat[i])
        lon_new.append(lon[j])
       
AOD_regrid = pd.DataFrame({'lat': lat_new, 'lon': lon_new, 'AOD' : AOD_new})

AOD_new = np.array(AOD_new).reshape(47,47)
lat_new = np.array(lat_new).reshape(47,47)
lon_new = np.array(lon_new).reshape(47,47)


# idx = np.where(orbit[:-1] != orbit[1:])[0]
# first_orbit =idx[0]
# latscut = aodlats[:first_orbit]
# lonscut = aodlons[:first_orbit]
# aodcut = aod[:first_orbit]


### Shows before regrid
# f, ax = plt.subplots(1, 1, figsize=(4, 4))
# img = ax.scatter(
#     x=aoddata.lon,
#     y=aoddata.lat,
#     c=aoddata.value,
#     s=0.15,
#     alpha=1,
#     cmap="RdYlBu_r")
# f.colorbar(img)
# plt.suptitle("Blue Band AOD", fontsize=12)

# ### Shows after regrid
# f, ax = plt.subplots(1, 1, figsize=(4, 4))
# img = ax.scatter(
#     x=lon_new,
#     y=lat_new,
#     c=AOD_new,
#     s=0.15,
#     alpha=1,
#     cmap="RdYlBu_r")
# f.colorbar(img)
# plt.suptitle("Blue Band AOD", fontsize=12)

AODfinal = pd.DataFrame(AOD_new)
AODfinal.to_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\AODregrid.csv")







# def find_nearest(x,y,gridx,gridy):
#     distance = (gridx-x)**2 + (gridy-y)**2
#     idx = np.where(distance==distance.min())
#     return [idx[0][0], idx[1][0]]

# find_nearest(lon,lat,aodlons,aodlats)

#data_set = xr.Dataset({"temp": (["lat", "lon"], data)},
                 # coords={"lon": lon,"lat": lat})


# # data coordinates and values
# # x = aoddata.lon
# # y = aoddata.lat
# # z = aoddata.value

# x = era5data.lon
# y = era5data.lat
# z = era5data.Temp

# # target grid to interpolate to
# xi = era5data.lon
# yi = era5data.lat
# xi,yi = np.meshgrid(xi,yi)

# # interpolate
# zi = griddata((x,y),z,(xi,yi),method='linear')



# # plot
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.contourf(xi,yi,zi,np.arange(0,1.01,0.01))
# plt.plot(x,y,'k.')
# plt.xlabel('xi',fontsize=16)
# plt.ylabel('yi',fontsize=16)
# #plt.savefig('interpolated.png',dpi=100)
# plt.close(fig)