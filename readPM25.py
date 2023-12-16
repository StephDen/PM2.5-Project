### 555b Final Project
### Read EPA PM2.5 csv files and extract date/variables of interest

#### Where to get PM2.5 data: https://www.epa.gov/outdoor-air-quality-data/download-daily-data 

import numpy as np
import pandas as pd

data_wash = pd.read_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\Washington_2021_PM25_allstations.csv")
data_or = pd.read_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\Oregon_2021_PM25_allstations.csv")
date = '07/18/2021' #date of interest: July 18, 2021

o_idx = data_or[data_or['Date']==date]
w_idx = data_wash[data_wash['Date']==date]

df = pd.concat([o_idx,w_idx])
df.reset_index()

dates = pd.DataFrame(df["Date"])
pm25 = pd.DataFrame(df["Daily Mean PM2.5 Concentration"])
lat = pd.DataFrame(df["SITE_LATITUDE"])
lon = pd.DataFrame(df["SITE_LONGITUDE"])
df_new = pd.concat([dates,pm25,lat,lon], axis=1)
df_new.reset_index(drop=True)

df_new.to_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\PM25.csv")