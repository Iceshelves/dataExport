#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:46:31 2021

@author: maaikeizeboud
"""

''' ----------
Import modules
------------'''
import rioxarray as rioxr
# import pathlib
# import geopandas as gpd
# import pandas as pd
# from rasterio.features import geometry_mask
# import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt
import rasterio as rio
#%% 
''' ----------
Load data for development (1 tile including labels)
------------'''

dataPath = '/Users/maaikeizeboud/Documents/Data/test/'
imName = 'test_labelled_tile.tif'

tile = rioxr.open_rasterio(dataPath + imName)
tile.isnull().any() 
src = rio.open(dataPath + imName)
src.crs
tile.rio.write_crs("epsg:3031", inplace=True) # set_crs(input_crs)#, inplace=True)

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(8,8))
ax1.imshow(tile[0]) 
ax2.imshow(tile[-1])

labels_tileraster = tile[-1]
tile = tile[0:3]

# create dataArray from np ndarray
da = tile
da_label = xr.DataArray(
            data=labels_tileraster,
            dims=["band","x", "y"])

#%%
''' ----------
Create cut-outs
    Actually read the tile, make cutouts, linked with labeldata
------------'''
da = tile
cutout_size = 20

# generate windows
da = da.rolling(x=cutout_size, y=cutout_size)
da = da.construct({'x': 'x_win', 'y': 'y_win'}, stride=cutout_size)

# drop NaN-containing windows
da = da.stack(sample=('x', 'y'))
da = da.dropna(dim='sample', how='any')

tile_cutouts = da.data.transpose(3, 1, 2, 0) # samples, x_win, y_win, bands: (250000, 20, 20, 3)



print(tile_cutouts.shape)

''' ----------
Normalize
------------'''

# normThreshold = tile_cutouts.max() #17264
normThreshold = 10000 

if normThreshold is not None:
    da = (da + 0.1) / (normThreshold + 1)
    da = da.clip(max=1)
    tile_cutouts = da.data.transpose(3, 1, 2, 0) # samples, x_win, y_win, bands: (250000, 20, 20, 4) 


''' ----------
Feed data to network
------------'''
x_train = tile_cutouts[:,:,:,0:3]
y_train = tile_cutouts[:,:,:,-1]
print(x_train.shape,y_train.shape)

