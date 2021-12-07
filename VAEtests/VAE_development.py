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
import pathlib
import geopandas as gpd
import pandas as pd
from rasterio.features import geometry_mask
import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt
import rasterio as rio
# import glob

import VAE
import dataset
# import TSNEembedding
import pipelineFunctions

import os
from tensorflow import keras
import datetime
import matplotlib.pyplot as plt
import shutil
import configparser

''' ----------
Load configuration settings
------------'''
# os.chdir(NEW_PATH)
def parse_config(config):
    """ Parse input arguments from dictionary or config file """
    if not isinstance(config, dict):
        parser = configparser.ConfigParser()
        parser.read(config)
        config = parser["train-VAE"]

    dataPath = config['dataPath']
    labelsPath = config['labelsPath']
    imName = config['imName']
    outputDir = config['outputDirectory']
    sizeCutOut = int(config['sizeCutOut'])
    #DATA
    balanceRatio = float(config['balanceRatio'])
    normThreshold = float(config['normalizationThreshold'])
    nBands = int(config['nBands'])
    # MODEL
    filter1 = int(config['filter1'])
    filter2 = int(config['filter2'])
    kernelSize = int(config['kernelSize'])
    denseSize = int(config['sizeCutOut'])
    latentDim = int(config['latentDim'])
    #vae:
    alpha = 5
    nEpochs = int(config['nEpochs'])
    batchSize = int(config['batchSize'])
    validationSplit = float(config['validationSplit'])
        
    return (outputDir, dataPath,labelsPath, imName,normThreshold , sizeCutOut, nBands, balanceRatio,
            filter1, filter2, kernelSize, denseSize,latentDim,
            alpha, nEpochs,batchSize,validationSplit)



# load .ini 
train_settings = r'/Users/maaikeizeboud/Documents/github/maaikeizb/Iceshelves/VAEtests/vae_versions/train-vae-tests.ini'

outputDir, data_path, labels_path, imName, norm_threshold ,cutout_size , n_bands, balance_ratio,\
    filter_1, filter_2, kernel_size, dense_size, latent_dim, \
    alpha, n_epochs,batch_size,train_val_split = parse_config(train_settings)

# Make outputDir
# using datetime module for naming the current model, so that old models
# do not get overwritten
ct = datetime.datetime.now()  # ct stores current time
ts = ct.timestamp()  # ts store timestamp of current time
os.makedirs(outputDir + 'model_' + str(int(ts)))
path = os.path.join(outputDir + 'model_' + str(int(ts)) )



#%% 
''' ----------
Load data for development (1 tile including labels)
------------'''

tile_data = rioxr.open_rasterio(data_path + imName)
tile_data.rio.write_crs("epsg:3031", inplace=True) # set_crs(input_crs)#, inplace=True)

## if tile does not include labels as 4th band:
if len(labels_path) > 0: # not a clean commant but 'if labels_path is not None' doesnt work. 

    # load labels for tht dataset
    labels = pipelineFunctions._read_labels(labels_path, verbose=True)
    # labels = labels.to_crs(tiles.crs)  # make sure same CRS is used
    labels = labels.to_crs("EPSG:3031")  # make sure same CRS is used

    # # select the only labels matching the tiles timespan
    labels = pipelineFunctions._filter_labels(labels,
                            '2019-11-01 00:00:00',#tiles.start_datetime.min(),
                            '2020-03-01 00:00:00')

    labels_raster = pipelineFunctions.mask_labels_tile(labels, tile_data)
    tile_data_np = tile_data.values
    tile_data_np = np.concatenate((tile_data_np[0:3],labels_raster));

    tile = xr.DataArray(data=tile_data_np,dims=['band','y','x'],
                        coords={'band':tile_data.coords['band'],
                                'y':tile_data[0].coords['y'],'x':tile_data[0].coords['x']})
else:
    tile = tile_data


''' ----------
Save training data(figure) and settings 
(save .ini as .txt file for easy reading)
------------'''
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(8,8))
ax1.imshow(tile[0]) 
ax2.imshow(tile[-1])
fig.savefig(os.path.join(path  , 'trained_data_and_labels') )

shutil.copyfile(train_settings, os.path.join(path , 'train-vae-tests.txt'))
shutil.copyfile('/Users/maaikeizeboud/Documents/github/maaikeizb/Iceshelves/VAEtests/VAE.py',os.path.join(path , 'VAE.py') )

''' ----------
Create cut-outs
    Actually read the tile, make cutouts, linked with labeldata
------------'''
da = tile

# generate windows
da = da.rolling(x=cutout_size, y=cutout_size)
da = da.construct({'x': 'x_win', 'y': 'y_win'}, stride=cutout_size)

# drop NaN-containing windows
da = da.stack(sample=('x', 'y'))
da = da.dropna(dim='sample', how='any')

# tile_cutouts = da.data.transpose(3, 1, 2, 0) # samples, x_win, y_win, bands: (250000, 20, 20, 3)
tile_cutouts = da.transpose('sample','x_win','y_win','band')
# print(tile_cutouts.shape)


''' ----------
Balance data 
------------'''    
if balance_ratio > 0:
    tile_cutouts_balanced = pipelineFunctions.balance_windowdata(tile_cutouts, balance_ratio)
    print('Balanced dataset with ratio labelled/unlabbeled {}'.format(balance_ratio))
else:
    tile_cutouts_balanced = tile_cutouts
    
x_train = tile_cutouts_balanced[:,:,:,0:3].values
y_train = tile_cutouts_balanced[:,:,:,-1].values
print(x_train.shape,y_train.shape)

''' ----------
Normalize
------------'''
# if tile_cutoouts is a numpy array:
# if norm_threshold is not None:
#     tile_bands_normalized = tile_cutouts[:,:,:,0:3] # only normralize RGB bands
#     tile_bands_normalized = (tile_bands_normalized + 0.1) / (norm_threshold + 1)
#     tile_bands_normalized = tile_bands_normalized.clip(max=1)
#     tile_cutouts[:,:,:,0:3] = tile_bands_normalized # substitute normalized values back to data; not touching the labels    
# if tile cutouts is a xarray
if norm_threshold is not None:
    x_train = (x_train + 0.1) / (norm_threshold + 1)
    x_train = x_train.clip(max=1)
    
    
''' ----------
Train Network
------------'''
    
epochcounter = 1  # start at 0 or adjust offset calculation

encoder_inputs, encoder, z, z_mean, z_log_var = VAE.make_encoder(cutout_size,n_bands,filter_1,kernel_size,filter_2,dense_size,latent_dim)
decoder = VAE.make_decoder(latent_dim,filter_1,filter_2,kernel_size)
vae = VAE.make_vae(encoder_inputs, z, z_mean, z_log_var, decoder,alpha)
vae.compile(optimizer=keras.optimizers.Adam())

# train including validation loss:
history = vae.fit(x_train, epochs=n_epochs, batch_size=batch_size, validation_split=train_val_split)

# save trained model
vae.save(os.path.join(path , 'model'))
encoder.save(os.path.join(path , 'encoder'))
# history.save(os.path.join(path , 'history'))


''' ----------
Save model loss figure
------------'''

# summarize history for loss
fig, ax = plt.subplots()
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_title('model loss')
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'val'], loc='upper right')

fig.savefig(os.path.join(path  , 'model_loss' ) )


''' ----------
Save model prediction figure
------------'''
# 
encoded_data,_,_ = encoder.predict(x_train);
print('---- succesfully encoded data; size: ', encoded_data.shape)

predicted_data = vae.predict(x_train); # reconstruct images (windows):
print('---- succesfully predicted data')


# PLOT ORIGINAL AND PREDICTED WINDOWS (slice data to be less than 400000 samples)
n_samples = 9

# x_train_xr= xr.DataArray( 
#                     data=x_train,dims=['sample','x_win','y_win','band'],
#                     coords={'band':da.isel(band=[0,1,2]).coords['band'],'y_win':da[0].coords['y_win'],'x_win':da[0].coords['x_win'],'sample':da[0].coords['sample']}
#                     )
x_train_xr= xr.DataArray(data=x_train,dims=['sample','x_win','y_win','band'],
                         coords={'band':tile_cutouts_balanced.isel(band=[0,1,2]).coords['band'],
                                 'y_win':tile_cutouts_balanced[0].coords['y_win'],
                                 'x_win':tile_cutouts_balanced[0].coords['x_win'],
                                 'sample':tile_cutouts_balanced.coords['sample']})
da_plot = x_train_xr.isel(sample=slice(0, n_samples))
fig_data = da_plot.isel(band=0).plot(x="x_win", y="y_win", col="sample", col_wrap=3).fig
fig_data.savefig(os.path.join(path  , 'windows_example_inputs' ) )

# predicted_data_xr= xr.DataArray(
#                     data=predicted_data,dims=['sample','x_win','y_win','band'],
#                     coords={'band':da.isel(band=[0,1,2]).coords['band'],'y_win':da[0].coords['y_win'],'x_win':da[0].coords['x_win'],'sample':da[0].coords['sample']}
#                     )
predicted_data_xr= xr.DataArray(data=predicted_data,dims=['sample','x_win','y_win','band'],
                                coords={'band':tile_cutouts_balanced.isel(band=[0,1,2]).coords['band'],
                                        'y_win':tile_cutouts_balanced[0].coords['y_win'],
                                        'x_win':tile_cutouts_balanced[0].coords['x_win'],
                                        'sample':tile_cutouts_balanced.coords['sample']})
predicted_data_plot = predicted_data_xr.isel(sample=slice(0, n_samples))
fig_predict = predicted_data_plot.isel(band=0).plot(x="x_win", y="y_win", col="sample", col_wrap=3, vmin=predicted_data_xr.values.min(), vmax=predicted_data_xr.values.max()).fig
fig_predict.savefig(os.path.join(path  , 'windows_example_predicted') )


''' ----------
Save model latent space clustering figure
------------'''
label_data = y_train
labels = np.nansum(label_data,axis=(1,2)) # compress labeldata from (Nsamples, 20, 20, 1) to (Nsamples, 1)
labels = labels/(labels.min() + labels.max()) # rescale to 0-1


if any(labels > 0):
    print('Labels in tile')

# Embed to 2D
encoded_2D_testdata = pipelineFunctions.embed_latentspace_2D(encoded_data,latent_dim)
print('Succesfully embedded data to 2D')


fig, ax = pipelineFunctions.plot_latentspace_clusters( encoded_2D_testdata , labels )
fig.savefig(os.path.join(path  , 'latent_space_clusters' ))

print('Done.')