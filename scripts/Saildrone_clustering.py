#basic packages
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib as mpl
import gsw
import seaborn as sns
import glob

#clustering packages
import itertools
from scipy import linalg, interpolate
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from sklearn import mixture
from sklearn.neighbors import NearestCentroid
import statsmodels.api as sm

#map packages
from shapely.geometry import mapping
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cartopy.crs as ccrs

SEED = 7


###Process Saildrone Data#################################################################
#read in data
saildrone = xr.open_dataset('../data/saildrone_westcoast_combined.nc')
saildrone.close()

#resample per day
saildrone = saildrone.to_dataframe().groupby('relativeID').resample('1D').mean(numeric_only = True).drop('relativeID', axis = 1).reset_index('relativeID').to_xarray()
saildrone = saildrone.rename({'SAL_MEAN':'SAL_CTD_MEAN'})

#extract out the year/time
saildrone = saildrone.where(((saildrone.time >= pd.to_datetime('2018-07-01')) & (saildrone.time <= pd.to_datetime('2018-09-30'))) |
                            ((saildrone.time >= pd.to_datetime('2019-07-01')) & (saildrone.time <= pd.to_datetime('2019-09-30'))), drop = True)

#filter out salinity
saildrone = saildrone.where(saildrone.SAL_CTD_MEAN >= 30, drop = True)

#create array with temp and salinity data
X = saildrone.to_dataframe()[['TEMP_CTD_MEAN','SAL_CTD_MEAN']].dropna(axis = 0).to_numpy()

#look at BIC scores
lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ["spherical", "tied", "diag", "full"]
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(
            n_components=n_components, covariance_type=cv_type, random_state = SEED
        )
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
clf = best_gmm
bars = []
Y_ = clf.predict(X)

#add the clusters to the data
saildrone_df = saildrone.to_dataframe().dropna()
saildrone_df['GMM_Labels'] = Y_

#add colors to dataframe
colors = pd.DataFrame({'GMM_Labels': range(6), 'color': ["navy", "turquoise", "cornflowerblue", "darkorange", "purple", "slategrey"]})
saildrone_df = saildrone_df.merge(colors)

#add centroids
centroids = saildrone_df.groupby('GMM_Labels').mean(numeric_only = True)


###Function to classify specified satellite dates#################################################################
def saildrone_classify(date0, date1):
    '''
    Function to open selected MUR SST and SMAP SSS data, classify those points based on the Saildrone clusters, and produce a series of figures
    
    Inputs: 
    date0 = start date as a string (in format YYYY-MM-DD)
    date1 = end date as a string (in format YYYY-MM-DD)
    
    Outputs: 
    Map of SST points
    Map of SSS points
    TS diagram of clusters
    Map of clusters
    '''
    
    #convert dates into date range
    date_range = pd.date_range(date0, date1, freq = '1D')
    
    ###Open SMAP files
    #create list of all SMAP files
    files = glob.glob('/Users/administrator/Library/CloudStorage/GoogleDrive/SMAP/GMM_Clustering/*.nc4')

    files_sel = []

    #loop through all dates and files
    for date in date_range:
        for file in files: 
            #select out the files that match the date
            if str(date).split(' ')[0].replace('-','') in file:
                files_sel.append(file)

    smap_df = pd.DataFrame()

    #open each file, extract the data, and combine it 
    for file in files_sel: 
        temp = xr.open_dataset(file)
        temp.close()
        temp_df = temp.to_dataframe()
        temp_df['time'] = pd.to_datetime(file.split('SSS_')[1].split('_8DAYS')[0])
        temp_df = temp_df.set_index('time', append = True)
        smap_df = pd.concat([smap_df, temp_df])

    #convert the dataframe to xarray
    smap = smap_df.groupby(level=smap_df.index.names).mean().to_xarray()
    
    if smap.latitude.values[0] < 0 :
        smap_df = smap.to_dataframe().dropna()
        smap_df = smap_df.reset_index(['longitude', 'latitude'])
        
        if smap_df['longitude'][0] < 0:
            lons = smap_df['longitude']
            lats = smap_df['latitude']
        else:
            lats = smap_df['longitude']
            lons = smap_df['latitude']
        smap_df['latitude'] = lats
        smap_df['longitude'] = lons
        smap = smap_df.set_index(['longitude', 'latitude'], append = True).to_xarray()
    print('Opened SMAP data')
    
    ###Open MUR data
    #if the date_range is older, open the ZARR and extract the data
    if date_range[0].year < 2019: 
        temp = xr.open_zarr('https://mur-sst.s3.us-west-2.amazonaws.com/zarr-v1')
        mur = temp.sel(time = slice(date_range[0], date_range[-1] + pd.DateOffset(1)), 
                       lat = slice(32.03, 47.94), 
                       lon = slice(-127.5, -117.3)).analysed_sst.load()
        mur = mur - 273.15
        mur = mur.to_dataset()

    #otherwise open the individuals files
    else: 

        #create list of all SMAP files
        files = glob.glob('/Users/administrator/Library/CloudStorage/GoogleDrive/MUR/GMM_Clustering/*.nc4')

        files_sel = []

        #loop through all dates and files
        for date in date_range:
            for file in files: 
                #select out the files that match the date
                if str(date).split(' ')[0].replace('-','') in file:
                    files_sel.append(file)

        #open the files
        mur = xr.open_mfdataset(files_sel)
        mur.close()
        
        mur['analysed_sst'] = xr.where(mur['analysed_sst'] > 200, mur['analysed_sst'] - 273.15, mur['analysed_sst'])

        
    print('Opened MUR data')

    ###Combine data together
    #open cali mask
    cali = gpd.read_file('./data/California_Mask_Erased/California_Mask_Erased.shp')

    #rename MUR variables
    mur = mur.rename({'lat':'latitude', 'lon':'longitude'})

    #filter out salinity
    smap = smap.where(smap.smap_sss >= 30, drop = True)

    #resample to a single timestamp
    mur = mur.mean(dim = 'time')
    smap = smap.mean(dim = 'time')

    #coarce mur data to smap grid
    mur_interp = mur.interp_like(smap.smap_sss)#, kwargs={"fill_value": np.nan})

    #join data together
    comb = xr.merge([smap, mur_interp]).to_dataframe().to_xarray()

    #clip data to shapefile
    comb.rio.set_spatial_dims(x_dim = "longitude", y_dim = "latitude", inplace = True)
    comb.rio.write_crs("epsg:4326", inplace = True)

    #clip the data
    comb = comb[['analysed_sst', 'smap_sss']].rio.clip(cali.geometry.apply(mapping), cali.crs, drop = True)

    ###Add density calculations
    #convert to dataframe, calculate density using gsw package, and convert back to xarray
    comb_df = comb.to_dataframe()
    comb_df['DENSITY_MEAN'] = gsw.sigma0(comb_df.smap_sss, comb_df.analysed_sst)
    comb = comb_df.to_xarray()

    #extract ou the minimum and maximum temp/salinity values
    mint=np.min(comb_df['analysed_sst'])
    maxt=np.max(comb_df['analysed_sst'])

    mins=np.min(comb_df['smap_sss'])
    maxs=np.max(comb_df['smap_sss'])

    #create an evenly spaced series based on the range of values
    tempL=np.linspace(mint-2,maxt+2,156)
    salL=np.linspace(mins-0.25,maxs+0.25,156)

    #create a meshgride and fill linspace with density values
    Tg, Sg = np.meshgrid(tempL,salL)
    sigma_theta = gsw.sigma0(Sg, Tg)
    cnt = np.linspace(sigma_theta.min(), sigma_theta.max(),156)

    #calculate the climatology for the three months 
    comb_temp = comb.to_dataframe().reset_index().dropna().reset_index()
    
    ###Recalculate distances to clusters
    #normalize data 
    def NormalizeData(data, col):
        return (data[col] - np.min(comb_temp[col])) / (np.max(comb_temp[col]) - np.min(comb_temp[col]))

    comb_temp['smap_sss_NORM'] = NormalizeData(comb_temp, 'smap_sss')
    comb_temp['analysed_sst_NORM'] = NormalizeData(comb_temp, 'analysed_sst')
    col = 'smap_sss'
    centroids['SAL_CTD_MEAN_NORM'] = (centroids['SAL_CTD_MEAN'] - np.min(comb_temp[col])) / (np.max(comb_temp[col]) - np.min(comb_temp[col]))
    col = 'analysed_sst'
    centroids['TEMP_CTD_MEAN_NORM'] = (centroids['TEMP_CTD_MEAN'] - np.min(comb_temp[col])) / (np.max(comb_temp[col]) - np.min(comb_temp[col]))

    dist0 = []
    dist1 = []
    dist2 = []
    dist3 = []
    dist4 = []
    dist5 = []

    #loop through each point in the satellite data
    for i in range(len(comb_temp)):
        #calculate the distance to each centroid
        dist0.append(np.sqrt((((comb_temp.iloc[i]['smap_sss_NORM'] - centroids.iloc[0]['SAL_CTD_MEAN_NORM'])**2) + ((comb_temp.iloc[i]['analysed_sst_NORM'] - centroids.iloc[0]['TEMP_CTD_MEAN_NORM'])**2))))
        dist1.append(np.sqrt((((comb_temp.iloc[i]['smap_sss_NORM'] - centroids.iloc[1]['SAL_CTD_MEAN_NORM'])**2) + ((comb_temp.iloc[i]['analysed_sst_NORM'] - centroids.iloc[1]['TEMP_CTD_MEAN_NORM'])**2))))
        dist2.append(np.sqrt((((comb_temp.iloc[i]['smap_sss_NORM'] - centroids.iloc[2]['SAL_CTD_MEAN_NORM'])**2) + ((comb_temp.iloc[i]['analysed_sst_NORM'] - centroids.iloc[2]['TEMP_CTD_MEAN_NORM'])**2))))
        dist3.append(np.sqrt((((comb_temp.iloc[i]['smap_sss_NORM'] - centroids.iloc[3]['SAL_CTD_MEAN_NORM'])**2) + ((comb_temp.iloc[i]['analysed_sst_NORM'] - centroids.iloc[3]['TEMP_CTD_MEAN_NORM'])**2))))
        dist4.append(np.sqrt((((comb_temp.iloc[i]['smap_sss_NORM'] - centroids.iloc[4]['SAL_CTD_MEAN_NORM'])**2) + ((comb_temp.iloc[i]['analysed_sst_NORM'] - centroids.iloc[4]['TEMP_CTD_MEAN_NORM'])**2))))
        dist5.append(np.sqrt((((comb_temp.iloc[i]['smap_sss_NORM'] - centroids.iloc[5]['SAL_CTD_MEAN_NORM'])**2) + ((comb_temp.iloc[i]['analysed_sst_NORM'] - centroids.iloc[5]['TEMP_CTD_MEAN_NORM'])**2))))

    #combine into a dataframe
    distances = pd.DataFrame({'Cluster_0':dist0, 'Cluster_1':dist1,
                             'Cluster_2':dist2, 'Cluster_3':dist3,
                             'Cluster_4':dist4, 'Cluster_5':dist5})

    #add in a relative id column
    distances = distances.reset_index()

    #convert from wide to long
    distances = distances.melt(id_vars=['index'], value_vars=['Cluster_0', 'Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4', 'Cluster_5'], value_name = 'distance', var_name = 'GMM_Labels_new')
    distances = distances.dropna(subset = 'distance')

    #select the smallest one from each point (ie the nearest centroid)
    distances = distances.loc[distances.groupby('index').distance.idxmin()].set_index('index')#.reset_index(drop=True).set_index('index')

    #join with the satellite data
    sat_reclass = comb_temp.join(distances)
    sat_reclass = sat_reclass.dropna(subset = 'GMM_Labels_new')

    ###TS plot with new clusters
    #create a figure
    fig, ax = plt.subplots(figsize=(6,4))

    #add denstiy lines
    cs = ax.contour(Sg, Tg, sigma_theta, colors='darkgrey', zorder=1)
    cl=plt.clabel(cs,fontsize=10,inline=True,fmt='%.0f')

    #add colors to dataframe
    colors = pd.DataFrame({'GMM_Labels_new': ['Cluster_0', 'Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4', 'Cluster_5'], 
                           'color_new': ["navy", "turquoise", "cornflowerblue", "darkorange", "purple", "slategrey"]})
    sat_reclass_col = sat_reclass.merge(colors, on = 'GMM_Labels_new')
    
    #join the centroids and data
    centroids_str = centroids.reset_index()[['GMM_Labels', 'TEMP_CTD_MEAN', 'SAL_CTD_MEAN']].rename({'GMM_Labels':'GMM_Labels_new', 'TEMP_CTD_MEAN':'TEMP_centroid', 'SAL_CTD_MEAN':'SAL_centroid'}, axis = 1)
    centroids_str['GMM_Labels_new'] = centroids_str['GMM_Labels_new'].apply(lambda x: "{}{}".format('Cluster_', x))
    sat_reclass_col = sat_reclass_col.merge(centroids_str)

    #add lines
    for ids, val in sat_reclass_col.iterrows():
        y = [val.analysed_sst, val.TEMP_centroid]
        x = [val.smap_sss, val.SAL_centroid]
        plt.plot(x, y, c = val.color_new, alpha = 0.15, zorder = 1)

    #plot points colored by cluster
    plt.scatter(y = sat_reclass_col['analysed_sst'], x = sat_reclass_col['smap_sss'], c = sat_reclass_col['color_new'], s = 10, alpha = 0.8)

    #add centroids
    #centroids = saildrone_df.groupby('GMM_Labels').mean(numeric_only = True)
    plt.scatter(y = centroids['TEMP_CTD_MEAN'], x = centroids['SAL_CTD_MEAN'], c = 'yellow', s = 200, edgecolors='black', marker = '*', zorder = 2) ###star centroid   

    #add title and format axes
    plt.title("GMM Clusters ("+date0[0:7]+")")

    plt.ylabel('MUR SST [°C]')
    plt.xlabel('SMAP SSS [PSU]')
    plt.subplots_adjust(hspace=0.35, bottom=0.02)
    plt.savefig('../figures/satellite_TS_with_GMMclusters_reclassified_' + str(date_range[0]).split(' ')[0] + '_' + str(date_range[-1]).split(' ')[0]+'.png', bbox_inches = 'tight')
    # plt.show()
    plt.close()
    print('Created TS diagram with clusters')
    
    ###Map with cluster colors
    #define latitude and longitude boundaries
    latr = [min(comb['latitude']), max(comb['latitude'])] 
    lonr = [max(comb['longitude']), min(comb['longitude'])] 

    # Select a region of our data, giving it a margin
    margin = 0.5 
    region = np.array([[latr[0]-margin,latr[1]+margin],[lonr[0]+margin,lonr[1]-margin]]) 

    #add state outlines
    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')

    # Create and set the figure context
    fig = plt.figure(figsize=(8,5), dpi = 72) 
    ax = plt.axes(projection=ccrs.PlateCarree()) 
    ax.coastlines(resolution='10m',linewidth=1,color='black') 
    ax.add_feature(cfeature.LAND, color='grey', alpha=0.3)
    ax.add_feature(states_provinces, linewidth = 0.5)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent([region[1,0],region[1,1],region[0,0],region[0,1]],crs=ccrs.PlateCarree()) 
    ax.set_xticks(np.round([*np.arange(region[1,1],region[1,0]+1,4)][::-1],0), crs=ccrs.PlateCarree()) 
    ax.set_yticks(np.round([*np.arange(region[0,0],region[0,1]+1,5)],0), crs=ccrs.PlateCarree()) 
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.gridlines(linestyle = '--', linewidth = 0.5)

    # Plot track data, color by temperature
    cmap = (mpl.colors.ListedColormap(["navy", "turquoise", "cornflowerblue", "darkorange", "purple", "slategrey"])) #west coast
    #cmap = (mpl.colors.ListedColormap(["cornflowerblue", "navy", "turquoise", "darkorange"])) #california

    plt.scatter(x = sat_reclass_col['longitude'], y = sat_reclass_col['latitude'], c = sat_reclass_col['color_new'], alpha = 0.9, s = 10)

    plt.title("Map of GMM Clusters ("+date0[0:7]+")", fontdict = {'fontsize' : 10})
    plt.savefig('../figures/satellite_map_with_GMMclusters_reclassified_' + str(date_range[0]).split(' ')[0] + '_' + str(date_range[-1]).split(' ')[0]+'.png', bbox_inches = 'tight', dpi = 150)
    # plt.show()
    plt.close()
    print('Created map with clusters')

    ###Map with SST
    #define latitude and longitude boundaries
    latr = [min(comb['latitude']), max(comb['latitude'])] 
    lonr = [max(comb['longitude']), min(comb['longitude'])] 

    # Select a region of our data, giving it a margin
    margin = 0.5 
    region = np.array([[latr[0]-margin,latr[1]+margin],[lonr[0]+margin,lonr[1]-margin]]) 

    #add state outlines
    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')

    # Create and set the figure context
    fig = plt.figure(figsize=(8,5), dpi = 72) 
    ax = plt.axes(projection=ccrs.PlateCarree()) 
    ax.coastlines(resolution='10m',linewidth=1,color='black') 
    ax.add_feature(cfeature.LAND, color='grey', alpha=0.3)
    ax.add_feature(states_provinces, linewidth = 0.5)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent([region[1,0],region[1,1],region[0,0],region[0,1]],crs=ccrs.PlateCarree()) 
    ax.set_xticks(np.round([*np.arange(region[1,1],region[1,0]+1,4)][::-1],0), crs=ccrs.PlateCarree()) 
    ax.set_yticks(np.round([*np.arange(region[0,0],region[0,1]+1,5)],0), crs=ccrs.PlateCarree()) 
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.gridlines(linestyle = '--', linewidth = 0.5)

    ax = comb.analysed_sst.plot(x = 'longitude', cmap = 'jet', vmin = 9, vmax = 23, add_colorbar = False)

    clb = plt.colorbar(ax, pad = 0.02)
    clb.ax.set_title('[°C]')

    plt.title("MUR SST ("+date0[0:7]+")", fontdict = {'fontsize' : 12})
    plt.savefig('../figures/satellite_map_MUR_SST_' + str(date_range[0]).split(' ')[0] + '_' + str(date_range[-1]).split(' ')[0]+'.png', bbox_inches = 'tight', dpi = 150)
    # plt.show()
    plt.close()
    print('Created map of SST')
    
    ###Map with SSS
    #define latitude and longitude boundaries
    latr = [min(comb['latitude']), max(comb['latitude'])] 
    lonr = [max(comb['longitude']), min(comb['longitude'])] 

    # Select a region of our data, giving it a margin
    margin = 0.5 
    region = np.array([[latr[0]-margin,latr[1]+margin],[lonr[0]+margin,lonr[1]-margin]]) 

    #add state outlines
    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')

    # Create and set the figure context
    fig = plt.figure(figsize=(8,5), dpi = 72) 
    ax = plt.axes(projection=ccrs.PlateCarree()) 
    ax.coastlines(resolution='10m',linewidth=1,color='black') 
    ax.add_feature(cfeature.LAND, color='grey', alpha=0.3)
    ax.add_feature(states_provinces, linewidth = 0.5)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent([region[1,0],region[1,1],region[0,0],region[0,1]],crs=ccrs.PlateCarree()) 
    ax.set_xticks(np.round([*np.arange(region[1,1],region[1,0]+1,4)][::-1],0), crs=ccrs.PlateCarree()) 
    ax.set_yticks(np.round([*np.arange(region[0,0],region[0,1]+1,5)],0), crs=ccrs.PlateCarree()) 
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.gridlines(linestyle = '--', linewidth = 0.5)

    ax = comb.smap_sss.plot(x = 'longitude', cmap = 'jet', vmin = 31, vmax = 34, add_colorbar = False)

    clb = plt.colorbar(ax, pad = 0.02)
    clb.ax.set_title('[PSU]')

    plt.title("SMAP SSS ("+date0[0:7]+")", fontdict = {'fontsize' : 12})
    plt.savefig('../figures/satellite_map_SMAP_SSS_' + str(date_range[0]).split(' ')[0] + '_' + str(date_range[-1]).split(' ')[0]+'.png', bbox_inches = 'tight', dpi = 150)
    # plt.show()
    plt.close()
    print('Created map of SSS')
    
