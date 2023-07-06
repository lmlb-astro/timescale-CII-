import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import astropy.wcs as wcs
import math

from scipy.odr import ODR, Model, Data, RealData
import scipy.ndimage
import random


#### DEFINE THE FUNCTIONS THAT WILL BE USED IN "calculateSizeAndTimescaleOfTheRegion.ipynb" ####

## define the linear functions for fitting
def linFunc(beta, x):
    y = beta[0]*x + beta[1]
    return y

def linFunc_noOff(beta, x):
    y = beta[0]*x
    return y

## fit a linear function with offset to the give x, y data using ODR
def fit_lin_func(list_x,list_y,err_x,err_y,init_guess,x_beg,x_end,num_x):
    data = RealData(list_x,list_y,sx=err_x,sy=err_y)
    linear = Model(linFunc)
    odr = ODR(data,linear,beta0=init_guess)
    output = odr.run()
    output.pprint()
    x = np.linspace(x_beg,x_end,num_x)
    y = linFunc(output.beta, x)
    
    return x, y

## fit a linear function with no offset to the give x, y data using ODR
def fit_lin_func_no_offset(list_x,list_y,err_x,err_y,init_guess,x_beg,x_end,num_x):
    data = RealData(list_x,list_y,sx=err_x,sy=err_y)
    linear_no = Model(linFunc_noOff)
    odr = ODR(data,linear_no,beta0=init_guess)
    output = odr.run()
    output.pprint()
    x = np.linspace(x_beg,x_end,num_x)
    y = linFunc_noOff(output.beta, x)
    
    return x, y


## Calculate the x and y positions in the map for the cut
def get_points_cuts(xb,xe,yb,ye,margin,x_cent,y_cent,a,angle_rad,mid_angle):
    x0 = xb + margin
    x1 = xe - margin
    y0 = y_cent + a*(x0-x_cent)
    y1 = y_cent + a*(x1-x_cent)
    ## to make sure all x and y values are part of the available map
    if(mid_angle < angle_rad < np.pi-mid_angle):
        y0 = yb + margin
        y1 = ye - margin
        x0 = x_cent + (y0-y_cent)/a + margin
        x1 = x_cent + (y1-y_cent)/a - margin
    return x0, x1, y0, y1

## Estimate the size of the region along a specific axis based on the indices and peak emission
def get_peak_diff(zi,min_intensity,peakDiffs,pixSize):
    zi[zi<min_intensity] = 0.
    midZi = int(0.5*len(zi) + 0.5)
    arr1 = zi[0:midZi]
    arr2 = zi[midZi:len(zi)]
    try:
        minInd = np.nanargmax(arr1)
        maxInd = midZi + np.nanargmax(arr2)
        if(zi[minInd]>min_intensity and zi[maxInd]>min_intensity):
            peakDiffs.append((maxInd-minInd)*pixSize)
        else:
            print("Could not determine the size of the region along one axis")
    except:
        print("Not possible to estimate peak-based size along this axis")
    return peakDiffs


## Estimate the (maximal) size of the region along a specific axis based on the indices
def get_max_diff(zi,min_intensity,maxDiffs,pixSize):
    midZi = len(zi)/2
    zi[np.isnan(zi)] = 0.
    inds = np.argwhere(zi>min_intensity)
    try:
        minInd = np.nanmin(inds)
        maxInd = np.nanmax(inds)
        if(minInd<midZi and maxInd>midZi):
            maxDiffs.append((maxInd-minInd)*pixSize)
    except:
        print("Not possible to estimate the maximal size along this axis")
        #print("The available indices are: " + str(minInd) + " " + str(maxInd) + ", middle value is:" + str(midZi))
    return maxDiffs


## Plot the intensity profiles
def plot_intensity_profiles(zis,save_profiles,path_profiles):
    for zi in zis:
        plt.plot(zi)
    plt.xlabel('distance (pixels)')
    plt.ylabel('T$_{mb}$ (K)')
    if(save_profiles):
        plt.savefig(path_profiles+nameSource+'_intensity_profiles.pdf',dpi=300)
    plt.show()
    plt.clf()

## plot the cuts on the map
def plot_cuts_on_map(xb, xe, yb, ye, x0s, x1s, y0s, y1s, dat, wcs_info, cmap_choice, save_map, nameSource, path_maps):
    plt.clf()
    fig, ax = plt.subplots()
    ax1 = fig.add_subplot(111, projection=wcs_info)
    im = ax1.imshow(dat, origin='lower', vmin=0., cmap = cmap_choice)
    
    plt.xlim([xb, xe])
    plt.ylim([yb, ye])
    plt.xlabel('RA [J2000]')
    plt.ylabel('DEC [J2000]')
    
    ## plot the axes
    for x0, x1, y0, y1 in zip(x0s,x1s,y0s,y1s):
        ax1.plot([x0, x1], [y0, y1], 'ro-')
        
    ## Finalize the plot of the map with the intensity cuts for the region
    cbar = fig.colorbar(im)
    cbar.set_label('$\int$T$_{mb}$dv (K km s$^{-1}$)', labelpad=15.,rotation=270.)
    
    ax.axis('off')
    if(save_map):
        plt.savefig(path_maps+nameSource+'integrated_map+cuts.pdf',dpi=300)
    plt.show()
    plt.clf()


## find size of the region along multiple cuts over the region
def getSizesFromCuts(xb,xe,yb,ye,theta,numCuts_f,margin,min_intensity,dat,wcs_info,pixSize,nameSource,save_map,save_profiles,cmap_choice, path_maps, path_profiles):
    zis = []; peakDiffs = []; maxDiffs = []
    x0s = []; x1s = []; y0s = []; y1s = []
    
    ## calculate the angle associated with the upper east pixel in the map
    x_cent = xb + 0.5*(xe-xb)
    y_cent = yb + 0.5*(ye-yb)
    mid_angle = np.arctan(y_cent/x_cent)

    ## get the intensity cut and add it to the plot
    for i in range(0,numCuts_f):
        ## get slope for the intensity cut
        angle_rad = i*theta*np.pi/180.
        a = math.tan(angle_rad)
        
        ## get the x and y positions in the map for the cut
        x0, x1, y0, y1 = get_points_cuts(xb,xe,yb,ye,margin,x_cent,y_cent,a,angle_rad,mid_angle)
        x0s.append(x0); x1s.append(x1); y0s.append(y0); y1s.append(y1)
        
        ## Extract the profiles from the data set
        num = int(2.*np.sqrt((x1-x0)**2 + (y1-y0)**2) + 0.5)
        cut_len = np.sqrt((x1-x0)**2 + (y1-y0)**2)*pixSize
        ind_len = cut_len/num
        x, y = np.linspace(x0,x1,num), np.linspace(y0, y1, num)
        
        ## Extract the profiles from the data set
        zi = dat[y.astype(np.int)-yb, x.astype(np.int)-xb]
        zis.append(zi)
        
        ## Estimate the size of the region along a specific axis based on the indices of the peak intensity position
        peakDiffs = get_peak_diff(zi,min_intensity,peakDiffs,ind_len)
        
        ## Estimate the size of the region along a specific axis based on the indices
        maxDiffs = get_max_diff(zi,min_intensity,maxDiffs,ind_len)
    
    ## plot the cuts onto the map
    plot_cuts_on_map(xb, xe, yb, ye, x0s, x1s, y0s, y1s, dat, wcs_info, cmap_choice, save_map, nameSource, path_maps)
    
    ## Plot the intensity profiles associated with the cuts
    plot_intensity_profiles(zis,save_profiles, path_profiles)
    
    return 0.5*np.nanmean(maxDiffs), 0.5*np.nanstd(maxDiffs), 0.5*np.nanmean(peakDiffs), 0.5*np.nanstd(peakDiffs) ## Returns the radius (*0.5) + standard deviation


## Remove all the data from a 2D map that is not within the specified pixel locations
def extract_region_in_map(data,x1,x2,y1,y2):
    data[:,0:x1] = np.nan
    data[:,x2:len(data[0])] = np.nan
    data[0:y1,:] = np.nan
    data[y2:len(data),:] = np.nan
    
    return data

## Remove all the data from a 3D spectral cube that is not within the specified pixel locations
def extract_region_in_cube(cube,x1,x2,y1,y2,z1,z2):
    cube[:,:,0:x1] = np.nan
    cube[:,:,x2:len(cube[0][0])] = np.nan
    cube[:,0:y1,:] = np.nan
    cube[:,y2:len(cube[0]),:] = np.nan
    cube[0:z1,:,:] = np.nan
    cube[z2:len(cube),:,:] = np.nan
    
    return cube


## Return the velocity and spectral array in a specific velocity interval
def get_velArr_and_spec_in_interval(v1,v2,crval,cdelt,crpix,velArr,spec):
    zmin = int((v1-crval)/cdelt + crpix + 0.5)
    zmax = int((v2-crval)/cdelt + crpix + 0.5)
    velArr = velArr[zmin:zmax]
    spec = spec[zmin:zmax]
    
    return velArr, spec


## Create a randomly sampled list of radii based on index parameters in a map and the pixel size
def sample_radius_size_from_indices(indices,sample_size,dimension,pixSize,corrFactSize):
    rList = []
    for i in range(0,sample_size):
        ## randomly sample  indices
        vals = random.choices(indices,k=dimension)
        x1 = vals[0][0]
        y1 = vals[0][1]
        x2 = vals[1][0]
        y2 = vals[1][1]
        
        ## calculate physical radius associated with the provided indices
        r = np.sqrt((x2-x1)**2 + (y2-y1)**2)*pixSize*0.5*corrFactSize ## times 0.5 to get radius
        rList.append(r)
    
    return rList
















