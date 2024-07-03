#### Library to handle the velocity axis of spectral data cube/hyper spectral image ####
import numpy as np

## Convert the velocity to a pixel index based on the provided header information
def vel_to_pixel(vel, crval, dv, crpix):
    return int((vel - crval)/dv + crpix + 0.5)
    

## Convert the pixel location to the velocity based on the provided header information
def pixel_to_vel(pix, crval, dv, crpix):
    return crval + dv*(pix - crpix)


## return an array of velocities based on a velocity range and  information from the header of a hyperimage
def create_velocity_array(min_vel, max_vel, dv):
    ## construct the velocity array using np.arange
    vel_arr = np.arange(start = min_vel, stop = max_vel, step = dv)
    
    return vel_arr


## return hyper_image where the z-axis has been reduced based on a provided velocity range and the header information
def reduce_z_axis_size(data, min_vel, max_vel, crval, dv, crpix):
    ## convert the minimal and maximal velocity to pixel indices
    z_min = vel_to_pixel(min_vel, crval, dv, crpix)
    z_max = vel_to_pixel(max_vel, crval, dv, crpix)
    
    ## cut the data
    data_reduced = data[z_min:z_max,:,:]
    
    ## print the pixel region tha tis cut out
    print("The minimal pixel along the z-axis is: {z_min}".format(z_min = z_min))
    print("The maximal pixel along the z-axis is: {z_max}".format(z_max = z_max))
    
    return data_reduced















