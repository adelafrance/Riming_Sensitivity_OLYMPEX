"""
Andrew DeLaFrance
April 2023

Script Function: 
Read McSnow model output
Dependent function for plot_model_output_2D.py 
"""

import numpy as np

def strip_ascii_data(fn, n):

    f = open(fn, 'r')
    len_list_z = int(f.readline().rstrip('\n').split('\t')[0].split()[1])+1
    len_list_x = int(f.readline().rstrip('\n').split('\t')[0].split()[1])

    unused_line1 = f.readline().rstrip('\n')
    unused_line2 = f.readline().rstrip('\n')

    data = np.full((n+1,len_list_z, len_list_x), np.nan) #time, z, x
    z_list = np.full((len_list_z), np.nan)
    time_list = np.full(n+1, np.nan) #minutes

    for t in range(n+1):
        time = int(float(f.readline().rstrip('\n').replace('"', '').split()[1]))
        time_list[t] = time
        for line in range(len_list_z):
            newline = f.readline().rstrip('\n')
            floatline = [float(x) for x in newline.split()]
            if t == 0:
                z_pos = floatline[0]
                z_list[line] = z_pos
            data[t,line,:] = floatline[1:len_list_x+1]
        unused_line1 = f.readline().rstrip('\n')
        unused_line2 = f.readline().rstrip('\n')

    var = data[:,:,:] #i.e., Fm = total mass flux kg/m^2/s

    return(z_list, var, time_list)
