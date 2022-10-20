#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:51:50 2022

@author: laura
"""

#%% ------- IMPORTS --------




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import scipy.odr.odrpack as odr
import math

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


#%% ------  -------

%matplotlib inline

# Read in the data

green = pd.read_csv('Data/green.csv')
purple = pd.read_csv('Data/purple.csv')
blue = pd.read_csv('Data/blue.csv')
yellow = pd.read_csv('Data/yellow.csv')


# Green

green_v = green['Voltage [V]']
green_a = green['Average Current (e-7 A)']
green_verr = 0.04*green_v
green_aerr = green['Standard Deviation']

# Purple

purple_v = purple['Voltage (V)']
purple_a = purple['Average Current (e-7 A)']
purple_verr = 0.04*purple_v
purple_aerr = purple['Standard Deviation']

# Blue

blue_v = blue['Voltage [V]']
blue_a = blue['Average Current (e-7 A)']
blue_verr = 0.04*blue_v
blue_aerr = blue['Standard Deviation']

# Yellow

yellow_v = yellow['Voltage [V]']
yellow_a = yellow['Average Current (e-7 A)']
yellow_verr = 0.04*yellow_v
yellow_aerr = yellow['Standard Deviation']



# CUT ALL DATA TO RELEVANT RANGE

cut_green_v = green_v[green_v<=0]
cut_purple_v = purple_v[purple_v<=0]
cut_blue_v = blue_v[blue_v<=0]
cut_yellow_v = yellow_v[yellow_v<=0]

cut_green_a = green_a.iloc[cut_green_v.index[0]:]
cut_purple_a = purple_a.iloc[cut_purple_v.index[0]:]
cut_blue_a = blue_a.iloc[cut_blue_v.index[0]:]
cut_yellow_a = yellow_a.iloc[cut_yellow_v.index[0]:]

cut_green_aerr = green_aerr.iloc[cut_green_v.index[0]:]
cut_purple_aerr = purple_aerr.iloc[cut_purple_v.index[0]:]
cut_blue_aerr = blue_aerr.iloc[cut_blue_v.index[0]:]
cut_yellow_aerr = yellow_aerr.iloc[cut_yellow_v.index[0]:]

cut_green_verr = green_verr.iloc[cut_green_v.index[0]:]
cut_purple_verr = purple_verr.iloc[cut_purple_v.index[0]:]
cut_blue_verr = blue_verr.iloc[cut_blue_v.index[0]:]
cut_yellow_verr = yellow_verr.iloc[cut_yellow_v.index[0]:]



# DEFINE FUNCTION TO FIT DATA

def physical_odr(B, V):
    # I = Imax + (Imin - Imax)/(1 + np.exp((V - Vc)/(Vo)))
    I = B[1] + (B[2] - B[1]) / (1 + np.exp((V - B[3])/(B[4])))
    return I


# FIT DATA

# ---- GREEN -----

plt.subplot(2, 2, 1)

plt.scatter(cut_green_v, cut_green_a, marker='.', color='green', label='Green')
plt.errorbar(cut_green_v, cut_green_a, yerr=cut_green_aerr, xerr=cut_green_verr, elinewidth=3, capsize=4, capthick=1.8, color='green', ls='None')

guess1 = (1, 0, 1, 1)
range1 = np.linspace(-1.1, 0.1, 1000)

para1, cov1 = sp.optimize.curve_fit(physical, cut_green_v, cut_green_a, guess1, maxfev=100000, sigma=cut_green_aerr)
mean1 = (para1[1])
error1 = np.sqrt(cov1[1][1])

plt.plot(range1,physical(range1, para1[0], para1[1], para1[2], para1[3]), color='green')

plt.xlabel('Voltage [V]')
plt.ylabel('Current (e-7 A)')
plt.grid()
plt.tight_layout()

print(para1)
print(np.sqrt(np.diag(cov1)))


# ---- PURPLE -----

plt.subplot(2, 2, 2)

plt.scatter(cut_purple_v, cut_purple_a, marker='.', color='purple', label='purple')
plt.errorbar(cut_purple_v, cut_purple_a, yerr=cut_purple_aerr, xerr=cut_purple_verr, elinewidth=3, capsize=4, capthick=1.8, color='purple', ls='None')

guess2 = (1, 0, 1, 1)
range2 = np.linspace(-1.4, 0.1, 1000)

para2, cov2 = sp.optimize.curve_fit(physical, cut_purple_v, cut_purple_a, guess2, maxfev=100000)

plt.plot(range2,physical(range2, para2[0], para2[1], para2[2], para2[3]), color='purple')

plt.xlabel('Voltage [V]')
plt.ylabel('Current (e-7 A)')
plt.grid()
plt.tight_layout()


# ---- BLUE -----

plt.subplot(2, 2, 3)

plt.scatter(cut_blue_v, cut_blue_a, marker='.', color='blue', label='blue')
plt.errorbar(cut_blue_v, cut_blue_a, yerr=cut_blue_aerr, xerr=cut_blue_verr, elinewidth=3, capsize=4, capthick=1.8, color='blue', ls='None')

guess3 = (1, 0, 1, 1)
range3 = np.linspace(-1.1, 0.1, 1000)

para3, cov3 = sp.optimize.curve_fit(physical, cut_blue_v, cut_blue_a, guess3, maxfev=100000)

plt.plot(range1,physical(range3, para3[0], para3[1], para3[2], para3[3]), color='blue')

plt.xlabel('Voltage [V]')
plt.ylabel('Current (e-7 A)')
plt.grid()
plt.tight_layout()



# ---- YELLOW -----

plt.subplot(2, 2, 4)

plt.scatter(cut_yellow_v, cut_yellow_a, marker='.', color='orange', label='yellow')
plt.errorbar(cut_yellow_v, cut_yellow_a, yerr=cut_yellow_aerr, xerr=cut_yellow_verr, elinewidth=3, capsize=4, capthick=1.8, color='orange', ls='None')

guess4 = (1, 0, 1, 1)
range4 = np.linspace(-1.1, 0.1, 1000)

para4, cov4 = sp.optimize.curve_fit(physical, cut_yellow_v, cut_yellow_a, guess4, maxfev=100000)

plt.plot(range4,physical(range4, para4[0], para4[1], para4[2], para4[3]), color='orange')

plt.xlabel('Voltage [V]')
plt.ylabel('Current (e-7 A)')
plt.grid()
plt.tight_layout()







#%% ------- PLANCK'S CONSTANT -------


# DEFINE FUNCTION TO FIND CUT-OFF VOLTAGE FROM FIT PARAMETERS

def Cutoff(Imax, Imin, Vc, Vo):
    # I = 0
    V = Vc + Vo * np.log(-Imin/Imax)
    return V


# DEFINE FUNCTION TO PROPAGATE ERRORS FOR EACH PARAMETER TO CUT-OFF VOLTAGE ERROR

def Error_prop(V, Imax, Imin, Vc, Vo, Imax_err, Imin_err, Vc_err, Vo_err):
    
    rel_Imin = Imin_err/Imin
    rel_Imax = Imax_err/Imax
    log = -Imin/Imax

    log_err = (log/log) * np.sqrt(rel_Imin**22 + rel_Imax**2)
    
    rel_log = log_err/log
    rel_Vo = Vo_err/Vo
    
    V_err = np.sqrt(Vc_err**2 + (Vo * np.log(log) * np.sqrt(rel_Vo**2 + rel_log**2))**2)
    rel_V = V_err/V
    
    return V_err


# FIND CUT-OFF VOLTAGE FOR EACH COLOUR

g_cutoff = Cutoff(para1[0], para1[1], para1[2], para1[3])
p_cutoff = Cutoff(para2[0], para2[1], para2[2], para2[3])
b_cutoff = Cutoff(para3[0], para3[1], para3[2], para3[3])
y_cutoff = Cutoff(para4[0], para4[1], para4[2], para4[3])

final_cutoffs = [abs(g_cutoff), abs(p_cutoff), abs(b_cutoff), abs(y_cutoff)]


# FIND ERROR ON THE CUT-OFF VOLTAGE FOR EACH COLOUR

g_cutoff_err = Error_prop(abs(g_cutoff), para1[0], para1[1], para1[2], para1[3], np.sqrt(cov1[0][0]), np.sqrt(cov1[1][1]), np.sqrt(cov1[2][2]), np.sqrt(cov1[3][3]))
p_cutoff_err = Error_prop(abs(p_cutoff), para2[0], para2[1], para2[2], para2[3], np.sqrt(cov2[0][0]), np.sqrt(cov2[1][1]), np.sqrt(cov2[2][2]), np.sqrt(cov2[3][3]))
b_cutoff_err = Error_prop(abs(b_cutoff), para3[0], para3[1], para3[2], para3[3], np.sqrt(cov3[0][0]), np.sqrt(cov3[1][1]), np.sqrt(cov3[2][2]), np.sqrt(cov3[3][3]))
y_cutoff_err = Error_prop(abs(y_cutoff), para4[0], para4[1], para4[2], para4[3], np.sqrt(cov4[0][0]), np.sqrt(cov4[1][1]), np.sqrt(cov4[2][2]), np.sqrt(cov4[3][3]))

print(g_cutoff_err)
print(p_cutoff_err)
print(b_cutoff_err)
print(y_cutoff_err)

# THESE VALUES SEEM TOO BIG???????

# ERRORS FOR BOTH CUT-OFF VOLTAGE AND FREQUENCY MUST BE INCLUDED SOMEHOW



# OBTAIN LIST OF FREQUENCY VALUES FOR THE COLOURS

colours = ['green', 'purple', 'blue', 'yellow']
wavelengths = np.array([546.1, 404.7, 435.8, 578.2]) # in nm
metre_wl = wavelengths*10**-9 # in m

freq = 3*10**8 / metre_wl


# PLOT FREQUENCY AGAINST CUT-OFF VOLTAGE

plt.scatter(freq, final_cutoffs)


# FIT THE GRAPH WITH A STRAIGHT LINE

phys_fit, phys_cov = np.polyfit(freq, final_cutoffs, 1, cov=True)
phys_line = np.poly1d(phys_fit)
phys_range = np.linspace(5e14, 7.5e14, 1000)

plt.plot(phys_range, phys_line(phys_range))


# Find gradient and convert to h value

grad = phys_fit[0]
grad_err = np.sqrt(phys_cov[0])

e = 1.9 * 10**(-19)

h = (grad * e)
h_err = grad_err*e



print('Gradient', grad, '±', grad_err)

print("Planck's Constant: ", h, '±', h_err)












