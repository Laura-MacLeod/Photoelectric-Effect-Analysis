#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 13:10:35 2022

@author: laura

Written by Laura MacLeod

"""


# ------ IMPORTS -------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)



#%%


# ------ ALL COLOURS PLOTS ------


%matplotlib inline

# Read in the data

green = pd.read_csv('/Users/laura/Library/CloudStorage/OneDrive-ImperialCollegeLondon/University/Year 3/PHYS60004 - Third Year Physics Laboratory/Photoelectric Effect/Data/green.csv')
purple = pd.read_csv('/Users/laura/Library/CloudStorage/OneDrive-ImperialCollegeLondon/University/Year 3/PHYS60004 - Third Year Physics Laboratory/Photoelectric Effect/Data/purple.csv')
blue = pd.read_csv('/Users/laura/Library/CloudStorage/OneDrive-ImperialCollegeLondon/University/Year 3/PHYS60004 - Third Year Physics Laboratory/Photoelectric Effect/Data/blue.csv')
yellow = pd.read_csv('/Users/laura/Library/CloudStorage/OneDrive-ImperialCollegeLondon/University/Year 3/PHYS60004 - Third Year Physics Laboratory/Photoelectric Effect/Data/yellow.csv')


# Green

green_v = green['Voltage [V]']
green_a = green['Average Current (e-7 A)']
green_verr = 0.0002*green_v
green_aerr = green['Standard Deviation']

plt.figure(1)
plt.plot(green_v, green_a, marker='.', color='green', label='Green')
plt.errorbar(green_v, green_a, yerr=green_aerr, xerr=green_verr, elinewidth=3, capsize=4, capthick=1.8, color='green', ls='None')



# Purple

purple_v = purple['Voltage (V)']
purple_a = purple['Average Current (e-7 A)']
purple_verr = 0.0002*purple_v
purple_aerr = purple['Standard Deviation']

plt.plot(purple_v, purple_a, marker='.', color='purple', label='Purple')
plt.errorbar(purple_v, purple_a, yerr=purple_aerr, xerr=purple_verr, elinewidth=3, capsize=4, capthick=1.8, color='purple', ls='None')


# Blue

blue_v = blue['Voltage [V]']
blue_a = blue['Average Current (e-7 A)']
blue_verr = 0.0002*blue_v
blue_aerr = blue['Standard Deviation']

plt.plot(blue_v, blue_a, marker='.', color='blue', label='Blue')
plt.errorbar(blue_v, blue_a, yerr=blue_aerr, xerr=blue_verr, elinewidth=3, capsize=4, capthick=1.8, color='blue', ls='None')


# Yellow

yellow_v = yellow['Voltage [V]']
yellow_a = yellow['Average Current (e-7 A)']
yellow_verr = 0.0002*yellow_v
yellow_aerr = yellow['Standard Deviation']

plt.plot(yellow_v, yellow_a, marker='.', color='orange', label='Yellow')
plt.errorbar(yellow_v, yellow_a, yerr=yellow_aerr, xerr=yellow_verr, elinewidth=3, capsize=4, capthick=1.8, color='orange', ls='None')


# Noise

plt.plot([-1.25 ,9], [10**-6, 10**-6], color='black', label='Noise')

plt.xlabel('Voltage [V]')
plt.ylabel('Current (e-7 A)')
plt.grid()
plt.legend()

plt.xlim(-1.3, 0)
plt.ylim(-0.03, 0.06)

# plt.savefig("limited-all-colours-graph.png")


#%%

#  ------- VOLTAGE - X, CURRENT = Y -------

%matplotlib inline


#Limit the data to voltages below 0V

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



# ----- Green Plot -----

# plt.figure('green')
plt.subplot(2, 2, 1)
plt.scatter(cut_green_v, cut_green_a, marker='o', color='green', label='Green')
plt.errorbar(cut_green_v, cut_green_a, yerr=cut_green_aerr, xerr=cut_green_verr, elinewidth=3, capsize=4, capthick=1.8, color='green', ls='None')

# Fit cubic to data, incorporating y error

green_model = np.poly1d(np.polyfit(cut_green_v, cut_green_a, 3, cov=False, w=1/cut_green_aerr))
green_model_line = np.linspace(-1, 0, 100)
plt.plot(green_model_line, green_model(green_model_line))

# Zero amps line

plt.plot([-1 ,0], [10**-6, 10**-6], color='black', label='Noise')

plt.xlabel('Voltage [V]')
plt.ylabel('Current (e-7 A)')
plt.grid()
plt.title("Green")


# ----- Purple Plot -----

# plt.figure('purple')
plt.subplot(2, 2, 2)
plt.scatter(cut_purple_v, cut_purple_a, marker='o', color='purple', label='Purple')
plt.errorbar(cut_purple_v, cut_purple_a, yerr=cut_purple_aerr, xerr=cut_purple_verr, elinewidth=3, capsize=4, capthick=1.8, color='purple', ls='None')

# Fit cubic to data, incorporating y error

purple_model = np.poly1d(np.polyfit(cut_purple_v, cut_purple_a, 3, cov=False, w=1/cut_purple_aerr))
purple_model_line = np.linspace(-1.3, 0, 100)
plt.plot(purple_model_line, purple_model(purple_model_line))

# Zero amps line

plt.plot([-1.25 ,0], [10**-6, 10**-6], color='black', label='Noise')

plt.xlabel('Voltage [V]')
plt.ylabel('Current (e-7 A)')
plt.grid()
plt.title("Purple")


# ----- Blue Plot -----

# plt.figure('blue')
plt.subplot(2, 2, 3)
plt.scatter(cut_blue_v, cut_blue_a, marker='o', color='blue', label='Blue')
plt.errorbar(cut_blue_v, cut_blue_a, yerr=cut_blue_aerr, xerr=cut_blue_verr, elinewidth=3, capsize=4, capthick=1.8, color='blue', ls='None')

# Fit cubic to data, incorporating y error

blue_model = np.poly1d(np.polyfit(cut_blue_v, cut_blue_a, 3, cov=False, w=1/cut_blue_aerr))
blue_model_line = np.linspace(-1, 0, 100)
plt.plot(blue_model_line, blue_model(blue_model_line))

# Zero amps line

plt.plot([-1 ,0], [10**-6, 10**-6], color='black', label='Noise')

plt.xlabel('Voltage [V]')
plt.ylabel('Current (e-7 A)')
plt.grid()
plt.title("Blue")


# ----- Yellow Plot -----

# plt.figure('yellow')
plt.subplot(2, 2, 4)
plt.scatter(cut_yellow_v, cut_yellow_a, marker='o', color='orange', label='Yellow')
plt.errorbar(cut_yellow_v, cut_yellow_a, yerr=cut_yellow_aerr, xerr=cut_yellow_verr, elinewidth=3, capsize=4, capthick=1.8, color='orange', ls='None')

# Fit cubic to data, incorporating y error

yellow_model = np.poly1d(np.polyfit(cut_yellow_v, cut_yellow_a, 3, cov=False, w=1/cut_yellow_aerr))
yellow_model_line = np.linspace(-1, 0, 100)
plt.plot(yellow_model_line, yellow_model(yellow_model_line))

# Zero amps line

plt.plot([-1 ,0], [10**-6, 10**-6], color='black', label='Noise')

plt.xlabel('Voltage [V]')
plt.ylabel('Current (e-7 A)')
plt.grid()
plt.title("Yellow")
plt.tight_layout()









#%%

# ------- VOLTAGE - Y, CURRENT = X -------

%matplotlib inline

# Disable some annoying warnings

import warnings
warnings.filterwarnings(action='ignore', category=np.VisibleDeprecationWarning)


# Define ln curve to fit data

def lncurve(x, a, b, c, d, e):
    return e*d*(-a * np.log(x+b)+c)


# Old exponential saturation curve that I can't get to work

def saturation(x, a, b, c, d, e):
    sat = e*(a - b * np.exp(c * x + d))
    return sat



# ----- Green Plot -----

# plt.figure('green')
plt.subplot(2, 2, 1)
plt.scatter(cut_green_a, cut_green_v, marker='o', color='green', label='Green')
plt.errorbar(cut_green_a, cut_green_v, xerr=cut_green_aerr, yerr=cut_green_verr, elinewidth=3, capsize=4, capthick=1.8, color='green', ls='None')

guess1 = [1, 2, -1, -4, 1]

green_range = np.linspace(cut_green_a[-1:], cut_green_a[:1], 1000)

para1, cov1 = sp.optimize.curve_fit(lncurve, cut_green_a, cut_green_v, guess1, maxfev=100000)
plt.plot(green_range,lncurve(green_range, para1[0], para1[1], para1[2], para1[3], para1[4]))
mean1 = (para1[1])
error1 = np.sqrt(cov1[1][1])

plt.ylabel('Voltage [V]')
plt.xlabel('Current (e-7 A)')
plt.grid()
plt.title("Green")


# ----- Purple Plot -----

# plt.figure('purple')
plt.subplot(2, 2, 2)
plt.scatter(cut_purple_a, cut_purple_v, marker='o', color='purple', label='Purple')
plt.errorbar(cut_purple_a, cut_purple_v, xerr=cut_purple_aerr, yerr=cut_purple_verr, elinewidth=3, capsize=4, capthick=1.8, color='purple', ls='None')

guess2 = [1, 2, -1, -4, 1]

purple_range = np.linspace(cut_purple_a[-1:], cut_purple_a[:1], 10-0)

para2, cov2 = sp.optimize.curve_fit(lncurve, cut_purple_a, cut_purple_v, guess2, maxfev=100000)
plt.plot(purple_range,lncurve(purple_range, para2[0], para2[1], para2[2], para2[3], para2[4]))
mean2 = (para2[1])
error2 = np.sqrt(cov2[1][1])

plt.ylabel('Voltage [V]')
plt.xlabel('Current (e-7 A)')
plt.grid()
plt.title("Purple")


# ----- Blue Plot -----

# plt.figure('blue')
plt.subplot(2, 2, 3)
plt.scatter(cut_blue_a, cut_blue_v, marker='o', color='blue', label='Blue')
plt.errorbar(cut_blue_a, cut_blue_v, xerr=cut_blue_aerr, yerr=cut_blue_verr, elinewidth=3, capsize=4, capthick=1.8, color='blue', ls='None')

guess3 = [1, 2, -1, -4, 1]

blue_range = np.linspace(cut_blue_a[-1:], cut_blue_a[:1], 1000)

para3, cov3 = sp.optimize.curve_fit(lncurve, cut_blue_a, cut_blue_v, guess3, maxfev=100000)
plt.plot(blue_range,lncurve(blue_range, para3[0], para3[1], para3[2], para3[3], para3[4]))
mean3 = (para3[1])
error3 = np.sqrt(cov3[1][1])

plt.ylabel('Voltage [V]')
plt.xlabel('Current (e-7 A)')
plt.grid()
plt.title("Blue")


# ----- Yellow Plot -----

# plt.figure('yellow')
plt.subplot(2, 2, 4)
plt.scatter(cut_yellow_a, cut_yellow_v, marker='o', color='orange', label='Yellow')
plt.errorbar(cut_yellow_a, cut_yellow_v, xerr=cut_yellow_aerr, yerr=cut_yellow_verr, elinewidth=3, capsize=4, capthick=1.8, color='orange', ls='None')

guess4 = [1, 2, -1, -4, 1]

yellow_range = np.linspace(cut_yellow_a[-1:], cut_yellow_a[:1], 1000)

para4, cov4 = sp.optimize.curve_fit(lncurve, cut_yellow_a, cut_yellow_v, guess4, maxfev=100000)
plt.plot(yellow_range,lncurve(yellow_range, para4[0], para4[1], para4[2], para4[3], para4[4]))
mean4 = (para4[1])
error4 = np.sqrt(cov4[1][1])

plt.ylabel('Voltage [V]')
plt.xlabel('Current (e-7 A)')
plt.grid()
plt.title("Yellow")


# Define function to find the nearest voltage datapoint to current=0 in the ln fit 

def intersect(xfit, yfit):
    diffs = []
    fitt = np.array(xfit)
    for i in xfit:
        diff = i-0
        diffs.append(abs(diff))
    minm = min(diffs)
    
    ind = diffs.index(minm)
    value = yfit[ind]
    # print(value)
    return value


# Find the cut-off voltages and their magnitudes for all the colours

green_zero = (intersect(green_range,lncurve(green_range, para1[0], para1[1], para1[2], para1[3], para1[4])))
purple_zero = (intersect(purple_range,lncurve(purple_range, para2[0], para2[1], para2[2], para2[3], para2[4])))
blue_zero = (intersect(blue_range,lncurve(blue_range, para3[0], para3[1], para3[2], para3[3], para3[4])))
yellow_zero = (intersect(yellow_range,lncurve(yellow_range, para4[0], para4[1], para4[2], para4[3], para4[4])))
    
green_cutoff = abs(intersect(green_range,lncurve(green_range, para1[0], para1[1], para1[2], para1[3], para1[4])))
purple_cutoff = abs(intersect(purple_range,lncurve(purple_range, para2[0], para2[1], para2[2], para2[3], para2[4])))
blue_cutoff = abs(intersect(blue_range,lncurve(blue_range, para3[0], para3[1], para3[2], para3[3], para3[4])))
yellow_cutoff = abs(intersect(yellow_range,lncurve(yellow_range, para4[0], para4[1], para4[2], para4[3], para4[4])))
  

# Plot their intersection points, and a line showing zero current

plt.subplot(2, 2, 1)
plt.plot([0, 0, 0], [0, green_zero, -1])
plt.plot([0], green_zero, marker='x', color='red', markersize=10)

plt.subplot(2, 2, 2)
plt.plot([0, 0, 0], [0, purple_zero, -1.2])
plt.plot([0], purple_zero, marker='x', color='red', markersize=10)

plt.subplot(2, 2, 3)
plt.plot([0, 0, 0], [0, blue_zero, -1])
plt.plot([0], blue_zero, marker='x', color='red', markersize=10)

plt.subplot(2, 2, 4)
plt.plot([0, 0, 0], [0, yellow_zero, -1])
plt.plot([0], yellow_zero, marker='x', color='red', markersize=10)
plt.tight_layout()


# Compile the cut-off voltages into a list with associated colours, wavelengths and frequencies

cutoffs = [green_cutoff,purple_cutoff,blue_cutoff,yellow_cutoff]

colours = ['green', 'purple', 'blue', 'yellow']
wavelengths = np.array([546.1, 404.7, 435.8, 578.2]) # in nm
metre_wl = wavelengths*10**-9 # in m

freq = 3*10**8 / metre_wl
 

# Plot graph of cut-off voltages and fit the data with a straight line

plt.figure(4)
plt.scatter(freq, cutoffs)

data, cov5=np.polyfit(freq, cutoffs, 1, cov=True)

lis = []
for i in data:
    for j in i:
        lis.append(j)
        
line = np.poly1d(lis)

model_line = np.linspace((3*10**8)/(4*10**-7), (3*10**8) / (6*10**-7), 1000)
plt.plot(model_line, line(model_line))

plt.xlabel("Frequency [Hz]")
plt.ylabel("Cut-Off Voltage [V]")
plt.grid()


# Find gradient and convert to h value

grad = data[0]
grad_err = np.sqrt(cov5[0,0])

e = 1.9 * 10**(-19)

h = (grad * e)
h_err = grad_err*e




print('Gradient', grad, '±', grad_err)

print("Planck's Constant: ", h, '±', h_err)

# print(data)
# print(cov5)


# print(cov5)





#%%

# ------- HALF INTENSITY BLUE COMPARISON --------

%matplotlib inline

blue_half = pd.read_csv('/Users/laura/Library/CloudStorage/OneDrive-ImperialCollegeLondon/University/Year 3/PHYS60004 - Third Year Physics Laboratory/Photoelectric Effect/Data/blue_half.csv')

blue_v_half = blue_half['Voltage [V]']
blue_a_half = blue_half['Average Current (e-7 A)']
blue_aerr_half = blue_half['Standard Deviation']


# Resize data

cut_blue_v_half = blue_v_half[blue_v_half<=0]
cut_blue_a_half = blue_a_half.iloc[cut_blue_v_half.index[0]:]
cut_blue_aerr_half = blue_aerr_half.iloc[cut_blue_v_half.index[0]:]


# Plot each dataset

plt.figure(1)
plt.scatter(cut_blue_a_half, cut_blue_v_half, marker='.', color='blue', label='Blue')
plt.errorbar(cut_blue_a_half, cut_blue_v_half, xerr=cut_blue_aerr_half, elinewidth=3, capsize=4, capthick=1.8, color='blue', ls='None')

plt.scatter(cut_blue_a, cut_blue_v, marker='.', color='blue', label='Blue')
plt.errorbar(cut_blue_a, cut_blue_v, xerr=cut_blue_aerr, elinewidth=3, capsize=4, capthick=1.8, color='blue', ls='None')

plt.ylabel('Voltage [V]')
plt.xlabel('Current (e-7 A)')
plt.grid()


# Fit the datasets

blue_range = np.linspace(cut_blue_a_half[-1:], cut_blue_a_half[:1], 1000)
blue_range_half = np.linspace(cut_blue_a[-1:], cut_blue_a[:1], 1000)

para3, cov3 = sp.optimize.curve_fit(lncurve, cut_blue_a, cut_blue_v, guess3, maxfev=100000)
plt.plot(blue_range_half,lncurve(blue_range_half, para3[0], para3[1], para3[2], para3[3], para3[4]), color='deepskyblue')
mean3 = (para3[1])
error3 = np.sqrt(cov3[1][1])

para3_half, cov3_half = sp.optimize.curve_fit(lncurve, cut_blue_a_half, cut_blue_v_half, guess3, maxfev=100000)
plt.plot(blue_range,lncurve(blue_range, para3_half[0], para3_half[1], para3_half[2], para3_half[3], para3_half[4]), color='darkblue')
mean3_half = (para3_half[1])
error3_half = np.sqrt(cov3_half[1][1])



# Find and plot the cut-off voltage for each dataset

blue_zero = (intersect(blue_range,lncurve(blue_range, para3[0], para3[1], para3[2], para3[3], para3[4])))
blue_cutoff = abs(intersect(blue_range,lncurve(blue_range, para3[0], para3[1], para3[2], para3[3], para3[4])))

plt.plot([0, 0, 0], [0, blue_zero, -1], color='grey')
plt.plot([0], blue_zero, marker='x', color='red', markersize=10)


blue_zero_half = (intersect(blue_range,lncurve(blue_range, para3_half[0], para3_half[1], para3_half[2], para3_half[3], para3_half[4])))
blue_cutoff_half = abs(intersect(blue_range,lncurve(blue_range, para3_half[0], para3_half[1], para3_half[2], para3_half[3], para3_half[4])))

plt.plot([0], blue_zero_half, marker='x', color='red', markersize=10)



print(blue_cutoff)
print(blue_cutoff_half)









