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
import scipy.odr.odrpack as odr

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
green_verr = 0.04*green_v
green_aerr = green['Standard Deviation']

plt.figure(1)
plt.plot(green_v, green_a, marker='.', color='green', label='Green')
plt.errorbar(green_v, green_a, yerr=green_aerr, xerr=green_verr, elinewidth=3, capsize=4, capthick=1.8, color='green', ls='None')



# Purple

purple_v = purple['Voltage (V)']
purple_a = purple['Average Current (e-7 A)']
purple_verr = 0.04*purple_v
purple_aerr = purple['Standard Deviation']

plt.plot(purple_v, purple_a, marker='.', color='purple', label='Purple')
plt.errorbar(purple_v, purple_a, yerr=purple_aerr, xerr=purple_verr, elinewidth=3, capsize=4, capthick=1.8, color='purple', ls='None')


# Blue

blue_v = blue['Voltage [V]']
blue_a = blue['Average Current (e-7 A)']
blue_verr = 0.04*blue_v
blue_aerr = blue['Standard Deviation']

plt.plot(blue_v, blue_a, marker='.', color='blue', label='Blue')
plt.errorbar(blue_v, blue_a, yerr=blue_aerr, xerr=blue_verr, elinewidth=3, capsize=4, capthick=1.8, color='blue', ls='None')


# Yellow

yellow_v = yellow['Voltage [V]']
yellow_a = yellow['Average Current (e-7 A)']
yellow_verr = 0.04*yellow_v
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
warnings.filterwarnings(action='ignore', category=RuntimeWarning)


# Define ln curve to fit data

def lncurve(x, a, b, c, d):
    return d*(a * np.log(x+b)+c)


# Old exponential saturation curve that I can't get to work

def saturation(x, a, b, c, d, e):
    sat = e*(a - b * np.exp(c * x + d))
    return sat



# ----- Green Plot -----

# plt.figure('green')
plt.subplot(2, 2, 1)
plt.scatter(cut_green_a, cut_green_v, marker='o', color='green', label='Green')
plt.errorbar(cut_green_a, cut_green_v, xerr=cut_green_aerr, yerr=cut_green_verr, elinewidth=3, capsize=4, capthick=1.8, color='green', ls='None')

guess1 = [1, 2, 1, 4]

green_range = np.linspace(cut_green_a[-1:], cut_green_a[:1], 1000)

para1, cov1 = sp.optimize.curve_fit(lncurve, cut_green_a, cut_green_v, guess1, maxfev=100000)
plt.plot(green_range,lncurve(green_range, para1[0], para1[1], para1[2], para1[3]))
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

guess2 = [1, 2, 1, 4]

purple_range = np.linspace(cut_purple_a[-1:], cut_purple_a[:1], 1000)

para2, cov2 = sp.optimize.curve_fit(lncurve, cut_purple_a, cut_purple_v, guess2, maxfev=100000)
plt.plot(purple_range,lncurve(purple_range, para2[0], para2[1], para2[2], para2[3]))
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

guess3 = [1, 2, 1, 4]

blue_range = np.linspace(cut_blue_a[-1:], cut_blue_a[:1], 1000)

para3, cov3 = sp.optimize.curve_fit(lncurve, cut_blue_a, cut_blue_v, guess3, maxfev=100000)
plt.plot(blue_range,lncurve(blue_range, para3[0], para3[1], para3[2], para3[3]))
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

guess4 = [1, 2, 1, 4]

yellow_range = np.linspace(cut_yellow_a[-1:], cut_yellow_a[:1], 1000)

para4, cov4 = sp.optimize.curve_fit(lncurve, cut_yellow_a, cut_yellow_v, guess4, maxfev=100000)
plt.plot(yellow_range,lncurve(yellow_range, para4[0], para4[1], para4[2], para4[3]))
mean4 = (para4[1])
error4 = np.sqrt(cov4[1][1])

plt.ylabel('Voltage [V]')
plt.xlabel('Current (e-7 A)')
plt.grid()
plt.title("Yellow")




# ------ Alternative Method -------



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

green_zero = (intersect(green_range,lncurve(green_range, para1[0], para1[1], para1[2], para1[3])))
purple_zero = (intersect(purple_range,lncurve(purple_range, para2[0], para2[1], para2[2], para2[3])))
blue_zero = (intersect(blue_range,lncurve(blue_range, para3[0], para3[1], para3[2], para3[3])))
yellow_zero = (intersect(yellow_range,lncurve(yellow_range, para4[0], para4[1], para4[2], para4[3])))
    
green_cutoff = abs(intersect(green_range,lncurve(green_range, para1[0], para1[1], para1[2], para1[3])))
purple_cutoff = abs(intersect(purple_range,lncurve(purple_range, para2[0], para2[1], para2[2], para2[3])))
blue_cutoff = abs(intersect(blue_range,lncurve(blue_range, para3[0], para3[1], para3[2], para3[3])))
yellow_cutoff = abs(intersect(yellow_range,lncurve(yellow_range, para4[0], para4[1], para4[2], para4[3])))
  


# ------- More Accurate Method, for Errors --------


# Using original fit function and parameters, y = e*d*(a * np.log(x+b)+c)

x0 = 0

y1=para1[3]*(para1[0]*np.log(x0+para1[1])+para1[2])
y2=para2[3]*(para2[0]*np.log(x0+para2[1])+para2[2])
y3=para3[3]*(para3[0]*np.log(x0+para3[1])+para3[2])
y4=para4[3]*(para4[0]*np.log(x0+para4[1])+para4[2])

green_cutoff = abs(y1)
purple_cutoff = abs(y2)
blue_cutoff = abs(y3)
yellow_cutoff = abs(y4)



# Plot their intersection points, and a line showing zero current

plt.subplot(2, 2, 1)
plt.plot([0, 0, 0], [0,y1, -1])
plt.plot([0], y1, marker='x', color='red', markersize=10)

plt.subplot(2, 2, 2)
plt.plot([0, 0, 0], [0, y2, -1.2])
plt.plot([0], y2, marker='x', color='red', markersize=10)

plt.subplot(2, 2, 3)
plt.plot([0, 0, 0], [0, y3, -1])
plt.plot([0], y3, marker='x', color='red', markersize=10)

plt.subplot(2, 2, 4)
plt.plot([0, 0, 0], [0, y4, -1])
plt.plot([0], y4, marker='x', color='red', markersize=10)
plt.tight_layout()





# Compile the cut-off voltages into a list with associated colours, wavelengths and frequencies

cutoffs = [green_cutoff,purple_cutoff,blue_cutoff,yellow_cutoff]

colours = ['green', 'purple', 'blue', 'yellow']
wavelengths = np.array([546.1, 404.7, 435.8, 578.2]) # in nm
metre_wl = wavelengths*10**-9 # in m

freq = 3*10**8 / metre_wl
 
# print(cutoffs)



# Propagate the errors

ga = np.sqrt(cov1[0][0])
gb = np.sqrt(cov1[1][1])
gc = np.sqrt(cov1[2][2])
gd = np.sqrt(cov1[3][3])

# print(ga)
# print(gb)
# print(gc)
# print(gd)


gerr = np.sqrt(np.diag(cov1))
print(gerr)



# Plot graph of cut-off voltages and fit the data with a straight line

plt.figure(4)
plt.scatter(freq, cutoffs)

data, cov5=np.polyfit(freq, cutoffs, 1, cov=True)

lis = []
for j in data:
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

# print(freq)

print(para1)
print(para2)
print(para3)
print(para4)




#%%

# ------- Trying the ODR Pack --------

%matplotlib inline


# Function in form for ODR

def lncurve_odr(B, x):
    return B[4]*(B[1] * np.log(x+B[2])+B[3])


# Create ODR fits for all colours

g_log = odr.Model(lncurve_odr)
g_vals = odr.RealData(cut_green_a, cut_green_v)
g_vals2 = odr.ODR(g_vals, b_log, beta0=[0., 0.07175217, 0.01609299, 0.10803696, 2.00043475])
g_out = g_vals2.run()
# g_out.pprint()
g_bet = g_out.beta
g_cov = g_out.cov_beta

p_log = odr.Model(lncurve_odr)
p_vals = odr.RealData(cut_purple_a, cut_purple_v)
p_vals2 = odr.ODR(p_vals, p_log, beta0=[0., 0.26534439, 0.01163041, 0.49440464, 1.5653418])
p_out = p_vals2.run()
p_out.pprint()
p_bet = p_out.beta
p_cov = p_out.cov_beta

b_log = odr.Model(lncurve_odr)
b_vals = odr.RealData(cut_blue_a, cut_blue_v)
b_vals2 = odr.ODR(b_vals, b_log, beta0=[0., 0.17296461, 0.06275943, 0.10908215, 2.33282591])
b_out = b_vals2.run()
# b_out.pprint()
b_bet = b_out.beta
b_cov = b_out.cov_beta

y_log = odr.Model(lncurve_odr)
y_vals = odr.RealData(cut_yellow_a, cut_yellow_v)
y_vals2 = odr.ODR(y_vals, y_log, beta0=[0., 0.53810543, 0.00538403, 1.74327582, 0.2641375])
y_out = y_vals2.run()
# y_out.pprint()
y_bet = y_out.beta
y_cov = y_out.cov_beta



# ----- Green Plot -----

# plt.figure('green')
plt.subplot(2, 2, 1)
plt.scatter(cut_green_a, cut_green_v, marker='o', color='green')
plt.errorbar(cut_green_a, cut_green_v, xerr=cut_green_aerr, yerr=cut_green_verr, elinewidth=3, capsize=4, capthick=1.8, color='green', ls='None')

green_range = np.linspace(cut_green_a[-1:]-0.004, (cut_green_a[:1]), 1000)

plt.plot(green_range,lncurve(green_range, g_bet[1], g_bet[2], g_bet[3], g_bet[4]), color='red', label='ODR')
plt.plot(green_range,lncurve(green_range, para1[0], para1[1], para1[2], para1[3]), color='green', label='curve_fit')

plt.ylabel('Voltage [V]')
plt.xlabel('Current (e-7 A)')
plt.grid()
plt.title("Green")
plt.legend()
plt.tight_layout()


# ----- Purple Plot -----

# plt.figure('purple')
plt.subplot(2, 2, 2)
plt.scatter(cut_purple_a, cut_purple_v, marker='o', color='purple', label='Purple')
plt.errorbar(cut_purple_a, cut_purple_v, xerr=cut_purple_aerr, yerr=cut_purple_verr, elinewidth=3, capsize=4, capthick=1.8, color='purple', ls='None')

purple_range = np.linspace(cut_purple_a[-1:]-0.004, (cut_purple_a[:1]), 1000)

plt.plot(purple_range,lncurve(purple_range, p_bet[1], p_bet[2], p_bet[3], p_bet[4]), color='red')
plt.plot(purple_range,lncurve(purple_range, para2[0], para2[1], para2[2], para2[3]), color='green')

plt.ylabel('Voltage [V]')
plt.xlabel('Current (e-7 A)')
plt.grid()
plt.title("Purple")


# ----- Blue Plot -----

# plt.figure('purple')
plt.subplot(2, 2, 3)
plt.scatter(cut_blue_a, cut_blue_v, marker='o', color='blue', label='Blue')
plt.errorbar(cut_blue_a, cut_blue_v, xerr=cut_blue_aerr, yerr=cut_blue_verr, elinewidth=3, capsize=4, capthick=1.8, color='blue', ls='None')

blue_range = np.linspace(cut_blue_a[-1:]-0.004, (cut_blue_a[:1]), 1000)

plt.plot(blue_range,lncurve(blue_range, b_bet[1], b_bet[2], b_bet[3], b_bet[4]), color='red')
plt.plot(blue_range,lncurve(blue_range, para3[0], para3[1], para3[2], para3[3]), color='green')

plt.ylabel('Voltage [V]')
plt.xlabel('Current (e-7 A)')
plt.grid()
plt.title("Blue")


# ----- Yellow Plot -----

# plt.figure('Yellow')
plt.subplot(2, 2, 4)
plt.scatter(cut_yellow_a, cut_yellow_v, marker='o', color='orange', label='Yellow')
plt.errorbar(cut_yellow_a, cut_yellow_v, xerr=cut_yellow_aerr, yerr=cut_yellow_verr, elinewidth=3, capsize=4, capthick=1.8, color='orange', ls='None')

yellow_range = np.linspace(cut_yellow_a[-1:]-0.004, (cut_yellow_a[:1]), 1000)

plt.plot(yellow_range,lncurve(yellow_range, y_bet[1], y_bet[2], y_bet[3], y_bet[4]), color='red')
plt.plot(yellow_range,lncurve(yellow_range, para4[0], para4[1], para4[2], para4[3]), color='green')

plt.ylabel('Voltage [V]')
plt.xlabel('Current (e-7 A)')
plt.grid()
plt.title("Yellow")
plt.tight_layout()



# Using original fit function and parameters, y = e*d*(a * np.log(x+b)+c)

x0 = 0

v1=g_bet[4]*(g_bet[1]*np.log(x0+g_bet[2])+g_bet[3])
v2=p_bet[4]*(p_bet[1]*np.log(x0+p_bet[2])+p_bet[3])
v3=b_bet[4]*(b_bet[1]*np.log(x0+b_bet[2])+b_bet[3])
v4=y_bet[4]*(y_bet[1]*np.log(x0+y_bet[2])+y_bet[3])

new_green_cutoff = abs(v1)
new_purple_cutoff = abs(v2)
new_blue_cutoff = abs(v3)
new_yellow_cutoff = abs(v4)



# Plot their intersection points, and a line showing zero current

plt.subplot(2, 2, 1)
plt.plot([0, 0, 0], [0,v1, -1])
plt.plot([0], v1, marker='x', color='red', markersize=10)

plt.subplot(2, 2, 2)
plt.plot([0, 0, 0], [0, v2, -1.2])
plt.plot([0], v2, marker='x', color='red', markersize=10)

plt.subplot(2, 2, 3)
plt.plot([0, 0, 0], [0, v3, -1])
plt.plot([0], v3, marker='x', color='red', markersize=10)

plt.subplot(2, 2, 4)
plt.plot([0, 0, 0], [0, v4, -1])
plt.plot([0], v4, marker='x', color='red', markersize=10)
plt.tight_layout()





# Compile the cut-off voltages into a list with associated colours, wavelengths and frequencies

new_cutoffs = [new_green_cutoff,new_purple_cutoff,new_blue_cutoff,new_yellow_cutoff]

colours = ['green', 'purple', 'blue', 'yellow']
wavelengths = np.array([546.1, 404.7, 435.8, 578.2]) # in nm
metre_wl = wavelengths*10**-9 # in m

freq = 3*10**8 / metre_wl



# print(cutoffs)
# print(new_cutoffs)


# Plot new cutoffs and old cutoffs on same plot

plt.figure('cutoffs')
plt.scatter(freq, cutoffs, color='green', label='curve_fit')
plt.scatter(freq, new_cutoffs, color='red', label='ODR')
plt.legend()


# Add linear fits for both new and old cutoffs

plt.plot(model_line, line(model_line), color='green')

odr_data, odr_cov5=np.polyfit(freq, new_cutoffs, 1, cov=True)

odr_lis = []
for j in odr_data:
    odr_lis.append(j)
        
odr_line = np.poly1d(odr_lis)

odr_model_line = np.linspace((3*10**8)/(4*10**-7), (3*10**8) / (6*10**-7), 1000)
plt.plot(odr_model_line, odr_line(odr_model_line), color='red')

plt.xlabel("Frequency [Hz]")
plt.ylabel("Cut-Off Voltage [V]")
plt.grid()
plt.legend()


# Find gradient and convert to h value for new and old cutoffs

odr_grad = odr_data[0]
odr_grad_err = np.sqrt(odr_cov5[0,0])

e = 1.9 * 10**(-19)

odr_h = (odr_grad * e)
odr_h_err = odr_grad_err*e



# Compare ODR and curve_fit h values

print("curve_fit Planck's Constant: ", h, '±', h_err)

print("ODR Planck's Constant: ", odr_h, '±', odr_h_err)





# ----- Errors ------

g_err = np.sqrt(np.diag(g_cov))
print(g_bet)
print(g_err)

p_err = np.sqrt(np.diag(p_cov))
print(p_bet)
print(p_err)

# print(p_bet.sd_beta)

# odr.stopreason()


'''

ODR fit not properly working, retrieving: 
    Reason(s) for Halting:
        Problem is not full rank at solution
        
Both the set of standard deviatations from the covariance matrix and the
provided standard errors are significantly too big for the beta parameters.

These errors are not usable.
  
'''
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
plt.plot(blue_range_half,lncurve(blue_range_half, para3[0], para3[1], para3[2], para3[3]), color='deepskyblue')
mean3 = (para3[1])
error3 = np.sqrt(cov3[1][1])

para3_half, cov3_half = sp.optimize.curve_fit(lncurve, cut_blue_a_half, cut_blue_v_half, guess3, maxfev=100000)
plt.plot(blue_range,lncurve(blue_range, para3_half[0], para3_half[1], para3_half[2], para3_half[3]), color='darkblue')
mean3_half = (para3_half[1])
error3_half = np.sqrt(cov3_half[1][1])



# Find and plot the cut-off voltage for each dataset

blue_zero = (intersect(blue_range,lncurve(blue_range, para3[0], para3[1], para3[2], para3[3])))
blue_cutoff = abs(intersect(blue_range,lncurve(blue_range, para3[0], para3[1], para3[2], para3[3])))

plt.plot([0, 0, 0], [0, blue_zero, -1], color='grey')
plt.plot([0], blue_zero, marker='x', color='red', markersize=10)


blue_zero_half = (intersect(blue_range,lncurve(blue_range, para3_half[0], para3_half[1], para3_half[2], para3_half[3])))
blue_cutoff_half = abs(intersect(blue_range,lncurve(blue_range, para3_half[0], para3_half[1], para3_half[2], para3_half[3])))

plt.plot([0], blue_zero_half, marker='x', color='red', markersize=10)



print(blue_cutoff)
print(blue_cutoff_half)







