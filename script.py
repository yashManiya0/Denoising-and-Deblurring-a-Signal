# for importing the required libraries

import numpy as np                                          # numpy for array handling and speedy operations
import pandas as pd                                         # pandas for reading data
import matplotlib.pyplot as plt                             # matplotlib for plotting graphs 


# Below are all the functions including DTFT, IFT, denoising, deblurring and calculating MSE

# denoise function for denoising the signal given as argument to the function (x)
def denoise(x):
    n = len(x)
    y = []
    for i in range(n):                                      # denoising by taking moving average 
        if i == 0:                                          # of samples of the given signal by
            y.append((3*x[i]+x[i+1]+x[i+2])/5)              # taking 5 elements at a time
        elif i == 1:
            y.append((2*x[i-1]+x[i]+x[i+1]+x[i+2])/5)
        elif i == n-2:
            y.append((x[i-2]+x[i-1]+x[i]+x[i+1]*2)/5)
        elif i == n-1:
            y.append((x[i-2]+x[i-1]+x[i]*3)/5)
        else:
            y.append((x[i-2]+x[i-1]+x[i]+x[i+1]+x[i+2])/5)
    return y


# DTFT function to calculate Discrete-Time Fourier Transform of argument to function (x)
def DTFT(x):
    y = []
    n = len(x)
    for k in range(N):
        s = 0
        for i in range(n):
            s += x[i]*np.exp(-2j * np.pi * k * i/N)         # x[i] is the ith sample of signal x and is multiplied with exponentation
        y.append(s)                                         # and the summation(s) is inserted to the signal y
    return y                                                # the signal y (DTFT of x) is returned


# DTFT_h function to calculate the DTFT of impulse response h
def DTFT_h(h):
    n = len(h)
    l = []
    for k in range(N):
        s = 0
        for i in range(n):
            s += h[i]*np.exp(-2j * np.pi * k * (i-2)/N)     # here we are taking (i-2) intead of i as we did in DTFT
        l.append(s)                                         # reason being the indices of h given in problem are [-2, -1, 0, 1, 2]
    return l


# IFT function to calculate the Inverse Fourier Transform of argument signal (x)
def IFT(x):
    y = []
    n = len(x)
    for k in range(n):
        s = 0
        for i in range(N):
            s += 1/N *x[i] *np.exp(2j * np.pi * k * i/N)    # calculating the summation of exponentials of all samples
        y.append(s)
    return y


# deblur function for Deblurring/ Sharpening the argument signal (y) using the impulse (h)
def deblur(y, h):
    l = []                                                  # creating a list l for storing division of DTFT(y) and DTFT(h)
    Y = DTFT(y)                                             # storing DTFT of y in Y
    H = DTFT_h(h)                                           # storing DTFT of h in H

    for i in range(N):
        if H[i]<0.25:                                       # neglecting values less than 0.25 to prevent the denominator to reach 0
            Hi = 0.25
        else:
            Hi = H[i]

        l.append(Y[i]/Hi)                                   # appending the division to list l

    X = IFT(l)                                              # taking Inverse Fourier Transform of l to get X 
    return X


# function for first denoising and then deblurring the signal (y) using impulse (h)
def first_denoise_then_deblur(y, h):
    denoised_y = denoise(y)                                 # denoising y first and storing in (denoised_y)
    return deblur(denoised_y, h)                            # returning the deblurred signal afterwards


# function for first deblurring and then denoising the signal (y) using impulse (h)
def first_deblur_then_denoise(y, h):
    deblurred_y = deblur(y, h)                              # deblurring signal (y) first using impulse (h), storing in (deblurred_y)
    return denoise(deblurred_y)                             # returning the denoised signal afterwards


# function for calculating Mean Square Error of the signal
def MSE(y):
    mse = 0
    for i in range(193):
        mse += (y[i] - x_n[i])**2
    mse /= 193
    return mse



#initialising the signal variables
x_n = []
y_n = []
h = [1/16, 4/16, 6/16, 4/16, 1/16]


# taking input of the data.csv path
a = input("Please enter the path of the data.csv file: ")
b = a.split()
if len(b)==0 or a[-4:]!=".csv":
    print("\nYou probaby provided the wrong path, default path is taken by the code. Please run the code again if results aren\'t visible.")
    path = "./data.csv"
else:
    path = a


a = pd.read_csv(path)
n = len(a)
print(a.to_string())
x_n = list(a['x[n]'])
y_n = list(a['y[n]'])


N = len(x_n)

x1 = [abs(x) for x in first_denoise_then_deblur(y_n, h)]    # first denoising the signal y[n] and then deblurring it would give x1

x2 = [abs(x) for x in first_deblur_then_denoise(y_n, h)]    # first deblurring the signal y[n] and then denoising it would give x2



#plotting the graphs
plt.plot(range(193), x_n, label="x_n", color="r")           #plotting graphs of x[n] and x1
plt.plot(range(193), x1, label="x1", color="g")
plt.legend()
plt.show()


plt.plot(range(193), x_n, label="x_n", color="r")           #plotting graphs of x[n] and x2
plt.plot(range(193), x2, label="x2", color="g")
plt.legend()
plt.show()


#printing the Mean Squared Errors of y[n], x1 and x2
print("\nMSE of y[n] : " + str(MSE(y_n)))
print("MSE of x1   : " + str(abs(MSE(x1))))
print("MSE of x2   : " + str(abs(MSE(x2)))+"\n")
