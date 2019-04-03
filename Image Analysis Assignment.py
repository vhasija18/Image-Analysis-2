import cv2
import numpy as np
from spicy import ndimage

## Histogram Equalization filter
def histogram_equlization(gray_image,n,m):
    h_equ= cv2.equalizeHist(gray_image)
    cv2.imshow('Histogram Equalization',h_equ)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return h_equ
## Power Law Transformation
def power_law_transform(gray_image,gamma):
     invGamma = 1.0 / gamma
     table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype("uint8")
     power_law = cv2.LUT(gray_image,table)
     cv2.imshow('Pwer law transform',power_law)
     cv2.waitKey(0)
     cv2.destroyAllWindows()
     return power_law
## Median Blur 
def median_blur(gray_image):
    median_filter = cv2.medianBlur(gray_image,3)
    cv2.imshow('Median Blur',median_filter)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return median_filter

## Median using ndimage
def median_ndimage(gray_image):
    median_filtered = ndimage.median_filter(gray_image,size=30)
    cv2.imshow('Median filter',median_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return median_filtered

## Contrast Limited Adaptive Histogram Equalization
def CLAHE(gray_image):
    clahe=cv2.createCLAHE(clipLimit =50.0,tileGridSize=(2,2))
    cl1=clahe.apply(gray_image)
    cv2.imshow('Contrast limited Adaptive Histogram Eq.',cl1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cl1

## 2D filtering 
def filtering_2D(gray_image):
    kernel = np.ones((3,3),np.float32)/9
    filter_2D = cv2.filter2D(gray_image,-1,kernel)
    cv2.imshow('2D filtering', filter_2D)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return filter_2D

## Edge detection using Canny filter
def canny_edgedetector(img):S
    edges = cv2.Canny(img,120,180)
    cv2.imshow('Canny Edge Detector',edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 

## Applying sobel filter on the processed image finding x and y gradient 
def sobel_x_y(img):
    ddepth = cv2.CV_16S
    delta = 0
    scale =1
    grad_y = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_x = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    cv2.imshow('Sobel X Gradient',abs_grad_x)
    cv2.imshow('Sobel Y Gradient',abs_grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv2.imshow("window_name", grad)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

## Applying prewit filter on the processed image finding x and y gradient
def prewit(img):
    kernel_x = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernel_y = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    prewit_x = cv2.filter2D(img,-1,kernel_x)
    prewit_y = cv2.filter2D(img,-1,kernel_y)
    cv2.imshow("Prewit X Gradient", prewit_x)
    cv2.imshow("Prewit Y Gradient", prewit_y)
    cv2.imshow("Final Prewit ",prewit_x+prewit_y )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


print("This program is used for enhacnce the image using various filters including linear and non-linear. After enhancing one can  apply sobel x edge detector to detect edges in x direction(horizontal) or sobel y detector to detect edges in y direction(vertical direction) or sobel edge detector which will detect edges in both horizontal and vertical axis or canny edge detector")
img =cv2.imread(input("Enter a image location and name. Note(While entering location use '/'  instead of '\'.) "))
img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print("Which type of filter you want") 
choice = int(input("Enter\n 1. for Histogram Equalization\n 2. for power law filter\n 3. for median filter\n 4. for Contrast limited Adaptive Histogram Equalization\n 5. for  2D filtering"  ))
if(choice == 1):
    filtered_image = histogram_equlization(img,120,180)
elif(choice == 2):
    gamma = float(input("Enter the value of gamma (gamma>1 increase brightness while gamma<1 increase darkness)"))
    filtered_image = power_law_transform(img,gamma)
elif(choice == 3):
    filtered_image = median_ndimage(img)
elif(choice == 4):
    filtered_image = CLAHE(img)
elif(choice == 5):
    filtered_image = filtering_2D(img)
blur_choice = input("Do you want to perform median blur? Enter Y/N.")
if(blur_choice == 'Y' or blur_choice == 'y'):
    filtered_image = median_blur(filtered_image)
print("Which type of edge detector you would like to use")
edge_choice= int(input("Enter\n  1. for Sobel\n  2. for Prewit\n  3. for Canny filter"))
if(edge_choice == 1):
    sobel_x_y(filtered_image)
elif(edge_choice == 2):
    prewit(filtered_image)
elif(edge_choice == 3):
    canny_edgedetector(filtered_image)

