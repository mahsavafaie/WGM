import cv2
import numpy as np
import os
import glob

def separate_green(image):
    img= cv2.imread(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (36,25,25), (86, 255, 255))
    ## slice the green
    imask = mask_green>0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]
    ## convert green to black and background to white
    green[np.where((green==[0,0,0]).all(axis=2))] = [255,255,255]
    green[np.where((green==[0,255,0]).all(axis=2))] = [0,0,0]
    return(green)

def separate_red(image):
    img= cv2.imread(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    imask = mask_red>0
    red = np.zeros_like(img, np.uint8)
    red[imask] = img[imask]
    red[np.where((red==[0,0,0]).all(axis=2))] = [255,255,255]
    red[np.where((red==[0,0,255]).all(axis=2))] = [0,0,0]
    return(red)

path = "C:/Users/mahsa/Desktop/From server/handwritten_photo/bin_mf_b/"

for image in glob.glob(path + "out_crf/*_out_crf.png"):
    print("green layer seperated from " + os.path.basename(image))
    cv2.imwrite(path + "crf_green/"+os.path.basename(image).split(".png")[0]+"_green.png", separate_green(image))
    print("red layer seperated from " + os.path.basename(image))
    cv2.imwrite(path + "crf_red/"+os.path.basename(image).split(".png")[0]+"_red.png", separate_red(image))
print("handwritten layer seperation finished!")
for image in glob.glob(path + "out/*_out.png"):
    print("green layer seperated from " + os.path.basename(image))
    cv2.imwrite(path + "green/"+os.path.basename(image).split(".png")[0]+"_green.png", separate_green(image))
    print("red layer seperated from " + os.path.basename(image))
    cv2.imwrite(path + "red/"+os.path.basename(image).split(".png")[0]+"_red.png", separate_red(image))
print("printed layer seperation finished!")
