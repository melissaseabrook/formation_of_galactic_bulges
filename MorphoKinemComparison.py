import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from imutils import contours
from skimage import measure
import imutils
import math
from scipy.optimize import curve_fit
from lmfit import Model
from astropy.stats import SigmaClip
import pandas as pd
import os
import seaborn as sns

def sigmaclip(image, sigma, box_size):
	#shows relative fluctuations in pixel intensities
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_background=np.median(gray)
    std_background=np.std(gray)
    for i in np.arange(0,257, box_size):
        for j in np.arange(0,257, box_size):
            i=int(i)
            j=int(j)
			#selects box in image
            area=gray[i:i+box_size, j:j+box_size]
            median=np.median(area)
            std=np.std(area)
			#colours pixel white if bright enough, black if not
            area[(area>(median + (sigma*std))) & (area>(median_background+(2*std_background)))]=255
            area[(area>(median + (sigma*std))) & (area<(median_background+(2*std_background)))]=0
            area[area<(median + (sigma*std))]=0
			#inserts masked box into image
            gray[i:i+box_size, j:j+box_size]=area
    return(gray)
        
def findlightintensity(image, radius, center):
	npix, npiy = image.shape[:2]
	x1 = np.arange(0,npix)
	y1 = np.arange(0,npiy)
	x,y = np.meshgrid(y1,x1)
	r=np.sqrt((x-center[0])**2+(y-center[1])**2)
	#r=r*300/256
	r=r.astype(np.int)
	radius=int(radius)
	image=image.mean(axis=2)
	tbin=np.bincount(r.ravel(),image.ravel()) #sum of image values in each radius bin
	nr=np.bincount(r.ravel()) #no in each radius bin
	radialprofile=(tbin)/(nr)
	cumulativebrightness=np.sum(radialprofile[0:radius])
	return cumulativebrightness


def findcenter(image):
	#finds coords of central bulge
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (11,11), 0)
	(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
	return maxVal, maxLoc


def findandlabelbulge(image, imagefile):
	#locates central bulge and diffuse halo, and marks this on the image
	imagecopy=image.copy()
	median=np.median(image)
	std=np.std(image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred1 = cv2.GaussianBlur(gray, ksize=(11, 11), sigmaX=3,sigmaY=3)
	thresh1 = cv2.threshold(blurred1, median + 5*std, 255, cv2.THRESH_BINARY)[1]
	thresh1 = cv2.erode(thresh1, None, iterations=2)
	#thresh1 = cv2.dilate(thresh1, None, iterations=4)

	blurred2 = cv2.GaussianBlur(gray, ksize=(11, 11), sigmaX=3,sigmaY=3)
	thresh2 = cv2.threshold(blurred2, median +std, 255, cv2.THRESH_BINARY)[1]
	thresh2 = cv2.dilate(thresh2, None, iterations=4)

	#blurred3 = cv2.GaussianBlur(gray, ksize=(11, 11), sigmaX=2,sigmaY=2)
	#thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,61,0)
	thresh3= sigmaclip(image,4,8)

	#find bulge
	# perform a connected component analysis on the thresholded
	# image, then initialize a mask to store only the "large" components
	labels1 = measure.label(thresh1, neighbors=8, background=0)
	mask1 = np.zeros(thresh1.shape, dtype="uint8")
	# loop over the unique components
	for label in np.unique(labels1):
		# if this is the background label, ignore it
		if label == 0:
			continue
		# otherwise, construct the label mask and count the number of pixels 
		labelMask = np.zeros(thresh1.shape, dtype="uint8")
		labelMask[labels1 == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		# if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
		if numPixels > 20:
			mask1 = cv2.add(mask1, labelMask)
	# find the contours in the mask, then sort them from left to right
	cnts = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	if len(cnts)>0:
		cnts = contours.sort_contours(cnts)[0]
		# loop over the contours
		bradius, bcX,bcY=0,0,0
		c=max(cnts, key=cv2.contourArea)
		(x, y, w, h) = cv2.boundingRect(c)
		((bcX, bcY), bradius) = cv2.minEnclosingCircle(c)
		cv2.circle(imagecopy, (int(bcX), int(bcY)), int(bradius),(0, 0, 255), 1)
		print("bulge radius:{},  bulge centre({},{})".format(bradius, bcX,bcY))
		if numPixels > 20: 
			cv2.putText(imagecopy, "bulge", (x, y - 2),
			cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)
	else:
		bradius, bcX,bcY=0,0,0
	#find halo
	labels2 = measure.label(thresh2, neighbors=8, background=0)
	mask2 = np.zeros(thresh2.shape, dtype="uint8")	
	for label in np.unique(labels2):
		if label == 0:
			continue
		labelMask = np.zeros(thresh2.shape, dtype="uint8")
		labelMask[labels2 == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		if numPixels > 20:
			mask2 = cv2.add(mask2, labelMask)
	cnts = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	if len(cnts)>0:
		cnts = contours.sort_contours(cnts)[0]
		hcX, hcY, hradius =0,0,0
		c=max(cnts, key=cv2.contourArea)
		(x, y, w, h) = cv2.boundingRect(c)
		((hcX, hcY), hradius) = cv2.minEnclosingCircle(c)
		cv2.circle(imagecopy, (int(hcX), int(hcY)), int(hradius),(255, 0, 0), 1)
		print("halo radius:{}, halo centre({},{})".format(hradius, hcX,hcY))
		if numPixels > 60: 
			cv2.putText(imagecopy, "halo", (x, y - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)
	else:
		hradius, hcX,hcY=0,0,0
	labels3 = measure.label(thresh3, neighbors=8, background=0)
	mask3 = np.zeros(thresh3.shape, dtype="uint8")
	for label in np.unique(labels3):
		if label == 0:
			continue
		labelMask = np.zeros(thresh3.shape, dtype="uint8")
		labelMask[labels3 == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		if numPixels > 0:
			mask3 = cv2.add(mask3, labelMask)
	cnts = cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	if len(cnts)>0:
		cnts = contours.sort_contours(cnts)[0]
		count=0
		for (i, c) in enumerate(cnts):
			if numPixels<10:
				(x, y, w, h) = cv2.boundingRect(c)
				cv2.drawContours(imagecopy, c, 0, (0,255,0),1)
				count+=1
	else:
		count=0

	bthradius=bradius/hradius
	print("halo radius:bulge radius ={}".format(bthradius))
	print("star count ={}".format(count))
	halo_intensity=findlightintensity(image, hradius, (hcX,hcY))
	bulge_intensity=findlightintensity(image, bradius, (bcX,bcY))
	bthintensity= bulge_intensity/halo_intensity
	print("halo intensity = {}, bulge intensity ={}, halo:bulge intensity ={}".format(halo_intensity, bulge_intensity, bthintensity))
	cv2.imwrite('galaxygraphsbinRecal/opencvfindbulge'+imagefile, imagecopy)
	return bthradius, bthintensity

def invertbth(r):
    if r !=0:
        return 1/r
    else:
        return 0

def plotbulgetodisc(df):
    df['htbradius']=df.apply(lambda x: invertbth(x.bthradius), axis=1)
    df['htbintensity']=df.apply(lambda x: invertbth(x.bthintensity), axis=1)
    sns.scatterplot(x='DiscToTotal',y='bthradius', data=df)
    sns.scatterplot(x='DiscToTotal',y='bthintensity', data=df)
    plt.show()
    sns.scatterplot(x='DiscToTotal',y='htbradius', data=df)
    sns.scatterplot(x='DiscToTotal',y='htbintensity', data=df)
    plt.show()
    sns.scatterplot(x='bthradius',y='bthintensity', data=df)
    plt.show()


if __name__ == "__main__":
    
    read_data=True
    if(read_data):
        df=pd.read_csv('EAGLEbulgehalo.csv')
    else:
        df=pd.read_csv('EAGLEimagesdf.csv')
        halobulgetemp=[]
        for filename in df['filename']:
            BGRimage=cv2.imread('galaxyimagebinRecal/'+filename)
            bthradius, bthintensity =findandlabelbulge(BGRimage, filename)
            halobulgetemp.append([filename, bthradius, bthintensity])
        halobulgedf=pd.DataFrame(halobulgetemp, columns=['filename', 'bthradius', 'bthintensity'])
        print(halobulgedf)
        print(df)
        df.filename.astype(str)
        halobulgedf.filename.astype(str)
        df=pd.merge(df, halobulgedf, on=['filename'], how='outer')
        df.to_csv('EAGLEbulgehalo.csv')

    plotbulgetodisc(df)


    


