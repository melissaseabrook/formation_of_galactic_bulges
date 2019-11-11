import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from imutils import contours
from skimage import measure
import imutils
import math
from scipy.optimize import curve_fit
from scipy import stats
from lmfit import Model
import seaborn as sns

def findcenter(image):
	#finds coords of central bulge
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (11,11), 0)
	(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
	return maxVal, maxLoc

def radial_profile(image, center):
	#returns average pixel intensity for all possible radius, centred around the central bulge
	npix, npiy = image.shape[:2]
	x1 = np.arange(0,npix)
	y1 = np.arange(0,npiy)
	x,y = np.meshgrid(y1,x1)
	r=np.sqrt((x-center[0])**2+(y-center[1])**2)
	r=r.astype(np.int)
	image=image.mean(axis=2)
	tbin=np.bincount(r.ravel(),image.ravel()) #sum of image values in each radius bin
	nr=np.bincount(r.ravel()) #no in each radius bin
	radialprofile=(tbin)/(nr)
	stdbins, bin_edges, binnumber=stats.binned_statistic(r.ravel(),image.ravel(), 'std', bins=len(radialprofile))
	stdbins[0]=stdbins[1]
	stdbins[stdbins<0.5]=0.5
	#meanbins, bin_edges, binnumber=stats.binned_statistic(r.ravel(),image.ravel(), 'mean', bins=len(radialprofile))
	return radialprofile, stdbins

def findeffectiveradius(radialprofile, r):
	totalbrightness=np.sum(radialprofile * 2 * np.pi *r)
	centralbrightness=radialprofile[0]
	cumulativebrightness=np.cumsum(radialprofile * 2 * np.pi *r)
	r_e_unnormalised=((np.abs((totalbrightness/2) - cumulativebrightness)).argmin())
	r_e=r_e_unnormalised*(30.0/256)/np.sqrt(2)
	i_e= radialprofile[r_e_unnormalised-1]
	return i_e, r_e, centralbrightness, totalbrightness

def findlightintensity(radialpofile, radius):
	radius=int(radius)
	cumulativebrightness=np.sum(radialpofile[0:radius])
	return cumulativebrightness

def SersicProfile(r, I_e, R_e, n):
	b=np.exp(0.6950 + np.log(n) - (0.1789/n))
	G=(r/R_e)**(1/n)
	return I_e*np.exp((-b*(G-1)))

def twocomponentmodel(r, I_e, R_e, n):
	b=np.exp(0.6950 + np.log(n) - (0.1789/n))
	I1= I_e*np.exp((-b*((r/R_e)**(1/n)-1)))
	I2= I_e*np.exp((-1.68*((r/R_e)-1)))
	return I1+I2

def findbulge(image, imagefile):
	#locates central bulge and diffuse halo, and marks this on the image
	imagecopy=image.copy()
	median=np.median(image)
	std=np.std(image)
	print(median)
	print(std)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred1 = cv2.GaussianBlur(gray, ksize=(11, 11), sigmaX=3,sigmaY=3)
	thresh1 = cv2.threshold(blurred1, median + 5*std, 255, cv2.THRESH_BINARY)[1]
	thresh1 = cv2.erode(thresh1, None, iterations=2)
	#thresh1 = cv2.dilate(thresh1, None, iterations=4)
	blurred2 = cv2.GaussianBlur(gray, ksize=(11, 11), sigmaX=3,sigmaY=3)
	thresh2 = cv2.threshold(blurred2, median +std, 255, cv2.THRESH_BINARY)[1]
	thresh2 = cv2.dilate(thresh2, None, iterations=4)
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
	print("bulge radius:{},  bulge centre({},{})".format(bradius, bcX,bcY))
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
	print("halo radius:{}, halo centre({},{})".format(hradius, hcX,hcY))
	return bradius,hradius, (hcX,hcY), (bcX,bcY)

def run_radial_profile(image, imagefile):
	#plots radius vs pixel intensity and its log
	width, height =image.shape[:2]
	cx=int(width/2)
	cy=int(height/2)
	maxVal, center = findcenter(image)
	rad, stdbins=radial_profile(image,center)
	max_r=np.sqrt(2)*15
	r= np.linspace(0, max_r, num=len(rad))
	
	i_e, r_e, centralbrightness, totalbrightness= findeffectiveradius(rad, r) 
	bindex,hindex, (hcX,hcY), (bcX,bcY) = findbulge(image, imagefile)
	hintensity=findlightintensity(rad,hindex)
	bintensity=findlightintensity(rad,hindex)
	r=r[1:int(hindex)]
	rad=rad[1:int(hindex)]
	stdbins=stdbins[1:int(hindex)]
	b1=bindex*30/256
	h1=hindex*30/256
	bindex=int(bindex)
	hindex=int(hindex)


	print("I_e={}, R_e={}".format(i_e, r_e))
	print("brad={}, hrad={}, bind={}, hind={}".format(b1,h1,bindex,hindex))
	popt1, pcov1=curve_fit(SersicProfile, r, rad, p0=(i_e, r_e, 1), bounds=((0,0,0.001), (np.inf,np.inf,10)), sigma=stdbins*6, absolute_sigma=True)
	popt2, pcov2 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r, rad, p0=3, sigma=stdbins*2, absolute_sigma=True)
	popt3, pcov3 = curve_fit(lambda x,n: twocomponentmodel(x, i_e, r_e, n), r, rad, sigma=stdbins*2, absolute_sigma=True)
	print("I_e,R_e,n={}".format(popt1))
	print("I_e={}, R_e={}, n={}".format(i_e, r_e, popt2))
	print("I_e={}, R_e={}, n={}".format(i_e, r_e, popt3))

	sns.set(style='whitegrid')
	fig=plt.figure()
	plt.subplot(211)
	plt.errorbar(r, rad, yerr=(stdbins*2), fmt='', color='k', capsize=0.5, elinewidth=0.5)
	plt.plot(r, SersicProfile(r,*popt1), 'r-', label='I_e={}, R_e={}, n={} free'.format(round(popt1[0],2),round(popt1[1],2),round(popt1[2],2)))
	plt.plot(r, SersicProfile(r,i_e, r_e, popt2 +pcov2[0,0]), 'g--')
	plt.plot(r, SersicProfile(r,i_e, r_e, popt2), 'g', label='I_e={}, R_e={}, fixed, n={} free'.format(round(i_e,2), round(r_e,2), round(popt2[0],2)))
	plt.plot(r, SersicProfile(r,i_e, r_e, popt2 -pcov2[0,0]) , 'g--')
	plt.fill_between(r, SersicProfile(r,i_e, r_e, popt2 -pcov2[0,0]**0.5), SersicProfile(r,i_e, r_e, popt2 +pcov2[0,0]**0.5), facecolor='gray', alpha=0.5)
	plt.title('Radius vs Pixel intensity'), plt.xlabel('Radius (kpc)'), plt.ylabel('Intensity'), plt.xlim(0), plt.ylim(0,250)
	plt.legend()
	
	plt.subplot(212)
	plt.errorbar(r, rad, yerr=(stdbins*2), fmt='', color='k',  capsize=0.5, elinewidth=0.5)
	plt.plot(r, SersicProfile(r,i_e, r_e, 1), 'r-', label='Exponential Disc, n=1')
	plt.plot(r, SersicProfile(r,i_e, r_e, 4), 'b-', label='De Vaucouleurs, n=4')
	plt.plot(r, twocomponentmodel(r,i_e, r_e, popt3), 'g-', label='2 component model')
	plt.title('Exponential Disc+Vaucouleurs, 2 component fit'), plt.xlabel('Radius (kpc)'), plt.ylabel('Intensity'), plt.xlim(0), plt.ylim(0,250)
	plt.legend()
	plt.tight_layout()
	plt.show()
	plt.savefig('galaxygraphsbinRecal/radialbrightnessprofile'+imagefile)

	#2 component fitting

	popt21, pcov21 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r[0:bindex], rad[0:bindex],sigma=stdbins[0:bindex]*2, absolute_sigma=True)
	popt22, pcov22 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r[bindex:hindex], rad[bindex:hindex], sigma=stdbins[bindex:hindex]*2, absolute_sigma=True)
	print("n1={}".format(popt21))
	print("n2={}".format(popt22))

	sns.set(style='whitegrid')
	plt.subplot(211)
	plt.errorbar(r[0:hindex], rad[0:hindex], yerr=(stdbins[0:hindex]*2), fmt='', color='k', capsize=0.5, elinewidth=0.5)
	plt.plot(r[0:bindex], SersicProfile(r[0:bindex],i_e, r_e,popt21), 'r-', label='bulge n={}'.format(round(popt21[0],2)))
	#plt.plot(r, SersicProfile(r,i_e, r_e, popt2 +pcov2[0,0]), 'g--')
	plt.plot(r[bindex:hindex], SersicProfile(r[bindex:hindex],i_e, r_e, popt22), 'g', label='halo n={} free'.format(round(popt22[0],2)))
	#plt.plot(r, SersicProfile(r,i_e, r_e, popt2 -pcov2[0,0]) , 'g--')
	#plt.fill_between(r, SersicProfile(r,i_e, r_e, popt2 -pcov2[0,0]**0.5), SersicProfile(r,i_e, r_e, popt2 +pcov2[0,0]**0.5), facecolor='gray', alpha=0.5)
	plt.title('Radius vs Pixel intensity'), plt.xlabel('Radius (kpc)'), plt.ylabel('Intensity'), plt.xlim(0), plt.ylim(0,250)
	plt.legend()

	plt.subplot(212)
	plt.errorbar(r[0:hindex], rad[0:hindex], yerr=(stdbins[0:hindex]*2), fmt='', color='k', capsize=0.5, elinewidth=0.5)
	plt.plot(r[0:bindex], SersicProfile(r[0:bindex],i_e, r_e, 4), 'b-', label='De Vaucouleurs, n=4')
	plt.plot(r[bindex:hindex], SersicProfile(r[bindex:hindex],i_e, r_e, 1), 'r-', label='Exponential Disc, n=1')
	plt.title('Exponential Disc+Vaucouleurs'), plt.xlabel('Radius (kpc)'), plt.ylabel('Intensity'), plt.xlim(0), #plt.ylim(0,250)
	plt.legend()
	plt.tight_layout()
	plt.show()
	plt.savefig('galaxygraphsbinRecal/2componentradialbrightnessprofile'+imagefile)



	"""
	plt.plot(r, rad)
	plt.title('Log'), plt.xlabel('log(Radius) (log(kpc))'), plt.ylabel('log(Intensity)')
	plt.xscale('log')
	plt.yscale('log')
	plt.tight_layout()
	plt.show()
	"""

if __name__ == "__main__":
	imagefile='RecalL0025N0752galface_924755.png'
	image=plt.imread('galaxyimagebinRecal/'+imagefile, 0)
	run_radial_profile(image, imagefile)