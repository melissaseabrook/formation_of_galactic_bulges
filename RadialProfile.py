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
	#r=r*300/256
	r=r.astype(np.int)
	image=image.mean(axis=2)
	tbin=np.bincount(r.ravel(),image.ravel()) #sum of image values in each radius bin
	nr=np.bincount(r.ravel()) #no in each radius bin
	radialprofile=(tbin)/(nr)
	return radialprofile

def radialprofiletoSB(radialprofile):
	ind=np.linspace(1, len(radialprofile), num=len(radialprofile))
	SB=radialprofile/(ind*2*np.pi)
	return SB, ind

def findeffectiveradius(radialprofile):
	totalbrightness=np.sum(radialprofile)
	centralbrightness=radialprofile[0]
	cumulativebrightness=np.cumsum(radialprofile)
	r_e=(np.abs((totalbrightness/2) - cumulativebrightness)).argmin() +1
	i_e= radialprofile[r_e-1]
	return totalbrightness, r_e, centralbrightness, i_e

def SersicProfile(r, I_e, R_e, n):
	b=np.exp(0.6950 + np.log(n) - (0.1789/n))
	G=(r/R_e)**(1/n)
	return I_e*np.exp((-b*(G-1)))

def SersicProfile2(r,k,b,n):
	return k*np.exp(-b*(r**(1/n)))

def run_radial_profile(image, imagefile):
	#plots radius vs pixel intensity and its log
	width, height =image.shape[:2]
	cx=int(width/2)
	cy=int(height/2)
	maxVal, center = findcenter(image)
	rad=radial_profile(image,center)
	SB, r=radialprofiletoSB(rad)
	totalbrightness, r_e , centralbrightness, i_e= findeffectiveradius(SB) 
	print("I_e={}, R_e={}".format(i_e, r_e))
	popt1,pcov1=curve_fit(SersicProfile, SB, r, p0=(i_e, r_e, 1), bounds=((0,0,0), (np.inf,np.inf,10)))
	popt2,pcov2 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), SB, r)
	print("I_e,R_e,n={}".format(popt1))
	print("I_e={}, R_e={}, n={}".format(i_e, r_e, popt2))
	plt.subplot(211)
	plt.plot(SB, r)
	plt.plot(SB, SersicProfile(SB,*popt1), 'r-')
	plt.plot(SB, SersicProfile(SB,i_e, r_e, popt2), 'k-')
	plt.title('Radius vs Pixel intensity'), plt.xlabel('Radius (kpc)'), plt.ylabel('Intensity'), plt.xlim(0)
	plt.subplot(212)
	plt.plot(rad)
	plt.title('Log'), plt.xlabel('log(Radius) (log(kpc))'), plt.ylabel('log(Intensity)')
	plt.xscale('log')
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig('galaxygraphsbinRecal/radialbrightnessprofile'+imagefile)
	plt.show()

if __name__ == "__main__":
	imagefile='RecalL0025N0752galface_659536.png'
	image=plt.imread('galaxyimagebinRecal/'+imagefile, 0)
	run_radial_profile(image, imagefile)

	#testcommit