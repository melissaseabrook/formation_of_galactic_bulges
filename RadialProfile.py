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

def findeffectiveradius(radialprofile, r):
	totalbrightness=np.sum(radialprofile * 2 * np.pi *r)
	centralbrightness=radialprofile[0]
	cumulativebrightness=np.cumsum(radialprofile * 2 * np.pi *r)
	r_e_unnormalised=((np.abs((totalbrightness/2) - cumulativebrightness)).argmin())
	r_e=r_e_unnormalised*(30.0/len(r))
	i_e= radialprofile[r_e_unnormalised-1]
	print(totalbrightness)
	print(r_e_unnormalised-1)
	print(cumulativebrightness[r_e_unnormalised-1])
	print(i_e)
	return i_e, r_e, centralbrightness, totalbrightness

def SersicProfile(r, I_e, R_e, n):
	b=np.exp(0.6950 + np.log(n) - (0.1789/n))
	G=(r/R_e)**(1/n)
	return I_e*np.exp((-b*(G-1)))


def run_radial_profile(image, imagefile):
	#plots radius vs pixel intensity and its log
	width, height =image.shape[:2]
	cx=int(width/2)
	cy=int(height/2)
	maxVal, center = findcenter(image)
	rad=radial_profile(image,center)
	r= np.linspace(0, 30.0, num=len(rad))
	#SB, r=radialprofiletoSB(rad)
	i_e, r_e, centralbrightness, totalbrightness= findeffectiveradius(rad, r) 
	print("I_e={}, R_e={}".format(i_e, r_e))
	popt1, pcov1=curve_fit(SersicProfile, r, rad, p0=(i_e, r_e, 1), bounds=((0,0,0.001), (np.inf,np.inf,10)))
	popt2, pcov2 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r, rad)
	print("I_e,R_e,n={}".format(popt1))
	print("I_e={}, R_e={}, n={}".format(i_e, r_e, popt2))

	plt.subplot(311)
	plt.plot(r, rad)
	plt.plot(r, SersicProfile(r,*popt1), 'r-', label='I_e={}, R_e={}, n={} parameter fit'.format(round(popt1[0],2),round(popt1[1],2),round(popt1[2],2)))
	plt.plot(r, SersicProfile(r,i_e, r_e, popt2), 'k-', label='I_e={}, R_e={}, fixed, n={} parameter fit'.format(round(i_e,2), round(r_e,2), round(popt2[0],2)))
	plt.title('Radius vs Pixel intensity'), plt.xlabel('Radius (kpc)'), plt.ylabel('Intensity'), plt.xlim(0), plt.ylim(0,250)
	plt.legend()

	plt.subplot(312)
	plt.plot(r, rad)
	plt.plot(r, SersicProfile(r,i_e, r_e, 1), 'r-', label='Exponential Disc, n=1')
	plt.plot(r, SersicProfile(r,i_e, r_e, 4), 'k-', label='De Vaucouleurs, n=4')
	plt.title('Exponential Disc+Vaucouleurs'), plt.xlabel('Radius (kpc)'), plt.ylabel('Intensity'), plt.xlim(0), plt.ylim(0,250)
	plt.legend()

	plt.subplot(313)
	plt.plot(r, rad)
	plt.title('Log'), plt.xlabel('log(Radius) (log(kpc))'), plt.ylabel('log(Intensity)')
	plt.xscale('log')
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig('galaxygraphsbinRecal/radialbrightnessprofile'+imagefile)
	plt.show()

if __name__ == "__main__":
	imagefile='RecalL0025N0752galface_838817.png'
	image=plt.imread('galaxyimagebinRecal/'+imagefile, 0)
	run_radial_profile(image, imagefile)