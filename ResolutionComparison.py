"""Compare the effects of using different resolutions of simulations"""

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
import seaborn as sns
import pandas as pd

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
	image=np.average(image, axis=2, weights=[0.2126,0.587,0.114])
	tbin=np.bincount(r.ravel(),image.ravel()) #sum of image values in each radius bin
	nr=np.bincount(r.ravel()) #no in each radius bin
	radialprofile=(tbin)/(nr)
	stdbins, bin_edges, binnumber=stats.binned_statistic(r.ravel(),image.ravel(), 'std', bins=len(radialprofile))
	countbins, bin_edges, binnumber=stats.binned_statistic(r.ravel(),image.ravel(), 'count', bins=len(radialprofile))
	#stdbins=stdbins/countbins
	stdbins[0:4]=np.max(stdbins)
	stdbins[stdbins<0.01]=0.01
	radialprofile, r_arr, binnumber=stats.binned_statistic(r.ravel(),image.ravel(), 'mean', bins=len(radialprofile))
	return radialprofile, stdbins, r_arr, nr

def nan_helper(y):
	return np.isnan(y), lambda z:z.nonzero()[0]

def vary_radial_profile(image, center, bintype):
	#returns average pixel intensity for all possible radius, centred around the central bulge
	npix, npiy = image.shape[:2]
	x1 = np.arange(0,npix)
	y1 = np.arange(0,npiy)
	x,y = np.meshgrid(y1,x1)
	r=np.sqrt((x-center[0])**2+(y-center[1])**2)
	r=r.astype(np.int)
	max_r=r.max()
	image=np.average(image, axis=2, weights=[0.2126,0.587,0.114])
	#im=image.astype(np.int)
	tbin=np.bincount(r.ravel(),image.ravel())  #sum of image values in each radius bin
	nr=np.bincount(r.ravel()) #no in each radius bin
	radialprofile=(tbin)/(nr)
	rads=np.linspace(0.0001, max_r, len(nr))
	if bintype=='equalradius':
		bins=rads
		print(bintype)
	elif bintype=='equallogradius':
		rads=np.linspace(0, np.log10(max_r), len(nr))
		bins=10**rads
		#bins=np.power(bins, 0.1)
		print(bintype)
	elif bintype=='equalintenisty':
		totalintensity=np.sum(image)
		cumulativebrightness=np.cumsum(tbin)
		bins, binedge,binnumber=stats.binned_statistic(cumulativebrightness, rads, 'max', bins=len(nr))
		nans, x =nan_helper(bins)
		bins[nans]=np.interp(x(nans), x(~nans),bins[~nans])
		print(bintype)

	elif bintype=='equallogintensity':
		logimage=np.log10(image)
		logtbin=np.bincount(r.ravel(),logimage.ravel())
		cumulativebrightness=np.cumsum(tbin)
		bins, binedge,binnumber=stats.binned_statistic(cumulativebrightness, rads, 'max', bins=len(nr))
		nans, x =nan_helper(bins)
		bins[nans]=np.interp(x(nans), x(~nans),bins[~nans])
		print(bintype)
	radialprofile, r_arr, binnumber=stats.binned_statistic(r.ravel(),image.ravel(), 'mean', bins=bins)
	nans, x =nan_helper(radialprofile)
	radialprofile[nans]=np.interp(x(nans), x(~nans),radialprofile[~nans])
	stdbins, bin_edges, binnumber=stats.binned_statistic(r.ravel(),image.ravel(), 'std', bins=bins)
	nans, x =nan_helper(stdbins)
	stdbins[nans]=np.interp(x(nans), x(~nans),stdbins[~nans])
	#countbins, bin_edges, binnumber=stats.binned_statistic(r.ravel(),image.ravel(), 'count', bins=len(r))
	#stdbins=stdbins/countbins

	stdbins[0:4]=np.max(stdbins)
	stdbins[stdbins<0.01]=0.01
	
	return radialprofile, stdbins, r_arr, nr

def findeffectiveradius(radialprofile, r, nr):
	#find effective raidus
	totalbrightness=np.sum(radialprofile * 2 * np.pi *nr*r)
	centralbrightness=radialprofile[0]
	cumulativebrightness=np.cumsum(radialprofile * 2 * np.pi *nr*r)
	r_e_index=((np.abs((totalbrightness/2) - cumulativebrightness)).argmin())
	r_e=r[r_e_index]
	#r_e=r_e_unnormalised*(30.0/256)
	i_e= radialprofile[r_e_index]
	return i_e, r_e, centralbrightness, totalbrightness

def findlightintensity(radialpofile, radius):
	#find total light intensity within a given radius
	radius=int(radius)
	cumulativebrightness=np.sum(radialpofile[0:radius])
	return cumulativebrightness

def SersicProfile(r, I_e, R_e, n):
	#Sersic Profile function
	b=(1.992*n)-0.3271
	G=(r/R_e)**(1/n)
	return I_e*np.exp((-b*(G-1)))

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

def findbulge(sim_name, image, imagefile, sigma_bulge, sigma_disc):
	#locates central bulge and diffuse disc, and marks this on the image
	image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	imagecopy=image.copy()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	median=np.median(gray)
	mean=np.mean(gray)
	std=np.std(gray)
	blurred1 = cv2.GaussianBlur(gray, ksize=(7,7), sigmaX=4,sigmaY=4)
	blurred2 = cv2.GaussianBlur(gray, ksize=(15, 15), sigmaX=3,sigmaY=3)
	thresh2 = cv2.threshold(blurred2, sigma_disc*std +median, 255, cv2.THRESH_BINARY)[1]
	thresh2 = cv2.dilate(thresh2, None, iterations=4)

	thresh3= sigmaclip(image,4,8)

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
	print('count={}, std={}, median={}, mean={}'.format(count,std,median,mean))
	if (std>20):
		thresh1 = cv2.threshold(blurred1,  (sigma_bulge-3)*std+median, 255, cv2.THRESH_BINARY)[1]
		thresh1 = cv2.erode(thresh1, None, iterations=2)
	else:
		thresh1 = cv2.threshold(blurred1,  (sigma_bulge)*std +median, 255, cv2.THRESH_BINARY)[1]
		thresh1 = cv2.erode(thresh1, None, iterations=2)
		


	labels1 = measure.label(thresh1, neighbors=8, background=0)
	mask1 = np.zeros(thresh1.shape, dtype="uint8")
	for label in np.unique(labels1):
		# if this is the background label, ignore it
		if label == 0:
			continue
		# otherwise, construct the label mask and count the number of pixels 
		labelMask = np.zeros(thresh1.shape, dtype="uint8")
		labelMask[labels1 == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		# if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
		if numPixels > 10:
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
		if numPixels > 10: 
			cv2.putText(imagecopy, "bulge", (x, y - 2),
				cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)
	else:
		((bcX, bcY), bradius) = ((0,0),0)
	#find disc
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
		if numPixels > 60: 
			cv2.putText(imagecopy, "disc", (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)
	else:
		((hcX, hcY), hradius) = ((0,0),0)
	cv2.imwrite('ResolutionComparison/opencvbulgedisc'+imagefile, imagecopy)
	cv2.destroyAllWindows()
	return bradius,hradius, (hcX,hcY), (bcX,bcY)

def plotchisquared(rad, r, i_e, r_e, stdbins, bindex, bhindex, hindex, i_ebulge,r_ebulge,isolated_bulge, nr, sim_name):
	#calculateSersicIndices()
	#chi vs n for n_disc
	chisdisc=[]
	chisbulge=[]
	ndisc=[]
	nbulge=[]
	chisdisca=[]
	chisbulgea=[]
	ndisca=[]
	nbulgea=[]
	ntot=[]
	chistot=[]
	for n in np.linspace(0.15,5.0, 100):
		n1, pcov1 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r, rad, p0=n, bounds=(n-0.001,n+0.001), sigma=stdbins, absolute_sigma=True)
		n1=n1[0]
		res= np.abs(rad - SersicProfile(r, i_e,r_e,n))
		n1_error=np.sqrt(sum((res[bindex:bhindex]/stdbins[bindex:bhindex])**2))
		#n1_error=pcov1[0,0]
		
		poptdisc, pcovdisc = curve_fit(lambda x,ni: SersicProfile(x, i_e, r_e, ni), r[bhindex:hindex], rad[bhindex:hindex], sigma=stdbins[bhindex:hindex], p0=n, bounds=(n-0.001,n+0.001), absolute_sigma=True)
		n_disc=poptdisc[0]
		#n_disc_error=pcovdisc[0,0]	
		isolated_discsim=SersicProfile(r, i_e, r_e, n)
		res= np.abs(rad- isolated_discsim)
		n_disc_error=np.sqrt(sum((res[bindex:hindex]/stdbins[bindex:hindex])**2))
		poptbulge, pcovbulge = curve_fit(lambda x,ni: SersicProfile(x, i_ebulge, r_ebulge, ni), r[0:bindex], isolated_bulge[0:bindex], sigma=stdbins[0:bindex], p0=n, bounds=(n-0.001,n+0.001), absolute_sigma=True)
		n_bulge= poptbulge[0]
		#n_bulge_error= pcovbulge[0,0]
		res= np.abs(isolated_bulge- SersicProfile(r, i_ebulge, r_ebulge, n_bulge))
		n_bulge_error=np.sqrt(sum((res[2:bindex]/stdbins[2:bindex])**2))
		
		poptdisca, pcovdisca=curve_fit(SersicProfilea, r[bindex:hindex], rad[bindex:hindex], p0=(i_e, r_e, n,0), bounds=((i_e-1,r_e-0.1,n-0.001,0), (i_e+1,r_e+0.1,n+0.001,20)), sigma=stdbins[bindex:hindex], absolute_sigma=True)
		n_disca=poptdisca[2]
		#n_disc_errora=pcovdisca[2,2]
		isolated_discsima=SersicProfilea(r, poptdisca[0], poptdisca[1],n_disca,poptdisca[3])
		res= np.abs(rad - isolated_discsim)
		n_disc_errora=np.sqrt(sum((res[bhindex:hindex]/stdbins[bhindex:hindex])**2))

		isolated_bulgea= rad - isolated_discsima
		isolated_bulgea[isolated_bulgea<0]=0
		i_ebulgea, r_ebulgea, centralbrightnessbulgea, totalbrightnessbulgea= findeffectiveradius(isolated_bulgea[0:bhindex], r[0:bhindex], nr[0:bhindex]) 
		poptbulgea, pcovbulgea=curve_fit(SersicProfilea, r[0:bindex], isolated_bulgea[0:bindex], p0=(i_ebulgea, r_ebulgea, n,0), bounds=((i_ebulgea-1,r_ebulgea-0.1,n-0.001,0), (i_ebulgea+1,r_ebulgea+0.1,n+0.001,20)), sigma=stdbins[0:bindex], absolute_sigma=True)
		n_bulgea= poptbulgea[2]
		#n_bulge_errora=pcovdisca[2,2]
		res= np.abs(rad - isolated_discsima-SersicProfilea(r, i_ebulgea, r_ebulgea, n_bulgea, poptbulgea[3]))
		n_bulge_errora=np.sqrt(sum((res[0:int(bindex/2)]/stdbins[0:int(bindex/2)])**2))
		
		chisdisc.append(n_disc_error)
		chisbulge.append(n_bulge_error)
		ndisc.append(n_disc)
		nbulge.append(n_bulge)
		chisdisca.append(n_disc_errora)
		chisbulgea.append(n_bulge_errora)
		ndisca.append(n_disca)
		nbulgea.append(n_bulgea)
		ntot.append(n1)
		chistot.append(n1_error)
	chisdisca=np.array(chisdisca)
	plt.plot(ntot, chistot, label='ntot')
	plt.plot(ndisca, chisdisca+0.5, label='ndisca')
	plt.plot(nbulgea, chisbulgea,label='nbulgea')
	plt.plot(ndisc, chisdisc, label='ndisc')
	plt.plot(nbulge, chisbulge,label='nbulge')
	plt.legend(), plt.xlim(0), plt.ylim(0)
	#plt.yscale('log')
	plt.xlabel('n'), plt.ylabel('$\chi ^2$')#, plt.ylim(0,0.025)
	plt.title('Plot of $\chi ^2$ generated for each n')
	plt.tight_layout()
	plt.savefig('galaxygraphsbin'+str(sim_name)+'/chisqrdvsn'+str(imagefile))
	plt.show()

def plotradialprofile(rad, r, i_e, r_e, stdbins, bindex, bhindex, hindex, n1_error, n_disc_error,n_bulge_error,isolated_discsim,isolated_bulge,isolated_bulgesim, totalsim, n_disc,n_bulge, n1, sim_name, imagefile, sigma_bulge,sigma_disc,radbintype):
	#plot radial profile and fits
	fig=plt.figure(figsize=(15,8))
	plt.subplot(221)
	n_total=n1
	plt.errorbar(r, rad, yerr=(stdbins), fmt='', color='k', capsize=0.5, elinewidth=0.5)
	plt.plot(r, SersicProfile(r,i_e, r_e, n_total) +n1_error, 'g--')
	plt.plot(r, SersicProfile(r,i_e, r_e, n_total), 'g', label='n={} +/- {}'.format(round(n_total,2), round(n1_error,2)))
	plt.plot(r, SersicProfile(r,i_e, r_e, n_total)- n1_error, 'g--')
	plt.fill_between(r, SersicProfile(r,i_e, r_e, n_total) -n1_error, SersicProfile(r,i_e, r_e, n_total) +n1_error, facecolor='gray', alpha=0.5)
	plt.title('Total Profile Fit'), plt.xlabel('Radius (kpc)'), plt.ylabel('Intensity'), plt.xlim(0), plt.ylim(0,250)
	plt.legend(bbox_to_anchor=(1.05,1), loc=2)

	plt.subplot(223)
	plt.errorbar(r, rad, yerr=(stdbins), fmt='', color='k', capsize=0.5, elinewidth=0.5, label='observed')
	plt.errorbar(r, isolated_discsim[0:hindex], yerr=stdbins[0:hindex], fmt='', color='r', capsize=0.5, elinewidth=0.5, label='disc sim, n_disc={} +/ {}'.format(round(n_disc,2),round(n_disc_error,2)))
	plt.errorbar(r, isolated_bulge[0:bindex], yerr=stdbins[0:bindex], fmt='', color='g', capsize=0.5, elinewidth=0.5, label='bulge observed')
	plt.errorbar(r[0:bindex], isolated_bulgesim[0:bindex], yerr=(n_bulge_error), fmt='', color='b', capsize=0.5, elinewidth=0.5, label='bulge sim, n_bulge={} +/ {}'.format(round(n_bulge,2), round(n_bulge_error,2)))
	plt.plot(r[0:hindex],totalsim, color='y', label='total sim')
	plt.legend(bbox_to_anchor=(1.05,1), loc=2)
	plt.title('Bulge Extracted Fit'), plt.xlabel('Radius (kpc)'), plt.ylabel('Intensity'), plt.xlim(0), plt.ylim(0,250)

	plt.subplot(122)
	cv2image=plt.imread('ResolutionComparison/opencvbulgedisc'+imagefile, 0)
	plt.imshow(cv2image)
	plt.title('image')
	plt.tight_layout()
	plt.savefig('ResolutionComparison/RadialProfile'+imagefile)
	plt.close()
	#plt.show()

def calculaten_total(rad, r, i_e, r_e, stdbins, bindex, bhindex, hindex, nr):
	try:
		n1, pcov1 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r, rad, p0=2, bounds=(0.01,8), sigma=stdbins, absolute_sigma=True)
		n_total=n1[0]
		n_error=pcov1[0]
	except:
		n_total=[np.nan]
		n_error=[np.nan]
		print('n1 nan')
	print("I_e={}, R_e={}, n_disc={}".format(i_e, r_e, n_total))
	

	try:
		poptdisc, pcovdisc = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r[bindex:bhindex], rad[bindex:bhindex], p0=1, sigma=stdbins[bindex:bhindex], bounds=(0,2), absolute_sigma=True)
		n_disc=poptdisc[0]
		n_disc_error=pcovdisc[0,0]
		print("I_edisc={}, R_edisc={}, n_disc={}".format(i_e, r_e, n_disc))
		isolated_discsim=SersicProfile(r, i_e, r_e, n_disc)
		isolated_bulge= rad - isolated_discsim
		isolated_bulge[isolated_bulge<0]=0
		i_ebulge, r_ebulge, centralbrightnessbulge, totalbrightnessbulge= findeffectiveradius(isolated_bulge[0:bhindex], r[0:bhindex], nr[0:bhindex]) 
		poptbulge, pcovbulge = curve_fit(lambda x,n: SersicProfile(x, i_ebulge, r_ebulge, n), r[0:bindex], isolated_bulge[0:bindex],p0=3, sigma=stdbins[0:bindex], bounds=(0,10), absolute_sigma=True)
		n_bulge= poptbulge[0]
		n_bulge_error= pcovbulge[0,0]
		print("I_ebulge={}, R_ebulge={}, n_bulge={}".format(i_ebulge,r_ebulge, n_bulge))
	except:
		n_disc=n_bulge=n_disc_error=n_bulge_error=np.nan
		print('n nan')
	return n_total, n_error,n_disc,n_bulge,n_disc_error,n_bulge_error

def calculateSersicIndices(rad, r, i_e, r_e, stdbins, bindex, bhindex, hindex, nr, sim_name, imagefile,sigma_bulge,sigma_disc,radbintype):
	a=np.empty((3,3))
	a=a.fill(np.nan)
	try:
		n1, pcov1 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r, rad, p0=2, bounds=(0.1,8), sigma=stdbins, absolute_sigma=True)
		n1=round(n1[0],2)
		res= np.abs(rad - SersicProfile(r, i_e,r_e,n1))
		n1_error=np.sqrt(sum((res[bindex:bhindex]/stdbins[bindex:bhindex])**2))
		n1_error=round(np.sqrt(n1_error),2)
	except:
		n1=0
		n1_error=0

	try:
		poptdisc, pcovdisc = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r[bindex:bhindex], rad[bindex:bhindex], sigma=stdbins[bindex:bhindex], bounds=(0,2), absolute_sigma=True)
		n_disc=round(poptdisc[0],2)
		isolated_discsim=SersicProfile(r, i_e, r_e, n_disc)
		res= np.abs(rad- isolated_discsim)
		n_disc_error=np.sqrt(sum((res[bindex:hindex]/stdbins[bindex:hindex])**2))
		n_disc_error=round(np.sqrt(n_disc_error),2)

		
		isolated_discsim=SersicProfile(r, i_e, r_e, n_disc)
		isolated_bulge= rad - isolated_discsim
		isolated_bulge[isolated_bulge<0]=0
		i_ebulge, r_ebulge, centralbrightnessbulge, totalbrightnessbulge= findeffectiveradius(isolated_bulge[0:bhindex], r[0:bhindex], nr[0:bhindex]) 
		poptbulge, pcovbulge = curve_fit(lambda x,n: SersicProfile(x, i_ebulge, r_ebulge, n), r[0:bindex], isolated_bulge[0:bindex],p0=4, sigma=stdbins[0:bindex], bounds=(0,10), absolute_sigma=True)
		n_bulge= round(poptbulge[0],1)
		res= np.abs(isolated_bulge- SersicProfile(r, i_ebulge, r_ebulge, n_bulge))
		n_bulge_error=np.sqrt(sum((res[2:bindex]/stdbins[2:bindex])**2))
		n_bulge_error=round(np.sqrt(n_bulge_error),2)
		
		
		isolated_bulgesim= SersicProfile(r, i_ebulge, r_ebulge, n_bulge)
		isolated_bulgesim[bindex:]=0
		totalsim=isolated_bulgesim+isolated_discsim
	except:
		isolated_discsim=isolated_bulge=isolated_bulgesim=totalsim=i_ebulge=r_ebulge=n_disc=n_bulge=0
		poptdisc=n_disc_error=n_bulge_error=poptbulge=0
	#plotchisquared(rad, r, i_e, r_e, stdbins, bindex, bhindex, hindex, i_ebulge,r_ebulge,isolated_bulge, nr)
	try:
		plotradialprofile(rad, r, i_e, r_e, stdbins, bindex, bhindex, hindex, pcov1, pcovdisc,pcovdisca,pcovbulge,pcovbulgea,isolated_discsim,isolated_bulge,isolated_bulgesim, isolated_discsima,isolated_bulgea,isolated_bulgesima, totalsim, totalsima, n_disc,n_disca,n_bulge,n_bulgea, n1, sim_name, imagefile, sigma_bulge,sigma_disc,radbintype)
	except:
		pass
	return n1, n1_error,  n_disc, n_disc_error,  n_bulge, n_bulge_error,  isolated_discsim, isolated_bulge, isolated_bulgesim, totalsim, i_ebulge, r_ebulge

def vary_sigma(image, imagefile, sim_name):
	#plots radius vs pixel intensity and its log
	std=np.std(image)
	BGRimage=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image2 = cv2.GaussianBlur(gray, ksize=(7,7), sigmaX=3,sigmaY=3)
	median=np.median(image2)
	sigma_max=(np.round((image2.max()-median)/std))-2
	print(sigma_max)
	maxVal, center = findcenter(image)
	rad, stdbins, r_arr, nr=radial_profile(image,center)
	max_r=np.sqrt(2)*15
	r=r_arr/256*30
	sigma_bulge_list=[]
	sigma_disc_list=[]
	sigma_disc_best_list=[]
	bulge_bestdisc_list=[]
	n_bestdisc_list=[]
	n_error_bestdisc_list=[]
	n_list=[]
	n_error_list=[]
	n_disc_list=[]
	n_disc_error_list=[]
	n_bulge_list=[]
	n_bulge_error_list=[]
	for i in np.linspace(1,sigma_max,25):
		sigma_bulge = i
		temp_disc_list=[]
		temp_n_error_list=[]
		temp_n_list=[]
		for j in np.arange(0.01, i-0.5, 0.4):
			sigma_disc = j
			print(sigma_bulge,sigma_disc)
			bindex,hindex, (hcX,hcY), (bcX,bcY) = findbulge(sim_name, BGRimage, imagefile, sigma_bulge, sigma_disc)
			print('bindex={},hindex={}'.format(bindex,hindex))
			#try:
			i_e, r_e, centralbrightness, totalbrightness= findeffectiveradius(rad[0:int(hindex)], r[0:int(hindex)], nr[0:int(hindex)]) 
			print('i_e={},r_e={}'.format(i_e,r_e))
			bhindex=int((bindex+hindex)/2)
			bindex=int(bindex)
			hindex=int(hindex)
			nr1=nr[1:hindex]
			r1=r[1:hindex]
			rad1=rad[1:hindex]
			stdbins1=stdbins[1:hindex]
			n_total, n_error,n_disc,n_bulge,n_disc_error,n_bulge_error=calculaten_total(rad1, r1, i_e, r_e, stdbins1, bindex, bhindex, hindex, nr1)
			print('ntotal={}, n_bulge={}'.format(n_total, n_bulge))
			#n1, pcov1, poptdisca, pcovdisca, poptbulgea, pcovbulgea, poptdisc, pcovdisc,  poptbulge, pcovbulge, isolated_discsima, isolated_bulgea, isolated_bulgesima, totalsima, isolated_discsim, isolated_bulge, isolated_bulgesim, totalsim, i_ebulge, r_ebulge, i_ebulgea, r_ebulgea = calculateSersicIndices(rad, r, i_e, r_e, stdbins, bindex, bhindex, hindex, nr, sim_name, imagefile, sigma_bulge,sigma_disc)
			sigma_bulge_list.append(sigma_bulge),sigma_disc_list.append(sigma_disc)
			n_list.append(n_total), n_error_list.append(n_error)
			n_disc_list.append(n_disc), n_disc_error_list.append(n_disc_error)
			n_bulge_list.append(n_bulge), n_bulge_error_list.append(n_bulge_error)
		
				#temp_disc_list.append(sigma_disc), temp_n_error_list.append(pcov1[0]), temp_n_list.append(n1[0])
			#except Exception as e:
				#print('2nd excep')
				#print(e)
		"""
		try:
			minerrorindex=np.argmin(temp_n_error_list)
			sigma_best=temp_disc_list[minerrorindex]
			n_best=temp_n_list[minerrorindex]
			sigma_disc_best_list.append(sigma_best)
			bulge_bestdisc_list.append(sigma_bulge), n_bestdisc_list.append(n_best), n_error_bestdisc_list.append(temp_n_error_list[minerrorindex])
			
		except:
			print('3rd excep')
		"""
	minerrorindex=np.argmin(n_error_list)
	sigma_bulge_total_best=sigma_bulge_list[minerrorindex]
	sigma_disc_total_best=sigma_disc_list[minerrorindex]

	minbulgeerrorindex=np.argmin(n_bulge_error_list)
	sigma_bulge_best=sigma_bulge_list[minerrorindex]
	mindiscerrorindex=np.argmin(n_disc_error_list)
	sigma_disc_best=sigma_disc_list[minerrorindex]
	print('sigma_bulge_total_best={}, sigma_disc_total_best={}'.format(sigma_bulge_total_best, sigma_disc_total_best))
	print('sigma_bulge_best={}, sigma_disc_best={}'.format(sigma_bulge_best, sigma_disc_best))

	fig0=plt.figure(figsize=(6,12))
	
	plt.subplot(411)
	plt.errorbar(sigma_disc_list, n_list, yerr=n_error_list, fmt='o',color='r', capsize=0.5, elinewidth=0.5, label='n', alpha=0.7, markersize=1)
	plt.xlabel('sigma disc'), plt.ylabel('n')
	plt.title('Varying Sigma For Radial Profile')
	plt.legend()
	
	plt.subplot(412)
	plt.errorbar(sigma_bulge_list, n_bulge_list, yerr=n_bulge_error_list, fmt='o',color='g', capsize=0.5, elinewidth=0.5, label='n', alpha=0.7, markersize=1)
	plt.xlabel('sigma bulge'), plt.ylabel('n bulge')
	plt.ylim(0,12)

	plt.subplot(413)
	plt.errorbar(sigma_disc_list, n_disc_list, yerr=n_disc_error_list, fmt='o',color='orange', capsize=0.5, elinewidth=0.5, label='n', alpha=0.7, markersize=1)
	plt.xlabel('sigma disc'), plt.ylabel('n disc')
	plt.ylim(0,12)
	"""
	plt.subplot(514)
	plt.plot(sigma_disc_list, n_list, 'ro', label='n', alpha=0.7, markersize=0.5)
	plt.plot(sigma_disc_list, n_disc_list, 'yo', label='disc', alpha=0.7, markersize=0.5)
	#plt.scatter(sigma_disc_best_list, n_bestdisc_list)
	plt.xlabel('sigma disc'), plt.ylabel('n')	
	plt.legend()
	"""
	plt.subplot(414)
	plt.scatter(sigma_bulge_list, sigma_disc_list, alpha=0.7)
	plt.xlabel('sigma bulge'),plt.ylabel('sigma disc')

	
	plt.text(0.5, -0.1, 'Best sigma bulge ={}, Best sigma disc={}'.format(np.round(sigma_bulge_best,2), np.round(sigma_disc_best,2)))
	plt.tight_layout()
	plt.savefig('galaxygraphsbin'+sim_name+'/VaryingSigma'+imagefile)
	
	plt.show()

def vary_radial_bins(image, imagefile, sim_name):
	#plots radius vs pixel intensity and its log
	maxVal, center = findcenter(image)
	bintypearray=['equalintenisty','equallogintensity', 'equalradius','equallogradius']
	for bintype in bintypearray:
		rad, stdbins, r_arr, nr=vary_radial_profile(image,center, bintype)
		r=r_arr*(30/256)
		sigma_bulge = 5
		sigma_disc = 1
		bindex,hindex, (hcX,hcY), (bcX,bcY) = findbulge(sim_name, image, imagefile, sigma_bulge, sigma_disc)
		print(bindex,hindex)
		hindex=(np.abs(r_arr-hindex)).argmin()
		bindex=(np.abs(r_arr-bindex)).argmin()
		print(bindex,hindex)
		i_e, r_e, centralbrightness, totalbrightness= findeffectiveradius(rad[0:int(hindex)], r[0:int(hindex)], nr[0:int(hindex)]) 
		nr=nr[1:int(hindex)]
		r=r[1:int(hindex)]
		rad=rad[1:int(hindex)]
		stdbins=stdbins[1:int(hindex)]
		bindex=int(bindex)
		hindex=int(hindex)
		bhindex=int((bindex+hindex)/2)
		n1, pcov1, poptdisca, pcovdisca, poptbulgea, pcovbulgea, poptdisc, pcovdisc,  poptbulge, pcovbulge, isolated_discsima, isolated_bulgea, isolated_bulgesima, totalsima, isolated_discsim, isolated_bulge, isolated_bulgesim, totalsim, i_ebulge, r_ebulge, i_ebulgea, r_ebulgea = calculateSersicIndices(rad, r, i_e, r_e, stdbins, bindex, bhindex, hindex, nr, sim_name, imagefile, sigma_bulge, sigma_disc, bintype)

def findassymetry(image):
	#find asymmetry based on 1 rotation
	image_arr = np.array(image)
	image_arr90 = np.rot90(image_arr)
	image_arr180 = np.rot90(image_arr90)
	resid= np.abs(image_arr-image_arr180)
	asymm=(np.sum(resid))/(np.sum(np.abs(image_arr)))
	return asymm

def binnedasymmetry(image):
	#find asymmetry based on 3 rotations
	image_arr = np.array(image)
	image_arr90 = np.rot90(image_arr)
	image_arr180 = np.rot90(image_arr90)
	image_arr270 = np.rot90(image_arr180)
	resid1= np.abs(image_arr-image_arr90)
	resid2= np.abs(image_arr-image_arr180)
	resid3= np.abs(image_arr-image_arr270)
	asymm1=(np.sum(resid1))/(np.sum(np.abs(image_arr)))
	asymm2=(np.sum(resid2))/(np.sum(np.abs(image_arr)))
	asymm3=(np.sum(resid3))/(np.sum(np.abs(image_arr)))
	meanasymm=np.mean([asymm1,asymm2,asymm3])
	asymmerror=np.std([asymm1,asymm2,asymm3])
	return meanasymm, asymmerror

def plotall2(df1):
	#plot all images to compare resolutions
	fig=plt.figure(figsize=(15,8))
	for i,image in enumerate(df1.imagefile):
		df=df1[df1.imagefile==image]
		plt.subplot(2,4, i+1)
		cv2image=plt.imread('ResolutionComparison/opencvbulgedisc'+image, 0)
		plt.imshow(cv2image)
		plt.axis('off')
		plt.title('n1={}$\pm${}, \n n_disc={}$\pm${}, n_bulge={}$\pm${}, \n asymm={}, \n bin_asymm={}$\pm${}'.format(df.n1.values[0], df.n1_error.values[0],  df.n_disc.values[0], df.n_disc_error.values[0],  df.n_bulge.values[0], df.n_bulge_error.values[0], df.asymm.values[0], df.meanasymm.values[0], df.asymmerror.values[0]))
	plt.tight_layout()
	plt.savefig('ResolutionComparison/Comparisonplot.png')
	plt.show()

def run_radial_profile(image, imagefile, sim_name):
	#vary_radial_bins(image, imagefile, sim_name)
	#vary_sigma(image, imagefile, sim_name)
	image2=np.average(image, axis=2, weights=[0.2126,0.587,0.114])
	asymm = findassymetry(image2)
	asymm=round(asymm,2)
	meanasymm, asymmerror=binnedasymmetry(image2)
	meanasymm=round(meanasymm,2)
	asymmerror=round(asymmerror,2)
	radbintype='equalradius'
	#plots radius vs pixel intensity and its log
	maxVal, center = findcenter(image)
	rad, stdbins, r_arr, nr=radial_profile(image,center)
	max_r=np.sqrt(2)*15
	r=r_arr/256*30
	sigma_bulge =6.5
	sigma_disc = 1.5
	bindex,hindex, (hcX,hcY), (bcX,bcY) = findbulge(sim_name, image, imagefile, sigma_bulge, sigma_disc)
	i_e, r_e, centralbrightness, totalbrightness= findeffectiveradius(rad[0:int(hindex)], r[0:int(hindex)], nr[0:int(hindex)]) 
	#r= np.linspace(0, max_r, num=maxr)
	nr=nr[1:int(hindex)]
	r=r[1:int(hindex)]
	rad=rad[1:int(hindex)]
	stdbins=stdbins[1:int(hindex)]
	#r= np.linspace(1/256, hindex/256, num=len(rad))
	b1=bindex*30/256
	h1=hindex*30/256
	bindex=int(bindex)
	hindex=int(hindex)
	bhindex=int((bindex+hindex)/2)
	n1, n1_error,  n_disc, n_disc_error,  n_bulge, n_bulge_error,  isolated_discsim, isolated_bulge, isolated_bulgesim, totalsim, i_ebulge, r_ebulge = calculateSersicIndices(rad, r, i_e, r_e, stdbins, bindex, bhindex, hindex, nr, sim_name, imagefile, sigma_bulge, sigma_disc, radbintype)
	#plotchisquared(rad, r, i_e, r_e, stdbins, bindex, bhindex, hindex, i_ebulge,r_ebulge,isolated_bulge, nr, sim_name)
	#plotradialprofile(rad, r, i_e, r_e, stdbins, bindex, bhindex, hindex, n1_error, n_disc_error,n_bulge_error,isolated_discsim,isolated_bulge,isolated_bulgesim, totalsim, n_disc,n_bulge, n1, sim_name, imagefile, sigma_bulge,sigma_disc,radbintype)
	return n1, n1_error,  n_disc, n_disc_error,  n_bulge, n_bulge_error, asymm, meanasymm, asymmerror, sigma_bulge, sigma_disc

if __name__ == "__main__":
	sim_name=['RecalL0025N0752','RefL0050N0752']
	imagefileRecalL0025N0752=['RecalL0025N0752galface_746518.png','RecalL0025N0752galface_770934.png','RecalL0025N0752galface_1642717.png','RecalL0025N0752galface_659536.png']
	imagefileRefL0050N0752=['RefL0050N0752galface_1881176.png','RefL0050N0752galface_126881.png','RefL0050N0752galface_1337479.png','RefL0050N0752galface_1343641.png']
	cols=['imagefile', 'n1', 'n1_error',  'n_disc', 'n_disc_error',  'n_bulge', 'n_bulge_error', 'asymm', 'meanasymm', 'asymmerror', 'col', 'row']
	list1=[]
	col=0
	for name in sim_name:
		print(name)
		
		if name=='RecalL0025N0752':
			row=0
			for imagefile in imagefileRecalL0025N0752:
				print(imagefile)
				image=plt.imread('galaxyimagebin'+name+'/'+imagefile, 0)
				n1, n1_error,  n_disc, n_disc_error,  n_bulge, n_bulge_error, asymm, meanasymm, asymmerror, sigma_bulge, sigma_disc=run_radial_profile(image, imagefile, name)
				col +=1
				list1.append([imagefile, n1, n1_error,  n_disc, n_disc_error,  n_bulge, n_bulge_error, asymm, meanasymm, asymmerror, col, row])
		elif name=='RefL0050N0752':
			row=1
			for imagefile in imagefileRefL0050N0752:
				print(imagefile)
				image=plt.imread('galaxyimagebin'+name+'/'+imagefile, 0)
				n1, n1_error,  n_disc, n_disc_error,  n_bulge, n_bulge_error, asymm, meanasymm, asymmerror, sigma_bulge, sigma_disc=run_radial_profile(image, imagefile, name)
				col +=1
				list1.append([imagefile, n1, n1_error,  n_disc, n_disc_error,  n_bulge, n_bulge_error, asymm, meanasymm, asymmerror, col, row])
	df1=pd.DataFrame(list1, columns=cols)
	print(df1)
	plotall2(df1)
	plt.tight_layout()
	plt.savefig('ResolutionComparison/RadialProfiletotal.png')
	plt.show()