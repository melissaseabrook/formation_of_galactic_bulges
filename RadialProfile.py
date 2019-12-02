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

def findeffectiveradius(radialprofile, r, nr):
	totalbrightness=np.sum(radialprofile * 2 * np.pi *nr*r)
	centralbrightness=radialprofile[0]
	cumulativebrightness=np.cumsum(radialprofile * 2 * np.pi *nr*r)
	r_e_unnormalised=((np.abs((totalbrightness/2) - cumulativebrightness)).argmin())
	r_e=r_e_unnormalised*(30.0/256)
	i_e= radialprofile[r_e_unnormalised]
	return i_e, r_e, centralbrightness, totalbrightness

def findlightintensity(radialpofile, radius):
	radius=int(radius)
	cumulativebrightness=np.sum(radialpofile[0:radius])
	return cumulativebrightness

def SersicProfile(r, I_e, R_e, n):
	b=(1.992*n)-0.3271
	G=(r/R_e)**(1/n)
	return I_e*np.exp((-b*(G-1)))

def SersicProfilea(r, I_e, R_e, n, a):
	b=(1.992*n)-0.3271
	G=(r/R_e)**(1/n)
	return I_e*np.exp((-b*(G-1)))+a

def twocomponentmodel(r, I_e, R_e, n):
	b=(1.992*n)-0.3271
	I1= I_e*np.exp((-b*((r/R_e)**(1/n)-1)))
	I2= I_e*np.exp((-1.68*((r/R_e)-1)))
	return I1+I2

def findbulge(image, imagefile):
	#locates central bulge and diffuse disc, and marks this on the image
	imagecopy=image.copy()
	median=np.median(image)
	std=np.std(image)
	print(median)
	print(std)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred1 = cv2.GaussianBlur(gray, ksize=(7,7), sigmaX=3,sigmaY=3)
	thresh1 = cv2.threshold(blurred1, median + 5*std, 255, cv2.THRESH_BINARY)[1]
	thresh1 = cv2.erode(thresh1, None, iterations=2)
	#thresh1 = cv2.dilate(thresh1, None, iterations=4)
	blurred2 = cv2.GaussianBlur(gray, ksize=(15, 15), sigmaX=3,sigmaY=3)
	thresh2 = cv2.threshold(blurred2, median + std, 255, cv2.THRESH_BINARY)[1]
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
	else:
		((bcX, bcY), bradius) = ((0,0),0)
	print("bulge radius:{},  bulge centre({},{})".format(bradius, bcX,bcY))
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
	else:
		((hcX, hcY), hradius) = ((0,0),0)
	print("disc radius:{}, disc centre({},{})".format(hradius, hcX,hcY))
	return bradius,hradius, (hcX,hcY), (bcX,bcY)

def run_radial_profile(image, imagefile, sim_name):
	#plots radius vs pixel intensity and its log
	width, height =image.shape[:2]
	cx=int(width/2)
	cy=int(height/2)
	maxVal, center = findcenter(image)
	rad, stdbins, r_arr, nr=radial_profile(image,center)
	bindex,hindex, (hcX,hcY), (bcX,bcY) = findbulge(image, imagefile)
	max_r=np.sqrt(2)*15
	r=r_arr/256*30
	#r= np.linspace(0, max_r, num=maxr)
	nr=nr[1:int(hindex)]
	r=r[1:int(hindex)]
	rad=rad[1:int(hindex)]
	#r= np.linspace(1/256, hindex/256, num=len(rad))
	
	stdbins=stdbins[1:int(hindex)]
	i_e, r_e, centralbrightness, totalbrightness= findeffectiveradius(rad, r, nr) 
	bindex,hindex, (hcX,hcY), (bcX,bcY) = findbulge(image, imagefile)
	
	
	#stdbins[0:2]=stdbins[0:2]*2
	b1=bindex*30/256
	h1=hindex*30/256
	bindex=int(bindex)
	hindex=int(hindex)
	bhindex=int((bindex+hindex)/2)

	
	
	poptdisca, pcovdisca=curve_fit(SersicProfilea, r[bhindex:hindex], rad[bhindex:hindex], p0=(i_e, r_e, 1,0), bounds=((i_e-1,r_e-0.1,0,0), (i_e+1,r_e+0.1,2,20)), sigma=stdbins[bhindex:hindex], absolute_sigma=True)
	n_disca=poptdisca[2]
	print("I_edisca={}, R_edisca={}, n_disca={}, adisca={}".format(poptdisca[0], poptdisca[1], n_disca, poptdisca[3]))
	isolated_discsima=SersicProfilea(r, poptdisca[0], poptdisca[1],n_disca,poptdisca[3])
	isolated_bulgea= rad - isolated_discsima
	isolated_bulgea[isolated_bulgea<0]=0
	i_ebulgea, r_ebulgea, centralbrightnessbulgea, totalbrightnessbulgea= findeffectiveradius(isolated_bulgea[0:bhindex], r[0:bhindex], nr[0:bhindex]) 
	poptbulgea, pcovbulgea=curve_fit(SersicProfilea, r[0:bindex], isolated_bulgea[0:bindex], p0=(i_ebulgea, r_ebulgea, 4,0), bounds=((i_ebulgea-1,r_ebulgea-0.1,0,0), (i_ebulgea+1,r_ebulgea+0.1,8,20)), sigma=stdbins[0:bindex], absolute_sigma=True)
	n_bulgea= poptbulgea[2]
	isolated_bulgesima= SersicProfilea(r, poptbulgea[0], poptbulgea[1], n_bulgea, poptbulgea[3])
	isolated_bulgesima[bindex:]=0
	totalsima=isolated_bulgesima+isolated_discsima
	print("I_ebulgea={}, R_ebulgea={}, n_bulgea={}, abulgea={}".format(poptbulgea[0],poptbulgea[1], n_bulgea, poptbulgea[2]))




	poptdisc, pcovdisc = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r[bindex:bhindex], rad[bindex:bhindex], sigma=stdbins[bindex:bhindex], bounds=(0,2), absolute_sigma=True)
	n_disc=poptdisc[0]
	n_disc_error=pcovdisc[0,0]
	print("I_edisc={}, R_edisc={}, n_disc={}".format(i_e, r_e, n_disc))
	isolated_discsim=SersicProfile(r, i_e, r_e, n_disc)
	isolated_bulge= rad - isolated_discsim
	isolated_bulge[isolated_bulge<0]=0
	i_ebulge, r_ebulge, centralbrightnessbulge, totalbrightnessbulge= findeffectiveradius(isolated_bulge[0:bhindex], r[0:bhindex], nr[0:bhindex]) 
	poptbulge, pcovbulge = curve_fit(lambda x,n: SersicProfile(x, i_ebulge, r_ebulge, n), r[0:bindex], isolated_bulge[0:bindex],p0=4, sigma=stdbins[0:bindex], bounds=(0,10), absolute_sigma=True)
	n_bulge= poptbulge[0]
	n_bulge_error= pcovbulge[0,0]
	print("I_ebulge={}, R_ebulge={}, n_bulge={}".format(i_ebulge,r_ebulge, n_bulge))
	isolated_bulgesim= SersicProfile(r, i_ebulge, r_ebulge, n_bulge)
	isolated_bulgesim[bindex:]=0
	totalsim=isolated_bulgesim+isolated_discsim


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
		res= rad - SersicProfile(r, i_e,r_e,n)
		n1_error=np.sqrt(sum((res[bindex:bhindex]/stdbins[bindex:bhindex])**2))
		#n1_error=pcov1[0,0]
		
		poptdisc, pcovdisc = curve_fit(lambda x,ni: SersicProfile(x, i_e, r_e, ni), r[bhindex:hindex], rad[bhindex:hindex], sigma=stdbins[bhindex:hindex], p0=n, bounds=(n-0.001,n+0.001), absolute_sigma=True)
		n_disc=poptdisc[0]
		#n_disc_error=pcovdisc[0,0]	
		isolated_discsim=SersicProfile(r, i_e, r_e, n)
		res= rad- isolated_discsim
		n_disc_error=np.sqrt(sum((res[bhindex:hindex]/stdbins[bhindex:hindex])**2))
		poptbulge, pcovbulge = curve_fit(lambda x,ni: SersicProfile(x, i_ebulge, r_ebulge, ni), r[0:bindex], isolated_bulge[0:bindex], sigma=stdbins[0:bindex], p0=n, bounds=(n-0.001,n+0.001), absolute_sigma=True)
		n_bulge= poptbulge[0]
		#n_bulge_error= pcovbulge[0,0]
		res= rad - SersicProfile(r, i_ebulge, r_ebulge, n_bulge)
		n_bulge_error=np.sqrt(sum((res[0:int(bindex/2)]/stdbins[0:int(bindex/2)])**2))
		
		poptdisca, pcovdisca=curve_fit(SersicProfilea, r[bindex:hindex], rad[bindex:hindex], p0=(i_e, r_e, n,0), bounds=((i_e-1,r_e-0.1,n-0.001,0), (i_e+1,r_e+0.1,n+0.001,20)), sigma=stdbins[bindex:hindex], absolute_sigma=True)
		n_disca=poptdisca[2]
		#n_disc_errora=pcovdisca[2,2]
		isolated_discsima=SersicProfilea(r, poptdisca[0], poptdisca[1],n_disca,poptdisca[3])
		res= rad - isolated_discsim
		n_disc_errora=np.sqrt(sum((res[bhindex:hindex]/stdbins[bhindex:hindex])**2))

		isolated_bulgea= rad - isolated_discsima
		isolated_bulgea[isolated_bulgea<0]=0
		i_ebulgea, r_ebulgea, centralbrightnessbulgea, totalbrightnessbulgea= findeffectiveradius(isolated_bulgea[0:bhindex], r[0:bhindex], nr[0:bhindex]) 
		poptbulgea, pcovbulgea=curve_fit(SersicProfilea, r[0:bindex], isolated_bulgea[0:bindex], p0=(i_ebulgea, r_ebulgea, n,0), bounds=((i_ebulgea-1,r_ebulgea-0.1,n-0.001,0), (i_ebulgea+1,r_ebulgea+0.1,n+0.001,20)), sigma=stdbins[0:bindex], absolute_sigma=True)
		n_bulgea= poptbulgea[2]
		#n_bulge_errora=pcovdisca[2,2]
		res= rad - SersicProfilea(r, i_ebulgea, r_ebulgea, n_bulgea, poptbulgea[3])
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
	plt.savefig('galaxygraphsbin'+sim_name+'/chisqrdvsn'+imagefile)
	plt.show()
	exit()


	fig=plt.figure(figsize=(15,8))

	plt.subplot(321)
	n1, pcov1 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r, rad, p0=2, bounds=(0.1,8), sigma=stdbins, absolute_sigma=True)
	print("I_e={}, R_e={}, n_disc={}".format(i_e, r_e, n1))
	n_total=n1[0]
	plt.errorbar(r, rad, yerr=(stdbins*2), fmt='', color='k', capsize=0.5, elinewidth=0.5)
	plt.plot(r, SersicProfile(r,i_e, r_e, n_total +pcov1[0,0]), 'g--')
	plt.plot(r, SersicProfile(r,i_e, r_e, n_total), 'g', label='n={}'.format(round(n_total,2)))
	plt.plot(r, SersicProfile(r,i_e, r_e, n_total -pcov1[0,0]) , 'g--')
	plt.fill_between(r, SersicProfile(r,i_e, r_e, n_total -pcov1[0,0]**0.5), SersicProfile(r,i_e, r_e, n_total +pcov1[0,0]**0.5), facecolor='gray', alpha=0.5)
	plt.title('Total Profile Fit'), plt.xlabel('Radius (kpc)'), plt.ylabel('Intensity'), plt.xlim(0), plt.ylim(0,250)
	plt.legend(bbox_to_anchor=(1.05,1), loc=2)

	plt.subplot(323)
	plt.errorbar(r, rad, yerr=(stdbins), fmt='', color='k', capsize=0.5, elinewidth=0.5, label='observed')
	plt.errorbar(r, isolated_discsim, yerr=(np.sqrt(pcovdisc[0,0])), fmt='', color='r', capsize=0.5, elinewidth=0.5, label='disc sim, n_disc={}'.format(round(n_disc,2)))
	plt.errorbar(r, isolated_bulge, yerr=(stdbins), fmt='', color='g', capsize=0.5, elinewidth=0.5, label='bulge observed')
	plt.errorbar(r[0:bindex], isolated_bulgesim[0:bindex], yerr=(np.sqrt(pcovbulge[0,0])), fmt='', color='b', capsize=0.5, elinewidth=0.5, label='bulge sim, n_bulge={}'.format(round(n_bulge,2)))
	plt.plot(r[0:hindex],totalsim, color='y', label='total sim')
	plt.legend(bbox_to_anchor=(1.05,1), loc=2)
	plt.title('Bulge Extracted Fit'), plt.xlabel('Radius (kpc)'), plt.ylabel('Intensity'), plt.xlim(0), plt.ylim(0,250)

	plt.subplot(325)
	plt.errorbar(r, rad, yerr=(stdbins), fmt='', color='k', capsize=0.5, elinewidth=0.5, label='observed')
	plt.errorbar(r, isolated_discsima, yerr=(np.sqrt(pcovdisca[2,2])), fmt='', color='r', capsize=0.5, elinewidth=0.5, label='disc sim, n_disca={}'.format(round(n_disca,2)))
	plt.errorbar(r, isolated_bulgea, yerr=(stdbins), fmt='', color='g', capsize=0.5, elinewidth=0.5, label='bulge observed')
	plt.errorbar(r[0:bindex], isolated_bulgesima[0:bindex], yerr=(np.sqrt(pcovbulgea[2,2])), fmt='', color='b', capsize=0.5, elinewidth=0.5, label='bulge sim, n_bulgea={}'.format(round(n_bulgea,2)))
	plt.plot(r[0:hindex],totalsima, color='y', label='total sim')
	plt.legend(bbox_to_anchor=(1.05,1), loc=2)
	plt.title('Bulge Extracted Fit with Background Translation'), plt.xlabel('Radius (kpc)'), plt.ylabel('Intensity'), plt.xlim(0), plt.ylim(0,250)

	

	plt.subplot(122)
	cv2image=plt.imread('galaxygraphsbin'+sim_name+'/BulgeDiscImages/opencvfindbulge'+imagefile, 0)
	plt.imshow(cv2image)
	plt.title('image')
	plt.tight_layout()
	plt.savefig('galaxygraphsbin'+sim_name+'/Sersicfitradialbrightnessprofile'+imagefile)
	plt.show()
	#plt.close()

	"""
	print("I_e={}, R_e={}".format(i_e, r_e))
	print("brad={}, hrad={}, bind={}, hind={}".format(b1,h1,bindex,hindex))
	popt1, pcov1=curve_fit(SersicProfile, r, rad, p0=(i_e, r_e, 1), bounds=((0,0,0.001), (np.inf,np.inf,10)), sigma=stdbins*6, absolute_sigma=True)
	popt2, pcov2 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r, rad, p0=3, sigma=stdbins*2, absolute_sigma=True)
	popt3, pcov3 = curve_fit(lambda x,n: twocomponentmodel(x, i_e, r_e, n), r, rad, sigma=stdbins*2, absolute_sigma=True)
	print("I_e,R_e,n={}".format(popt1))
	print("I_e={}, R_e={}, n={}".format(i_e, r_e, popt2))
	print("I_e={}, R_e={}, n={}".format(i_e, r_e, popt3))

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
	plt.savefig('galaxygraphsbinRecal/radialbrightnessprofile'+imagefile)
	plt.show()
	
	
	#2 component fitting
	popt21, pcov21 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r[0:bindex], rad[0:bindex],sigma=stdbins[0:bindex]*2, absolute_sigma=True)
	popt22, pcov22 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r[bindex:hindex], rad[bindex:hindex], sigma=stdbins[bindex:hindex]*2, absolute_sigma=True)
	print("n1={}".format(popt21))
	print("n2={}".format(popt22))

	plt.subplot(211)
	plt.errorbar(r[0:hindex], rad[0:hindex], yerr=(stdbins[0:hindex]*2), fmt='', color='k', capsize=0.5, elinewidth=0.5)
	plt.plot(r[0:bindex], SersicProfile(r[0:bindex],i_e, r_e,popt21), 'r-', label='bulge n={}'.format(round(popt21[0],2)))
	#plt.plot(r, SersicProfile(r,i_e, r_e, popt2 +pcov2[0,0]), 'g--')
	plt.plot(r[bindex:hindex], SersicProfile(r[bindex:hindex],i_e, r_e, popt22), 'g', label='disc n={} free'.format(round(popt22[0],2)))
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
	plt.savefig('galaxygraphsbinRecal/2componentradialbrightnessprofile'+imagefile)
	plt.show()
	plt.close()
	
	
	plt.plot(r, rad)
	plt.title('Log'), plt.xlabel('log(Radius) (log(kpc))'), plt.ylabel('log(Intensity)')
	plt.xscale('log')
	plt.yscale('log')
	plt.tight_layout()
	plt.show()
	"""

if __name__ == "__main__":
	sim_name=['RecalL0025N0752','']
	imagefileRecalL0025N0752=['RecalL0025N0752galface_726306.png','']
	#sim_name=['RecalL0025N0752', 'RefL0025N0376','RefL0050N0752']
	#imagefileRecalL0025N0752=['RecalL0025N0752galface_646493.png','RecalL0025N0752galface_737885.png','RecalL0025N0752galface_746518.png','RecalL0025N0752galface_853401.png','RecalL0025N0752galface_4938.png','RecalL0025N0752galface_621500.png','RecalL0025N0752galface_726306.png','RecalL0025N0752galface_51604.png']
	imagefileRefL0025N0376=['RefL0025N0376galface_1.png','RefL0025N0376galface_135107.png','RefL0025N0376galface_154514.png','RefL0025N0376galface_160979.png','RefL0025N0376galface_172979.png']
	imagefileRefL0050N0752=['RefL0050N0752galface_2273534.png','RefL0050N0752galface_2276263.png','RefL0050N0752galface_514258.png','RefL0050N0752galface_2355640.png','RefL0050N0752galface_2639531.png']

	for name in sim_name:
		print(name)
		if name=='RecalL0025N0752':
			for imagefile in imagefileRecalL0025N0752:
				print(imagefile)
				image=plt.imread('galaxyimagebin'+name+'/'+imagefile, 0)
				run_radial_profile(image, imagefile, name)
		elif name=='RefL0025N0376':
			for imagefile in imagefileRefL0025N0376:
				print(imagefile)
				image=plt.imread('galaxyimagebin'+name+'/'+imagefile, 0)
				run_radial_profile(image, imagefile, name)
		elif name=='RefL0050N0752':
			for imagefile in imagefileRefL0050N0752:
				print(imagefile)
				image=plt.imread('galaxyimagebin'+name+'/'+imagefile, 0)
				run_radial_profile(image, imagefile, name)
