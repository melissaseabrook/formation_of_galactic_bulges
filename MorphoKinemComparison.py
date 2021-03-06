"""Investigate the z=0 population of galaxies"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from imutils import contours
from skimage import measure
import imutils
import math
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as mcol
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata, UnivariateSpline
import statmorph
import photutils
import scipy.ndimage as ndi
from astropy.modeling import models, fitting

def logx(x):
    #base 10 log
    if x !=0:
        return np.log10(x)
    else:
        return 0

def divide(x,y):
    #divide x by y
    if y !=0:
        return x/y
    else:
        return 0

def threshtonan(x, thresh):
    #convert values below theshold to nan
    if x<thresh:
        return np.nan
    else:
        return x

def threshtonanupper(x, thresh):
    #convert values above theshold to nan
    if x>thresh:
        return np.nan
    else:
        return x

def extract_val(x):
    #extract middle of string
    try:
        return x[1:7]
    except:
        return np.nan

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
    #find light intensity at given radius
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
    total_brightness=np.sum(radialprofile)
    return cumulativebrightness, total_brightness

def findcenter(image):
    #finds coords of central bulge
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11,11), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    return maxVal, maxLoc

def findandlabelbulge(image, imagefile, sim_name):
    #locates central bulge and diffuse disc, and marks this on the image
    print(imagefile)
    median=np.median(image)
    std=np.std(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred1 = cv2.GaussianBlur(gray, ksize=(7, 7), sigmaX=3,sigmaY=3)

    if (std>20):
        thresh1 = cv2.threshold(blurred1,  (3.5*std)+median, 255, cv2.THRESH_BINARY)[1]
        
    else:
        thresh1 = cv2.threshold(blurred1,  (6.5*std) +median, 255, cv2.THRESH_BINARY)[1]
    thresh1 = cv2.erode(thresh1, None, iterations=2)
    thresh1 = cv2.dilate(thresh1, None, iterations=4)

    blurred2 = cv2.GaussianBlur(gray, ksize=(15, 15), sigmaX=3,sigmaY=3)
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
        print("bulge radius:{},  bulge centre({},{})".format(bradius, bcX,bcY))
    else:
        bradius, bcX,bcY=0,0,0
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
        print("disc radius:{}, disc centre({},{})".format(hradius, hcX,hcY))
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
                count+=1
    else:
        count=0

    btdradius=bradius/hradius
    print("disc radius:bulge radius ={}".format(btdradius))
    print("star count ={}".format(count))
    disc_intensity, total_intensity=findlightintensity(image, hradius, (hcX,hcY))
    bulge_intensity, total_intensity=findlightintensity(image, bradius, (bcX,bcY))
    btdintensity= bulge_intensity/disc_intensity
    btotalintensity=bulge_intensity/total_intensity
    btotalradius=bradius/256
    cv2.destroyAllWindows()
    print("disc intensity = {}, bulge intensity ={}, disc:bulge intensity ={}".format(disc_intensity, bulge_intensity, btdintensity))
    return btdradius, btdintensity, count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius

def invertbtd(r):
    #invert btd
    if r !=0:
        return 1/r
    else:
        return 0

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
	stdbins[0]=stdbins[1]
	stdbins[stdbins<0.1]=0.1
	radialprofile, r_arr, binnumber=stats.binned_statistic(r.ravel(),image.ravel(), 'mean', bins=len(nr))
	#meanbins, bin_edges, binnumber=stats.binned_statistic(r.ravel(),image.ravel(), 'mean', bins=len(radialprofile))
	return radialprofile, r_arr, stdbins, nr

def findeffectiveradius(radialprofile, r, nr):
    #find radius at which half the total light is contained
	totalbrightness=np.sum(radialprofile * 2 * np.pi *r*nr)
	centralbrightness=radialprofile[0]
	cumulativebrightness=np.cumsum(radialprofile * 2 * np.pi *r*nr)
	r_e_index=((np.abs((totalbrightness/2) - cumulativebrightness)).argmin())
	r_e=r[r_e_index]
	#r_e=r_e_unnormalised*(30.0/256)
	i_e= radialprofile[r_e_index]
	return i_e, r_e, centralbrightness, totalbrightness

def findeffectiveradiusfrac(radialprofile, r, nr, frac):
    #find radius at which fraction of total light is contained
    print(len(radialprofile), len(r),len(nr))
    totalbrightness=np.sum(radialprofile * 2 * np.pi *nr*r)
    centralbrightness=radialprofile[0]
    cumulativebrightness=np.cumsum(radialprofile * 2 * np.pi *nr*r)
    r_e_index=((np.abs((totalbrightness*frac) - cumulativebrightness)).argmin())
    r_e=r[r_e_index]
    i_e= radialprofile[r_e_index]
    return i_e, r_e, centralbrightness, totalbrightness

def SersicProfile(r, I_e, R_e, n):
    #sersic profile function
	b=np.exp(0.6950 + np.log(n) - (0.1789/n))
	G=(r/R_e)**(1/n)
	return I_e*np.exp((-b*(G-1)))

def SersicProfilea(r, I_e, R_e, n, a):
    #sersic profile with a background translation
	b=(2*n)-(1/3)
	G=(r/R_e)**(1/n)
	return I_e*np.exp((-b*(G-1)))+a

def findconcentration(rad, r, nr):
    #find concentration
    i20, r80, cb,tb=findeffectiveradiusfrac(rad, r, nr, 0.8)
    i20, r20, cb, tb=findeffectiveradiusfrac(rad, r, nr, 0.2)
    con=5*np.log10(r80/r20)
    return con, r80,r20

def findassymetry(image):
    #find asymmetry
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
    asymm=np.mean([asymm1,asymm2,asymm3])
    asymmerror=np.std([asymm1,asymm2,asymm3])
    return asymm, asymmerror

def twoDsersicfit(image, i_e, r_e, guess_n, center):
    #fit a 2d sersic index
    try:
        blur = cv2.GaussianBlur(image, ksize=(11,11), sigmaX=3,sigmaY=3)
        x0,y0=center
        z=blur.copy()
        ny, nx = blur.shape
        y, x = np.mgrid[0:ny, 0:nx]
        sersicinit=models.Sersic2D(amplitude = i_e, r_eff = r_e, n=guess_n, x_0=x0, y_0= y0)
        fit_sersic = fitting.LevMarLSQFitter()
        sersic_model = fit_sersic(sersicinit, x, y, z, maxiter=500, acc=1e-5)
        nd=sersic_model.n.value
        sim=sersic_model(x,y)
        #=ma.log10(image)
        #logimg=logimg.filled(0)
        #logsim=ma.log10(sim)
        #logsim=logsim.filled(0)
        #res = logx(np.sum(np.abs(logimg - logsim)))
        #nd_error=np.sqrt(res)
        res = (np.abs(blur - sim))
        n2d_error=np.sqrt(np.sum(res)/nx*ny)
    except:
        nd=np.nan
        n2d_error=np.nan
    return nd, n2d_error

def findsersicindex(image, bindex, dindex):
    #produce different sersic indices
    image2=np.average(image, axis=2, weights=[0.2126,0.587,0.114])
    asymm, asymmerror=findassymetry(image2)

    try:
        maxVal, center = findcenter(image)
        rad, r_arr,stdbins, nr=radial_profile(image,center)
        max_r=np.sqrt(2)*15
        r= np.linspace(0, max_r, num=len(rad))
        i_e, r_e, centralbrightness, totalbrightness= findeffectiveradius(rad, r, nr) 
        r=r_arr/256*30
        con, r80, r20=findconcentration(rad, r_arr[1:], nr)

        nr=nr[1:int(dindex)]
        r=r[1:int(dindex)]
        rad=rad[1:int(dindex)]
        stdbins=stdbins[1:int(dindex)]
        stdbins[0:2]=stdbins[0:2]*2
        b1=bindex*30/256
        h1=dindex*30/256
        bindex=int(bindex)
        dindex=int(dindex)
        bdindex=int((bindex+dindex)/2)

        
        
        n1, pcov1 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r, rad, p0=2, bounds=(0.1,10), sigma=stdbins, absolute_sigma=True)
        print("I_e={}, R_e={}, n_disc={}".format(i_e, r_e, n1))
        n_total=n1[0]
        res= np.abs(rad - SersicProfile(r, i_e,r_e,n_total))
        n_total_error=np.sqrt(sum((res[bindex:bdindex]/stdbins[bindex:bdindex])**2))
        n_total_error1=np.sqrt(pcov1[0])

        try:
            n2d, n2d_error=twoDsersicfit(image2, i_e, r_e, n_total, center)
        except:
            n2d=np.nan
            n2d_error=np.nan

        poptdisca, pcovdisca=curve_fit(SersicProfilea, r[bindex:dindex], rad[bindex:dindex], p0=(i_e, r_e, 1,0), bounds=((i_e-0.5,r_e-0.1,0.1,0), (i_e+0.5,r_e+0.1,2,20)), sigma=stdbins[bindex:dindex], absolute_sigma=True)
        n_disca=poptdisca[2]
        res= np.abs(rad - SersicProfile(r, i_e,r_e,n_disca))
        n_disca_error=np.sqrt(sum((res[bindex:bdindex]/stdbins[bindex:bdindex])**2))

        print("I_edisca={}, R_edisca={}, n_disca={}, adisca={}".format(poptdisca[0], poptdisca[1], n_disca, poptdisca[3]))
        isolated_discsima=SersicProfilea(r, poptdisca[0], poptdisca[1], n_disca,poptdisca[3])
        isolated_bulgea= rad - isolated_discsima
        isolated_bulgea[isolated_bulgea<0]=0
        i_ebulgea, r_ebulgea, centralbrightnessbulgea, totalbrightnessbulgea= findeffectiveradius(isolated_bulgea[0:bdindex], r[0:bdindex], nr[0:bdindex]) 
        poptbulgea, pcovbulgea=curve_fit(SersicProfilea, r[0:bindex], isolated_bulgea[0:bindex], p0=(i_ebulgea, r_ebulgea, 4,0), bounds=((i_ebulgea-1,r_ebulgea-0.1,0.01,0), (i_ebulgea+1,r_ebulgea+0.1,10,20)), sigma=stdbins[0:bindex], absolute_sigma=True)
        n_bulgea= poptbulgea[2]
        res= np.abs(rad - isolated_discsima - SersicProfile(r, i_e,r_e,n_bulgea))
        n_bulgea_error=np.sqrt(sum((res[bindex:bdindex]/stdbins[bindex:bdindex])**2))
        print("I_ebulgea={}, R_ebulgea={}, n_bulgea={}, abulgea={}".format(poptbulgea[0],poptbulgea[1], n_bulgea, poptbulgea[2]))

        poptdisc, pcovdisc = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r[bindex:dindex], rad[bindex:dindex], sigma=stdbins[bindex:dindex], bounds=(0.0001,10), absolute_sigma=True)
        n_disc=poptdisc[0]
        res= np.abs(rad - SersicProfile(r, i_e,r_e,n_disc))
        n_disc_error=np.sqrt(sum((res[bindex:bdindex]/stdbins[bindex:bdindex])**2))

        print("I_edisc={}, R_edisc={}, n_disc={}".format(i_e, r_e, n_disc))
        isolated_discsim=SersicProfile(r, i_e, r_e, n_disc)
        isolated_bulge= rad - isolated_discsim
        isolated_bulge[isolated_bulge<0]=0
        i_ebulge, r_ebulge, centralbrightnessbulge, totalbrightnessbulge= findeffectiveradius(isolated_bulge[0:bdindex], r[0:bdindex], nr[0:bdindex]) 
        poptbulge, pcovbulge = curve_fit(lambda x,n: SersicProfile(x, i_ebulge, r_ebulge, n), r[0:bindex], isolated_bulge[0:bindex],p0=4, sigma=stdbins[0:bindex], bounds=(0,10), absolute_sigma=True)
        n_bulge= poptbulge[0]
        res= np.abs(rad - isolated_discsim - SersicProfile(r, i_e,r_e,n_bulge))
        n_bulge_error=np.sqrt(sum((res[bindex:bdindex]/stdbins[bindex:bdindex])**2))
        
        print("I_ebulge={}, R_ebulge={}, n_bulge={}".format(i_ebulge,r_ebulge, n_bulge))


        exponential_discsim=SersicProfile(r, i_e, r_e, 1)
        isolated_bulge2= rad - exponential_discsim
        isolated_bulge2[isolated_bulge2<0]=0
        i_ebulge2, r_ebulge2, centralbrightnessbulge2, totalbrightnessbulge2= findeffectiveradius(isolated_bulge2[0:bdindex], r[0:bdindex], nr[0:bdindex]) 
        poptbulge2, pcovbulge2 = curve_fit(lambda x,n: SersicProfile(x, i_ebulge2, r_ebulge2, n), r[0:bindex], isolated_bulge[0:bindex],p0=4, bounds=(0,10), sigma=stdbins[0:bindex], absolute_sigma=True)
        n_bulge_exp= poptbulge[0]
        res= np.abs(rad - exponential_discsim - SersicProfile(r, i_e,r_e,n_bulge_exp))
        n_bulge_exp_error=np.sqrt(sum((res[bindex:bdindex]/stdbins[bindex:bdindex])**2))

        print("n_bulge={}".format(n_bulge_exp))
    except:
        n_total=np.nan
        n_bulge=np.nan
        n_disc=np.nan
        n_bulgea=np.nan
        n_disca=np.nan
        n_bulge_exp=np.nan
        n_total_error=np.nan
        n_total_error1=np.nan
        n_bulge_error=np.nan
        n_disc_error=np.nan
        n_bulgea_error=np.nan
        n_disca_error=np.nan
        n_bulge_exp_error=np.nan
        con=np.nan
        r80=np.nan
        r20=np.nan
        try:
            n2d, n2d_error=twoDsersicfit(image2, 100, 7, 1.5, (128,128))
        except:
            n2d=np.nan
            n2d_error=np.nan
    return n_total, n2d,n2d_error, n_disca, n_bulgea, n_disc, n_bulge, n_bulge_exp, n_total_error,n_total_error1, n_disca_error, n_bulgea_error, n_disc_error, n_bulge_error, n_bulge_exp_error, con, r80, r20, asymm, asymmerror

def runstatmorph(image):
    #run statmorph
    image2=np.average(image, axis=2, weights=[0.2126,0.587,0.114])
    gain = 1000.0
    threshold = photutils.detect_threshold(image2, 1.5)
    npixels = 7  # minimum number of connected pixels
    segm = photutils.detect_sources(image2, threshold, npixels)
    # Keep only the largest segment
    label = np.argmax(segm.areas) + 1
    segmap = segm.data == label
    segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
    segmap = segmap_float > 0.5
    source_morphs = statmorph.source_morphology(image2, segmap, gain=gain)
    morph = source_morphs[0]
    if morph.flag ==1:
        morph_c=morph.concentration
        morph_asymm=morph.asymmetry
        morph_smoothness=morph.smoothness
        morph_sersic_rhalf=morph.sersic_rhalf*30/256
        morph_xc_asymmetry=morph.xc_asymmetry
        morph_yc_asymmetry=morph.yc_asymmetry
        if morph.flag_sersic==0:
            morph_sersic_n=morph.sersic_n
        else:
            morph_sersic_n=np.nan
    else:
        morph_c=morph_asymm=morph_sersic_n=morph_smoothness=morph_sersic_rhalf=morph_xc_asymmetry=morph_yc_asymmetry=np.nan
    return morph_c, morph_asymm, morph_sersic_n, morph_smoothness, morph_sersic_rhalf, morph_xc_asymmetry, morph_yc_asymmetry

def drop_numerical_outliers(df, z_thresh):
    #remove outliers
    constrains=df.select_dtypes(include=[np.number]).apply(lambda x: np.abs(stats.zscore(x)) <z_thresh).all(axis=1)
    df.drop(df.index[~constrains], inplace=True)

def removeoutlierscolumn(df, column_name, sigma):
    #remove outliers from specific column
    df=df[np.abs(df[column_name]-df[column_name].mean())<=(sigma*df[column_name].std())]
    return df

def invert(var):
    #invert varaible
    if var != 0:
        return(1/var)*10
    else:
        return 0

def categorise(asymm, param, thresh):
    #categorise based on asymmetry and morphological parameter
    if asymm > 0.35:
        return 'asymmetric'
    elif param > thresh+0.1:
        return 'bulge'
    elif param < thresh-0.1:
        return 'disc'
    else:
        return 'border'

def colourise(asymm, param, thresh):
    #colourise parameters based on asymetry and threshold.
    if asymm > 0.35:
        return 'y'
    elif param > thresh+0.1:
        return 'r'
    elif param < thresh-0.1:
        return 'b'
    else:
        return 'purple'

def cleanandtransformdata(df):
    #clean and manipulate data
    print(df.shape)
    #drop_numerical_outliers(df, 3)
    #df=df=df.reset_index()
    #print(df.shape)
    
    df['BulgeToTotal']=df.apply(lambda x: (1-x.DiscToTotal), axis=1)
    df['logBHmass']=df.apply(lambda x: logx(x.BHmass), axis=1)
    df['logmass']=df.apply(lambda x: logx(x.Starmass), axis=1)
    df['sSFR']=df.apply(lambda x: divide(x.SFR*1e12,x.Starmass), axis=1)
    df['logsSFR']=df.apply(lambda x: logx(x.sSFR), axis=1)
    df['sDMmass']=df.apply(lambda x: divide(x.Starmass,x.DMmass), axis=1)
    df['logsDMmass']=df.apply(lambda x: logx(x.sDMmass), axis=1)
    df['logDMmass']=df.apply(lambda x: logx(x.DMmass), axis=1)
    df['logSFR']=df.apply(lambda x: logx(x.SFR), axis=1)
    df['dtototal']=df.apply(lambda x: (1-x.btdintensity), axis=1)
    df['dtbradius']=df.apply(lambda x: invertbtd(x.btdradius), axis=1)
    df['dtbintensity']=df.apply(lambda x: invertbtd(x.btdintensity), axis=1)
    #df['morph_asymm']=df.apply(lambda x: np.abs(x.morph_asymm), axis=1)
    df['ZBin']=pd.qcut(df.Z, 6)
    df['categoryn']=df.apply(lambda x: categorise(x.asymm, x.n_total, 1.5), axis=1)
    df['categorybt']=df.apply(lambda x: categorise(x.asymm, x.BulgeToTotal, 0.5), axis=1)
    df['categoryn2d']=df.apply(lambda x: categorise(x.asymm, x.n2d, 1.5), axis=1)
    df['colourn']=df.apply(lambda x: colourise(x.asymm, x.n_total, 1.5), axis=1)
    df['colourbt']=df.apply(lambda x: colourise(x.asymm, x.BulgeToTotal, 0.5), axis=1)
    df['colourn2d']=df.apply(lambda x: colourise(x.asymm, x.n2d, 1.5), axis=1)
    df['vHsqrd']=df.apply(lambda x: divide(6.67*10e-11*x.M200, x.R200), axis=1)
    print(df.shape)

    df['sSFR']=df.apply(lambda x: divide(x.SFR,x.Starmass), axis=1)
    df['logsSFR']=df.apply(lambda x: logx(x.sSFR), axis=1)
    df['n2d_error']=df.apply(lambda x: x.n2d_error/(256*2), axis=1)
    df['n1total']=df.apply(lambda x: x.n_total, axis=1)
    df['n1total_error']=df.apply(lambda x: extract_val(x.n_total_error1), axis=1)
    df['n1total_error']= df['n1total_error'].astype(float)
    df['n1total_error']=df.apply(lambda x: x.n1total_error*50, axis=1)
    df=df[df.n2d_error<1]
    df=df[df.n_total>0]
    df=df[df.n2d>0]
    df=df[df.logsSFR<0]
    df=df[df.asymm<0.3]
    df['loggas']=df.apply(lambda x: logx(x.Gasmass), axis=1)
    df['sgas']=df.apply(lambda x: divide(x.Gasmass,x.Starmass), axis=1)
    df['logsgas']=df.apply(lambda x: logx(x.sgas), axis=1)
    df['sDMmassType']=df.apply(lambda x: divide(x.StarmassType,x.DMmass), axis=1)
    df['logsDMmassType']=df.apply(lambda x: logx(x.sDMmassType), axis=1)
    df['sBHmass']=df.apply(lambda x: divide(x.BHmass,x.Starmass), axis=1)
    df['logsBHmass']=df.apply(lambda x: logx(x.sBHmass), axis=1)
    df['logmassType']=df.apply(lambda x: logx(x.StarmassType), axis=1)
    df['sSFRType']=df.apply(lambda x: divide(x.SFR*1e12,x.StarmassType), axis=1)
    df['logsSFRType']=df.apply(lambda x: logx(x.sSFRType), axis=1)
    df['logMsMh']=df.apply(lambda x: divide(x.logmass, x.logDMmass), axis=1)
    df=df[df.logDMmass>11]
    df['sSFR']=df.apply(lambda x: divide(x.SFR,x.Starmass), axis=1)
    df['logsSFR']=df.apply(lambda x: logx(x.sSFR), axis=1)

    df['n_total']=df.apply(lambda x: threshtonan(x.n_total, 0.4), axis=1)
    df['n_disc']=df.apply(lambda x: threshtonan(x.n_disc, 0.1), axis=1)
    df['n_bulge']=df.apply(lambda x: threshtonan(x.n_disc, 0.4), axis=1)
    df['n_bulge_exp']=df.apply(lambda x: threshtonan(x.n_disc, 0.1), axis=1)
    df['n2d']=df.apply(lambda x: threshtonan(x.n_disc, 0.4), axis=1)
    df['n_total']=df.apply(lambda x: threshtonanupper(x.n_total, 5), axis=1)
    df['n_disc']=df.apply(lambda x: threshtonanupper(x.n_disc, 4), axis=1)
    df['n_bulge']=df.apply(lambda x: threshtonanupper(x.n_disc, 6), axis=1)
    df['n_bulge_exp']=df.apply(lambda x: threshtonanupper(x.n_disc, 5), axis=1)
    df['n2d']=df.apply(lambda x: threshtonanupper(x.n_disc, 5), axis=1)

    df=df[df.logsSFR<0]
    df=df[df.logsSFR>-12]
    n2df=df[df.n2d>0.5]
    ndf=df[df.n_total>0.5]
    n2ndf=ndf[ndf.n2d>0.5]
    #vdf=df.dropna(subset=['vHsqrd'])
    #vdf=vdf[vdf.vHsqrd<40]
    #n2dvdf=vdf[vdf.n2d>0]

def threeDplot(df, x,y,z, column_size, column_colour):
    #produce 3D plot
    df['BHmassbin']=pd.cut(df.logBHmass, 30)
    df['BHmasscounts']=df.groupby('BHmassbin')['BHmassbin'].transform('count')
    df['fracofbin']=df.apply(lambda x: (10.0/x.BHmasscounts), axis=1)
    print(df[['logBHmass','BHmassbin', 'BHmasscounts', 'fracofbin']])
    fig=plt.figure()
    minx=df[x].min()
    miny=df[y].min()
    minz=df[z].min()
    maxx=df[x].max()
    maxy=df[y].max()
    maxz=df[z].max()
    size=70*(df[column_size])/(df[column_size].max())
    ax = fig.gca(projection='3d')
    norm=plt.Normalize(df[column_colour].min(), df[column_colour].max())
    sm=plt.cm.ScalarMappable(cmap='autumn', norm=norm)
    sm.set_array([])
    ax.scatter(df[x],df[y],df[z], c=sm.to_rgba(df[column_colour]), s=size)
    
    xi = np.linspace(minx, maxx, 100)
    yi = np.linspace(miny, maxy, 100)
    zi = np.linspace(minz, maxz, 100)
    
    hist, binx, biny=np.histogram2d(df[y], df[x],  bins=7)
    X = np.linspace(minx, maxx, hist.shape[0])
    Y = np.linspace(miny, maxy, hist.shape[1])
    X,Y=np.meshgrid(X,Y)
    ax.contourf(X,Y,hist, zdir='z', offset=minz, cmap=cm.YlOrRd, alpha=0.4)
    
    hist, binx, biny=np.histogram2d(df[z], df[x], bins=7)
    X = np.linspace(minx, maxx, hist.shape[0])
    Z = np.linspace(minz, maxz, hist.shape[1])
    X,Z=np.meshgrid(X,Z)
    ax.contourf(X,hist,Z, zdir='y', offset=maxy, cmap=cm.YlOrRd, alpha=0.4)

    hist, binx, biny=np.histogram2d(df[y], df[z], bins=7)
    Y = np.linspace(miny, maxy, hist.shape[0])
    Z = np.linspace(minz, maxz, hist.shape[1])
    Z,Y=np.meshgrid(Z,Y)
    ax.contourf(hist,Y,Z, zdir='x', offset=minx, cmap=cm.YlOrRd, alpha=0.4)
    """
    
    C1 = griddata((df[x], df[y]), df['BHmasscounts'], (xi[None,:], yi[:,None]), method='linear')
    X1, Y1 = np.meshgrid(xi, yi)
    ax.contourf(X1, Y1, C1, zdir='z', offset=minz, cmap=cm.YlOrRd, alpha=0.4)
    
    C2 = griddata((df[y], df[z]), df['BHmasscounts'], (yi[None,:], zi[:,None]), method='linear')
    Y2, Z2 = np.meshgrid(yi, zi)
    ax.contourf(C2, Y2, Z2, zdir='x', offset=minx, cmap=cm.YlOrRd, alpha=0.4)
    

    C3 = griddata((df[x], df[z]), df['BHmasscounts'], (xi[None,:], zi[:,None]), method='linear')
    X3, Z3 = np.meshgrid(xi, zi)
    ax.contourf(X3, C3, Z3, zdir='y', offset=maxy, cmap=cm.YlOrRd, alpha=0.4)
    """
    #ax.scatter(df[x], df[y],  zdir='z', zs=minz, c=sm1.to_rgba(df[column_colour]), marker='*',s=size)
    #ax.scatter(df[y], df[z], zdir='x', zs=minx, c=sm2.to_rgba(df[column_colour]), s=size)
    #ax.scatter(df[x], df[z], zdir='y', zs=maxy,  c=sm3.to_rgba(df[column_colour]), s=size)
    
    
    ax.set_xlim(minx,maxx)
    ax.set_ylim(miny,maxy)
    ax.set_zlim(minz,maxz)
    fig.colorbar(sm).set_label(column_colour)
    ax.set_xlabel(x), ax.set_ylabel(y),ax.set_zlabel(z)

    plt.show()

def colorbarplot(df, x,y, column_size, column_colour, column_marker, xname, yname, colourname):
    #produce coloured scatter plot with colorbar and varying sizes and shapes of markers.
    Norm=mcol.DivergingNorm(vmin=df[column_colour].min(), vcenter=df[column_colour].median(), vmax=df[column_colour].max())
    #Cmap=mcol.LinearSegmentedColormap.from_list("cmop", ['red','mediumorchid','blue'])
    Cmap='Autumn'
    df['marker_bin']=pd.qcut(df.Z, [0,0.15,0.85,1], labels=['low','okay','high'])
    markers={"low":'^', "okay":'o', 'high':'s'}
    sm=plt.cm.ScalarMappable(cmap=Cmap, norm=Norm)
    sm.set_array([])
    ax=sns.relplot(x=x, y=y, size=column_size, sizes=(10,100), hue=column_colour, palette=Cmap, style='marker_bin', markers=markers,data=df)
    ax._legend.remove()
    ax.set(xlabel=xname, ylabel=yname)
    ax.fig.colorbar(sm).set_label(colourname)
    plt.subplots_adjust(top=0.9)
    #ax.fig.suptitle(''+x+' vs '+y+', coloured by'+column_colour+', sized by'+column_size+', shaped by'+column_marker+'')
    ax.savefig('galaxygraphsbin'+sim_name+'/'+x+'vs'+y+'2.png')
    plt.show()

def stackedhistogram(df, param1, param2, param3, param5):
    #produce plot of stacked histograms and their errors
    plt.subplot(211)
    colors1=['yellow','r','blue','green']
    colors2=['r','blue','green']
    labels=[param5, param1, param2, param3]
    plt.title('Histograms of Sersic Indices and Errors')
    plt.hist([df[param5], df[param1],df[param2],df[param3]], bins=20,  color=colors1, label=labels)
    plt.xlabel('Sersic Index')
    plt.legend()
    
    df[df.n_total_error>10]=np.nan
    df[df.n_disc_error>12]=np.nan
    df[df.n_bulge_error>12]=np.nan
    
    plt.subplot(212)
    plt.hist([df[param1+'_error'],df[param2+'_error'],df[param3+'_error']], bins=20,color=colors2, label=labels)
    plt.xlabel('Error')
    plt.tight_layout()
    plt.savefig('galaxygraphsbin'+sim_name+'/histogramof'+param1+param2+param3+param5+'.png')
    plt.show()

def disthistogram(df, param1, param2, param3, param4, name1,name2,name3,name4):
    #produce histogrms using distplot
    fig, axs =plt.subplots(2,1)
    colors1=['yellow','r','blue','green']
    colors2=['r','blue','green']
    labels=[param1, param2, param3,param4]
    df[df.n_disc>6]=np.nan
    names=[name1,name2,name3,name4]
    for i,param in enumerate(labels):
        print(param)
        temp=df[df[param]>0.1]
        temp['n1total']=df.apply(lambda x: x.n1total +1, axis=1)
        temp['n2d']=df.apply(lambda x: x.n2d +1, axis=1)
        temp['n_disc']=df.apply(lambda x: x.n_disc +1, axis=1)
        temp['n_bulge']=df.apply(lambda x: x.n_bulge +1, axis=1)
        sns.distplot(temp[param], label=names[i], ax=axs[0], hist=False, kde=True, norm_hist=True)
    axs[0].set_xlabel(r'Sersic Index')
    axs[0].set_ylabel('Count')
    axs[0].legend()
    axs[0].set_xlim(1,8)
    axs[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:('%g') % (x * 50)))
    axs[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:('%g') % (x -1)))


    df[df.n_total_error>8]=np.nan
    df[df.n_disc_error>8]=np.nan
    df[df.n_bulge_error>8]=np.nan
    
    for param in labels:
        temp=df[df[param+'_error']>0]
        temp['n1total_error']=df.apply(lambda x: x.n1total_error +1, axis=1)
        temp['n2d_error']=df.apply(lambda x: x.n2d_error +1, axis=1)
        temp['n_disc_error']=df.apply(lambda x: x.n_disc_error +1, axis=1)
        temp['n_bulge_error']=df.apply(lambda x: x.n_bulge_error +1, axis=1)
        sns.distplot(temp[param+'_error'], ax=axs[1], hist=False,  kde=True, norm_hist=True)
        #sns.distplot(temp[param+'_error'], hist=False, norm_hist=False, kde=False)
    axs[1].set_xlabel('Error')
    axs[1].set_ylabel('Count')
    axs[1].set_xlim(1,10)
    axs[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:('%g') % (x -1)))
    axs[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:('%g') % (x * 50)))
    #locs, labels = axs[1].get_xticks()
    #labels = [60*j for j in locs]
    #axs[1].set_xticks(locs, labels)
    plt.tight_layout()
    plt.savefig('galaxygraphsbin'+sim_name+'/histogramof'+param1+param2+param3+param4+'.png')
    plt.show()

def subplothistograms(df, param1, param2, param3, param4, param5, param6):
    #produce individual subplots of histograms
    plt.subplot(6,2,1)
    plt.hist(df[param2], 50)
    plt.xlabel(param2)
    plt.subplot(6,2,2)
    plt.hist(df[param2+'_error'], 50)
    plt.xlabel(param2+'_error')
    
    plt.subplot(6,2,3)
    plt.hist(df[param2], 50)
    plt.xlabel(param2)
    plt.subplot(6,2,4)
    plt.hist(df[param2+'_error'], 50)
    plt.xlabel(param2+'_error')

    plt.subplot(6,2,5)
    plt.hist(df[param3], 50)
    plt.xlabel(param3)
    plt.subplot(6,2,6)
    plt.hist(df[param3+'_error'], 50)
    plt.xlabel(param3+'_error')

    plt.subplot(6,2,7)
    plt.hist(df[param4], 50)
    plt.xlabel(param4)
    plt.subplot(6,2,8)
    plt.hist(df[param4+'_error'], 50)
    plt.xlabel(param3+'_error')

    plt.subplot(6,2,9)
    plt.hist(df[param5], 50)
    plt.xlabel(param5)
    plt.subplot(6,2,10)
    plt.hist(df[param5+'_error'], 50)
    plt.xlabel(param5+'_error')

    plt.subplot(6,2,11)
    plt.hist(df[param6], 50)
    plt.xlabel(param6)
    plt.subplot(6,2,12)
    plt.hist(df[param6+'_error'], 50)
    plt.xlabel(param6+'_error')

    plt.tight_layout()
    plt.show()

def binvalue(df, paramx, paramy, binno):
    #bin parameter, giving median, upper and lower quantiles, standard deviation and bin edges
    binedgs=np.linspace(df[paramx].min(), df[paramx].max(), binno)
    binedgs2=np.linspace(df[paramx].min(), df[paramx].max(), binno -1)
    medianvals=[]
    stdvals=[]
    lowquart=[]
    uppquart=[]
    for i in range(0,binno -1):
        bindf=df[df[paramx]>binedgs[i]]
        bindf=bindf[bindf[paramx]<binedgs[i+1]]
        med=bindf[paramy].median()
        std=(bindf[paramy].std())/2
        low=bindf[paramy].quantile(0.10)
        high=bindf[paramy].quantile(0.90)

        medianvals.append(med)
        stdvals.append(std)
        lowquart.append(low)
        uppquart.append(high)
    return medianvals, binedgs2, lowquart, uppquart, stdvals

def colourscatter(df,x,y, column_colour, thresh):
    #produce scatter plot with PDFS
    Norm=mcol.DivergingNorm(vmin=df[column_colour].min(), vcenter=thresh, vmax=df[column_colour].max())
    #Cmap=mcol.LinearSegmentedColormap.from_list("cmop", ['tomato','cornflowerblue'])
    Cmap='seismic'
    fig=plt.figure()
    gs=fig.add_gridspec(4,4)
    ax1=fig.add_subplot(gs[1:,1:])
    axtop=fig.add_subplot(gs[0, 1:])
    axleft=fig.add_subplot(gs[1:, 0])

    #fig, axs=plt.subplots(2,2, sharey='row', sharex='col')
    sm=plt.cm.ScalarMappable(cmap=Cmap, norm=Norm)
    sm.set_array([])

    #create pdf
    N=len(df)
    n=int(N/20)

    axleft.set_xlabel('PDF')
    axtop.set_ylabel('PDF')
    axleft.set_ylabel(r'log(SFR) [$M_{\odot}Gyr^{-1}$]')
    ax1.set_xlabel(r'$log(M_{gas})  [M_{\odot}])$')

    ax1.scatter(df[x],df[y], c=df[column_colour], cmap=Cmap, norm=Norm, alpha=0.5, s=10)
    lowdf=df[df[column_colour]<thresh -0.1]
    highdf=df[df[column_colour]>thresh +0.1]
    dflist=[df, highdf, lowdf]
    cs=['k', 'r', 'b']
    for i,df in enumerate(dflist):
        medianvals, binedgs, lowquart, uppquart, std=binvalue(df, x, y, 10)
        ax1.errorbar(binedgs, medianvals, color=cs[i], yerr=(std), fmt='', capsize=0.5, elinewidth=0.5)

        py,y1=np.histogram(df[y], bins=n)
        y1=y1[:-1]+(y1[1]-y1[0])/2
        f=UnivariateSpline(y1,py,s=n)
        axleft.plot(f(y1), y1, color=cs[i])

        px,x1=np.histogram(df[x], bins=n)
        x1=x1[:-1]+(x1[1]-x1[0])/2
        f=UnivariateSpline(x1,px,s=n)
        axtop.plot(x1, f(x1), color=cs[i])


    plt.subplots_adjust(right=0.8, wspace=0, hspace=0)
    cbar_ax=fig.add_axes([0.85,0.15,0.05,0.8])
    sm=plt.cm.ScalarMappable(cmap=Cmap, norm=Norm)
    sm.set_array([])
    cbar=plt.colorbar(sm, cax=cbar_ax).set_label(column_colour)

    plt.savefig('galaxygraphsbin'+sim_name+'/plot'+x+''+y+'colouredby'+column_colour+'.png')
    plt.show()

def plotbulgetodisc(df, sim_name):
    #plot different plots. A few examples are shown
    print(df.columns.values)

    colorbarplot(df, 'btdradius', 'BulgeToTotal', 'logmass', 'logsSFR', 'logBHmass', r'$r_{bulge} / r_{disc}$', 'BulgeToTotal',r'log(sSFR) [$yr^{-1}$]')
    disthistogram(df, 'n1total', 'n2d', 'n_disc', 'n_bulge', r'$n_{1d}$', r'$n_{2d}$',r'$n_{disc}$',r'$n_{bulge}$' )
    #stackedhistogram(df, 'n_total','n_disc','n_bulge', 'n2d')
    colourscatter(df, 'gas','sSFR','n2d', 1.4)
    threeDplot(df, 'asymm','DiscToTotal','logBHmass', 'mass', 'logsSFR')
    stackedhistogram(df, 'n_total','n_disc','n_bulge','n_bulge_exp', 'morph_sersic_n')

if __name__ == "__main__":
    sim_name='RefL0050N0752'
    read_all_data=False
    read_extra_data=True
    read_image_data=False
    if(read_all_data):
        print('.........reading.......')
        df=pd.read_csv('EAGLEbulgedisc'+sim_name+'final.csv')
    else:
        if(read_image_data):
            print('.........writing.......')
            df=pd.read_csv('EAGLEimagesdf'+sim_name+'.csv')
            discbulgetemp=[]
            for filename in df['filename']:
                BGRimage=cv2.imread('galaxyimagebin'+sim_name+'/'+filename)
                btdradius, btdintensity, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius =findandlabelbulge(BGRimage, filename, sim_name)
                n_total, n2d,n2d_error, n_disca, n_bulgea, n_disc, n_bulge, n_bulge_exp, n_total_error,n_total_error1, n_disca_error, n_bulgea_error, n_disc_error, n_bulge_error, n_bulge_exp_error, con, r80, r20, asymm, asymmerror=findsersicindex(BGRimage, bradius, hradius)
                #morph_c, morph_asymm, morph_sersic_n, morph_smoothness, morph_sersic_rhalf, morph_xc_asymmetry, morph_yc_asymmetry=runstatmorph(BGRimage)
                #discbulgetemp.append([filename, btdradius, btdintensity,n_total, n2d, n_disca, n_bulgea, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disca_error, n_bulgea_error, n_disc_error, n_bulge_error, n_bulge_exp_error, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius, con, r80, r20, asymm, morph_c, morph_asymm, morph_sersic_n, morph_smoothness, morph_sersic_rhalf, morph_xc_asymmetry, morph_yc_asymmetry])
                discbulgetemp.append([filename, btdradius, btdintensity,n_total, n2d,n2d_error, n_disca, n_bulgea, n_disc, n_bulge, n_bulge_exp, n_total_error,n_total_error1, n_disca_error, n_bulgea_error, n_disc_error, n_bulge_error, n_bulge_exp_error, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius, con, r80, r20, asymm, asymmerror])
            #discbulgedf=pd.DataFrame(discbulgetemp, columns=['filename', 'btdradius', 'btdintensity','n_total','n2d','n_disca','n_bulgea','n_disc','n_bulge','n_bulge_exp', 'n_total_error', 'n_disca_error', 'n_bulgea_error', 'n_disc_error', 'n_bulge_error', 'n_bulge_exp_error', 'star_count', 'discradius', 'bulgeradius', 'disc_intensity', 'bulge_intensity', 'btotalintensity', 'btotalradius', 'con', 'r80', 'r20', 'asymm', 'morph_c', 'morph_asymm', 'morph_sersic_n', 'morph_smoothness', 'morph_sersic_rhalf', 'morph_xc_asymmetry', 'morph_yc_asymmetry'])
            discbulgedf=pd.DataFrame(discbulgetemp, columns=['filename', 'btdradius', 'btdintensity','n_total','n2d','n2d_error','n_disca','n_bulgea','n_disc','n_bulge','n_bulge_exp', 'n_total_error','n_total_error1', 'n_disca_error', 'n_bulgea_error', 'n_disc_error', 'n_bulge_error', 'n_bulge_exp_error', 'star_count', 'discradius', 'bulgeradius', 'disc_intensity', 'bulge_intensity', 'btotalintensity', 'btotalradius', 'con', 'r80', 'r20', 'asymm', 'asymmerror'])
            df.filename.astype(str)
            discbulgedf.filename.astype(str)
            df=pd.merge(df, discbulgedf, on=['filename'], how='outer')
            df.to_csv('EAGLEbulgedisc'+sim_name+'.csv')

        if(read_extra_data):
            print('.........reading.......')
            df1=pd.read_csv('EAGLEbulgedisc'+sim_name+'.csv')
            cleanandtransformdata(df1)
            df2=pd.read_csv('EAGLEextradatadf'+sim_name+'.csv')
            df= pd.merge(df1, df2, on=['filename'], how='left').drop_duplicates()
            df.to_csv('EAGLEbulgedisc'+sim_name+'final.csv')
        
    plotbulgetodisc(df, sim_name)


    


