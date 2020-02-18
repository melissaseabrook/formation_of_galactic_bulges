import numpy as np
from numpy import *
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
from matplotlib import cm
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.colors as mcol
import seaborn as sns
import pygtc
from PIL import Image
import cv2
from imutils import contours
from skimage import measure
import imutils
import photutils
from scipy import stats
from scipy.optimize import curve_fit
import scipy.ndimage as ndi
from scipy.interpolate import griddata, UnivariateSpline
import pylab
import networkx as nx
from astropy.cosmology import Planck13
from astropy import constants as const
from astropy.modeling import models, fitting
import statmorph


#sns.set_style('whitegrid')
def logx(x):
    if x !=0:
        if x>0:
            return np.log10(x)
        if x<0:
            return -np.log10(-x)
    else:
        return 0

def divide(x,y):
    if y !=0:
        return x/y
    else:
        return 0

def invert(var):
    if var != 0:
        return(1/var)*10
    else:
        return 0

def zerotonan(x):
    #convert zeroes to nans
    if x==0:
        return np.nan
    else:
        return x

def zerotonancappedz(frac, z):
    #convert zeroes to nans
    if z<0.001:
        return np.nan
    elif frac==0:
        return np.nan
    else:
        return frac

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
    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median=np.median(gray)
    std=np.std(gray)
    blurred1 = cv2.GaussianBlur(gray, ksize=(7, 7), sigmaX=3,sigmaY=3)
    if (std>20):
        thresh1 = cv2.threshold(blurred1,  (3.5*std)+median, 255, cv2.THRESH_BINARY)[1]
    else:
        thresh1 = cv2.threshold(blurred1,  (6.5*std) +median, 255, cv2.THRESH_BINARY)[1]
    thresh1 = cv2.erode(thresh1, None, iterations=2)
    thresh1 = cv2.dilate(thresh1, None, iterations=4)

    blurred2 = cv2.GaussianBlur(gray, ksize=(15, 15), sigmaX=3,sigmaY=3)
    thresh2 = cv2.threshold(blurred2, median +(1.5*std), 255, cv2.THRESH_BINARY)[1]
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
                (x, y, w, h) = cv2.boundingRect(c)
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
    print("disc intensity = {}, bulge intensity ={}, disc:bulge intensity ={}".format(disc_intensity, bulge_intensity, btdintensity))
    cv2.destroyAllWindows()
    return btdradius, btdintensity, count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius

def invertbtd(r):
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
    radialprofile, r_arr, binnumber=stats.binned_statistic(r.ravel(),image.ravel(), 'mean', bins=len(radialprofile))
    #meanbins, bin_edges, binnumber=stats.binned_statistic(r.ravel(),image.ravel(), 'mean', bins=len(radialprofile))
    return radialprofile, r_arr, stdbins, nr

def findeffectiveradius(radialprofile, r, nr):
    totalbrightness=np.sum(radialprofile * 2 * np.pi *r*nr)
    centralbrightness=radialprofile[0]
    cumulativebrightness=np.cumsum(radialprofile * 2 * np.pi *r*nr)
    r_e_unnormalised=((np.abs((totalbrightness/2) - cumulativebrightness)).argmin())
    r_e=r_e_unnormalised*(30.0/256)
    i_e= radialprofile[r_e_unnormalised]
    return i_e, r_e, centralbrightness, totalbrightness

def SersicProfile(r, I_e, R_e, n):
    b=np.exp(0.6950 + np.log(n) - (0.1789/n))
    G=(r/R_e)**(1/n)
    return I_e*np.exp((-b*(G-1)))

def findeffectiveradiusfrac(radialprofile, r, nr, frac):
    print(len(radialprofile), len(r),len(nr))
    totalbrightness=np.sum(radialprofile * 2 * np.pi *nr*r)
    centralbrightness=radialprofile[0]
    cumulativebrightness=np.cumsum(radialprofile * 2 * np.pi *nr*r)
    r_e_index=((np.abs((totalbrightness*frac) - cumulativebrightness)).argmin())
    r_e=r[r_e_index]
    i_e= radialprofile[r_e_index]
    return i_e, r_e, centralbrightness, totalbrightness

def findconcentration(rad, r, nr):
    i20, r80, cb,tb=findeffectiveradiusfrac(rad, r, nr, 0.8)
    i20, r20, cb, tb=findeffectiveradiusfrac(rad, r, nr, 0.2)
    con=5*np.log10(r80/r20)
    return con, r80,r20

def findassymetry(image):
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
        #sim=sersic_model(x,y)
        #=ma.log10(image)
        #logimg=logimg.filled(0)
        #logsim=ma.log10(sim)
        #logsim=logsim.filled(0)
        #res = logx(np.sum(np.abs(logimg - logsim)))
        #nd_error=np.sqrt(res)
        nd_error=np.nan
    except:
        nd=np.nan
        nd_error=np.nan
    return nd, nd_error

def findsersicindex(image, bindex, dindex):
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

        n1, pcov1 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r, rad, p0=3, bounds=(0.0001,10), sigma=stdbins, absolute_sigma=True)
        print("I_e={}, R_e={}, n_disc={}".format(i_e, r_e, n1))
        n_total=n1[0]
        n_total_error=pcov1[0,0]

        n2d, n2d_error=twoDsersicfit(image2, i_e, r_e, n_total, center)


        poptdisc, pcovdisc = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r[bindex:dindex], rad[bindex:dindex], sigma=stdbins[bindex:dindex], bounds=(0.0001,10), absolute_sigma=True)
        n_disc=poptdisc[0]
        n_disc_error=pcovdisc[0,0]
        print("I_edisc={}, R_edisc={}, n_disc={}".format(i_e, r_e, n_disc))
        isolated_discsim=SersicProfile(r, i_e, r_e, n_disc)
        isolated_bulge= rad - isolated_discsim
        isolated_bulge[isolated_bulge<0]=0
        i_ebulge, r_ebulge, centralbrightnessbulge, totalbrightnessbulge= findeffectiveradius(isolated_bulge[0:bdindex], r[0:bdindex], nr[0:bdindex]) 
        poptbulge, pcovbulge = curve_fit(lambda x,n: SersicProfile(x, i_ebulge, r_ebulge, n), r[0:bindex], isolated_bulge[0:bindex],p0=4, sigma=stdbins[0:bindex], bounds=(0,10), absolute_sigma=True)
        n_bulge= poptbulge[0]
        n_bulge_error= pcovbulge[0,0]
        print("I_ebulge={}, R_ebulge={}, n_bulge={}".format(i_ebulge,r_ebulge, n_bulge))


        exponential_discsim=SersicProfile(r, i_e, r_e, 1)
        isolated_bulge2= rad - exponential_discsim
        isolated_bulge2[isolated_bulge2<0]=0
        i_ebulge2, r_ebulge2, centralbrightnessbulge2, totalbrightnessbulge2= findeffectiveradius(isolated_bulge2[0:bdindex], r[0:bdindex], nr[0:bdindex]) 
        poptbulge2, pcovbulge2 = curve_fit(lambda x,n: SersicProfile(x, i_ebulge2, r_ebulge2, n), r[0:bindex], isolated_bulge[0:bindex],p0=4, bounds=(0,10), sigma=stdbins[0:bindex], absolute_sigma=True)
        n_bulge_exp= poptbulge[0]
        n_bulge_exp_error= pcovbulge2[0,0]
        print("n_bulge={}".format(n_bulge_exp))


    except:
        n_total=np.nan
        n2d=np.nan
        n2d_error = np.nan
        n_bulge=np.nan
        n_disc=np.nan
        n_bulge_exp=np.nan
        n_total_error=np.nan
        n_bulge_error=np.nan
        n_disc_error=np.nan
        n_bulge_exp_error=np.nan
        con=np.nan
        r80=np.nan
        r20=np.nan
    return n_total, n2d, n2d_error, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disc_error, n_bulge_error, n_bulge_exp_error, con, r80, r20, asymm, asymmerror

def runstatmorph(image):
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
    constrains=df.select_dtypes(include=[np.number]).apply(lambda x: np.abs(stats.zscore(x)) <z_thresh).all(axis=1)
    df.drop(df.index[~constrains], inplace=True)

def removeoutlierscolumn(df, column_name, sigma):
    df=df[np.abs(df[column_name]-df[column_name].mean())<=(sigma*df[column_name].std())]
    return df

def getImage(path):
    return OffsetImage(plt.imread('evolvinggalaxyimagebinmainbranch'+sim_name+'/'+path), zoom=0.15)

def categorise(asymm, param, thresh):
    if asymm > 0.35:
        return 'A'
    elif param > thresh:
        return 'B'
    else:
        return 'D'

def cleanandtransformdata(df):
    print(df.shape)
    df.sort_values(['z','ProjGalaxyID'], ascending=[False,True], inplace=True)
    df['lbt']=df.apply(lambda x: -round(Planck13.lookback_time(x.z).value, 1), axis=1)
    df['lbt2']=df.apply(lambda x: round(Planck13.lookback_time(x.z).value, 1), axis=1)
    df['zrounded']=df.apply(lambda x: np.round(x.z, decimals=1), axis=1)

    
    df['lookbacktime']=df.apply(lambda x: -(Planck13.lookback_time(x.z).value)*(1e9), axis=1)
    """
    df['dlbt']=df.groupby('ProjGalaxyID')['lookbacktime'].diff()
    df['dSFR']=df.groupby('ProjGalaxyID')['SFR'].diff()
    df['dBHmass']=df.groupby('ProjGalaxyID')['BHmass'].diff()
    df['dSIM']=df.groupby('ProjGalaxyID')['StellarInitialMass'].diff()
    df['dD2T']=df.groupby('ProjGalaxyID')['DiscToTotal'].diff()
    df['dn_total']=df.groupby('ProjGalaxyID')['n_total'].diff()
    df['dz']=df.groupby('ProjGalaxyID')['z'].diff()
    df['dz']=df.apply(lambda x: -x.dz, axis=1)
    """

    """
    df['dSFRdz']=df.apply(lambda x: (x.dSFR)/(x.dz), axis=1)
    df['dBHmassdz']=df.apply(lambda x: (x.dBHmass)/(x.dz), axis=1)
    df['dSIMdz']=df.apply(lambda x: (x.dSIM)/(x.dz), axis=1)
    df['dn_totaldz']=df.apply(lambda x: (x.dn_total)/(x.dz), axis=1)
    df['dD2Tdz']=df.apply(lambda x: (x.dD2T)/(x.dz), axis=1)
    df['dSFRdt']=df.apply(lambda x: (x.dSFR)/(x.dlbt), axis=1)
    df['dBHmassdt']=df.apply(lambda x: (x.dBHmass)/(x.dlbt), axis=1)
    df['dSIMdt']=df.apply(lambda x: (x.dSIM)/(x.dlbt), axis=1)
    df['dn_totaldt']=df.apply(lambda x: (x.dn_total)/(x.dlbt), axis=1)
    df['dD2Tdt']=df.apply(lambda x: (x.dD2T)/(x.dlbt), axis=1)
    """

    #drop_numerical_outliers(df, 3)
    #df=df=df.reset_index()
    #print(df.shape)
    
    df['num']= df.groupby('ProjGalaxyID')['ProjGalaxyID'].transform('count')
    print(df.shape)
    #df=df[df.num>7]
    print(df.ProjGalaxyID.nunique())
    df['BulgeToTotal']=df.apply(lambda x: (1-x.DiscToTotal), axis=1)
    df['logBHmass']=df.apply(lambda x: logx(x.BHmass), axis=1)
    df['logDMmass']=df.apply(lambda x: logx(x.BHmass), axis=1)
    df['logmass']=df.apply(lambda x: logx(x.Starmass), axis=1)
    df['loggasmass']=df.apply(lambda x: logx(x.Gasmass), axis=1)
    df['sSFR']=df.apply(lambda x: divide(x.SFR,x.Starmass), axis=1)
    df['sBHmass']=df.apply(lambda x: divide(x.BHmass,x.Starmass), axis=1)
    df['sDMmass']=df.apply(lambda x: divide(x.Starmass, x.DMmass), axis=1)
    df['logSFR']=df.apply(lambda x: logx(x.SFR), axis=1)
    df['logsSFR']=df.apply(lambda x: logx(x.sSFR), axis=1)
    df['logsBHmass']=df.apply(lambda x: logx(x.sBHmass), axis=1)
    df['logsDMmass']=df.apply(lambda x: logx(x.sDMmass), axis=1)
    df['logDMmass']=df.apply(lambda x: logx(x.DMmass), axis=1)
    df['dtototal']=df.apply(lambda x: (1-x.btdintensity), axis=1)
    df['dtbradius']=df.apply(lambda x: invertbtd(x.btdradius), axis=1)
    df['dtbintensity']=df.apply(lambda x: invertbtd(x.btdintensity), axis=1)

    df['categoryn']=df.apply(lambda x: categorise(x.asymm, x.n_total, 1.5), axis=1)
    df['categorybt']=df.apply(lambda x: categorise(x.asymm, x.BulgeToTotal, 0.5), axis=1)
    df['categoryn2d']=df.apply(lambda x: categorise(x.asymm, x.n2d, 1.6), axis=1)

    df['massquantile']=pd.qcut(df['logmass'], 5, labels=False)
    grouped=df[['zrounded','massquantile','sSFR']].groupby(['zrounded','massquantile']).agg({'sSFR':['median', 'std']})
    grouped=grouped.xs('sSFR', axis=1, drop_level=True)
    df=pd.merge(df, grouped, on=['zrounded','massquantile'], how='left')
    df=df.rename({'median':'sSFR_median', 'std':'sSFR_std'}, axis=1)
    df['sSFRpermass']=df.apply(lambda x: divide((x.sSFR-x.sSFR_median)*1e14, x.sSFR_std*1e12), axis=1)
    df['logsSFRpermass']=df.apply(lambda x: logx(x.sSFRpermass), axis=1)

    grouped=df[['zrounded','massquantile','logsDMmass']].groupby(['zrounded','massquantile']).agg({'logsDMmass':['median', 'std']})
    grouped=grouped.xs('logsDMmass', axis=1, drop_level=True)
    df=pd.merge(df, grouped, on=['zrounded','massquantile'], how='left')
    df=df.rename({'median':'logsDMmass_median', 'std':'logsDMmass_std'}, axis=1)
    df['sDMmasspermass']=df.apply(lambda x: divide((x.logsDMmass-x.logsDMmass_median)*1e14, x.logsDMmass_std*1e12), axis=1)
    df['logsDMmasspermass']=df.apply(lambda x: logx(x.sDMmasspermass), axis=1)

    grouped=df[['zrounded','massquantile','DMEllipticity']].groupby(['zrounded','massquantile']).agg({'DMEllipticity':['median', 'std']})
    grouped=grouped.xs('DMEllipticity', axis=1, drop_level=True)
    df=pd.merge(df, grouped, on=['zrounded','massquantile'], how='left')
    df=df.rename({'median':'DMEllipticity_median', 'std':'DMEllipticity_std'}, axis=1)
    df['DMEllipticitypermass']=df.apply(lambda x: divide((x.DMEllipticity-x.DMEllipticity_median)*1e14, x.DMEllipticity_std*1e12), axis=1)
    df['logDMEllipticitypermass']=df.apply(lambda x: logx(x.DMEllipticitypermass), axis=1)



    """

    df['roundlogmass']=df.apply(lambda x: (np.round(x.logmass*2, decimals=1)/2), axis=1)
    df['roundlogmass2']=df.apply(lambda x: (np.round(x.logmass*5, decimals=1)/5), axis=1)
    df['counts']=df.groupby(['zrounded', 'roundlogmass'])['ProjGalaxyID'].transform('count')
    df['frac']=df.apply(lambda x: 1/x.counts, axis=1)
    df['catcounts']=df.groupby(['zrounded','categoryn', 'roundlogmass'])['ProjGalaxyID'].transform('size')
    df['catfrac']=df.apply(lambda x: x.catcounts/x.counts, axis=1)

    df['roundsSFR2']=df.apply(lambda x: (np.round(x.logsSFR*2, decimals=1)/2), axis=1)
    df['roundsSFR']=df.apply(lambda x: (np.round(x.logsSFR, decimals=1)), axis=1)
    df['sfrcounts']=df.groupby(['zrounded', 'roundsSFR'])['ProjGalaxyID'].transform('size')
    df['catsfrcounts']=df.groupby(['zrounded','categoryn', 'roundsSFR'])['ProjGalaxyID'].transform('size')
    df['catsfrfrac']=df.apply(lambda x: x.catsfrcounts/x.sfrcounts, axis=1)

    df['roundBHmass2']=df.apply(lambda x: (np.round(x.logBHmass*5, decimals=1)/5), axis=1)
    df['roundBHmass']=df.apply(lambda x: (np.round(x.logBHmass*2, decimals=1)/2), axis=1)
    df['BHcounts']=df.groupby(['zrounded', 'roundBHmass'])['ProjGalaxyID'].transform('size')
    df['catBHcounts']=df.groupby(['zrounded','categoryn', 'roundBHmass'])['ProjGalaxyID'].transform('size')
    df['catBHfrac']=df.apply(lambda x: x.catBHcounts/x.BHcounts, axis=1)

    df['roundDMmass2']=df.apply(lambda x: (np.round(x.logDMmass*2, decimals=1)/2), axis=1)
    df['roundDMmass']=df.apply(lambda x: (np.round(x.logDMmass, decimals=1)), axis=1)
    df['DMcounts']=df.groupby(['zrounded', 'roundDMmass'])['ProjGalaxyID'].transform('size')
    df['catDMcounts']=df.groupby(['zrounded','categoryn', 'roundDMmass'])['ProjGalaxyID'].transform('size')
    df['catDMfrac']=df.apply(lambda x: x.catDMcounts/x.DMcounts, axis=1)
    """
    return df

def mergercolor(x):
    if x>0.01:
        return 'g'
    else:
        return 'yellow'

def mergercolor2(x, col):
    if x>df[col].median():
        return 'g'
    else:
        return 'yellow'

def mergerinvestigation(df2):
    #return df with only galaxies which have undergone a merger
    merg=df2[df2.z>0.1]
    merg=merg[merg.Starmassmergerfrac>0.]
    #merg=merg.dropna(subset=['Stargasmergerfrac'])
    print(merg['Stargasmergerfrac'])
    merggal=merg.ProjGalaxyID.unique()
    print(merggal)
    df=df2[df2.ProjGalaxyID.isin(merggal)]
    print(df)
    return df

def threeDplot(df, x,y,z, column_size, column_colour):
    df['BHmassbin']=pd.cut(df.logBHmass, 10)
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
    
    hist, binx, biny=np.histogram2d(df[y], df[x],  bins=5, weights=df['fracofbin'])
    X = np.linspace(minx, maxx, hist.shape[0])
    Y = np.linspace(miny, maxy, hist.shape[1])
    X,Y=np.meshgrid(X,Y)
    ax.contourf(X,Y,hist, zdir='z', offset=minz, cmap=cm.YlOrRd, alpha=0.6)
    
    hist, binx, biny=np.histogram2d(df[z], df[x], bins=5, weights=df['fracofbin'])
    X = np.linspace(minx, maxx, hist.shape[0])
    Z = np.linspace(minz, maxz, hist.shape[1])
    X,Z=np.meshgrid(X,Z)
    ax.contourf(X,hist,Z, zdir='y', offset=maxy, cmap=cm.YlOrRd, alpha=0.6)

    hist, binx, biny=np.histogram2d(df[y], df[z], bins=5, weights=df['fracofbin'])
    Y = np.linspace(miny, maxy, hist.shape[0])
    Z = np.linspace(minz, maxz, hist.shape[1])
    Z,Y=np.meshgrid(Z,Y)
    ax.contourf(hist,Y,Z, zdir='x', offset=minx, cmap=cm.YlOrRd, alpha=0.6)
    
    ax.set_xlim(minx,maxx)
    ax.set_ylim(miny,maxy)
    ax.set_zlim(minz,maxz)
    fig.colorbar(sm).set_label(column_colour)
    ax.set_xlabel(x), ax.set_ylabel(y),ax.set_zlabel(z)

    plt.show()

def colorbarplot(df, x,y, column_size, column_colour, column_marker):
    norm=plt.Normalize(df[column_colour].min(), df[column_colour].max())
    df['marker_bin']=pd.qcut(df[column_marker], [0,0.15,0.85,1], labels=['low','okay','high'])
    markers={"low":'^', "okay":'o', 'high':'s'}
    #Cmap=mcol.LinearSegmentedColormap.from_list("cmop", ['red','blue'])
    Cmap='autumn'
    sm=plt.cm.ScalarMappable(cmap=Cmap, norm=norm)
    sm.set_array([])
    ax=sns.relplot(x=x, y=y, size=column_size, sizes=(10,100), hue=column_colour, palette=Cmap, style='marker_bin', markers=markers,data=df)
    ax._legend.remove()
    ax.fig.colorbar(sm).set_label(column_colour)
    plt.subplots_adjust(top=0.9)
    ax.fig.suptitle(''+x+' vs '+column_marker+', coloured by'+column_colour+', sized by'+column_size+', shaped by'+column_marker+'')
    ax.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/'+x+'vs'+y+'.png')
    plt.show()

def stackedhistogram(df, param1, param2, param3, param4):
    plt.subplot(211)
    colors=['r','blue','green','purple']
    labels=[param1, param2, param3, param4]
    plt.title('Histograms of Sersic Indices and Errors')
    plt.hist([df[param1],df[param2],df[param3],df[param4]], bins=50, histtype='step', stacked=True, fill=False, color=colors, label=labels)
    plt.xlabel('Sersic Index')
    
    plt.subplot(212)
    plt.hist([df[param1+'_error'],df[param2+'_error'],df[param3+'_error'],df[param4+'_error']], bins=50, histtype='step', stacked=True, fill=False, color=colors, label=labels)
    plt.xlabel('Error')
    plt.legend()
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/histogramofsersicindices.png')
    plt.show()

def subplothistograms(df, param1, param2, param3, param4, param5, param6):

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

def evolutionplot(df, param, param_size, param2):
    fig, (ax1,ax2)=plt.subplots(2,1, sharex=True)
    #plt.subplot(211)
    sns.scatterplot(x='z',y=param, hue='ProjGalaxyID',data=df, size=param_size, palette=sns.color_palette('hls', df.ProjGalaxyID.nunique()), legend=False, ax=ax1)
    sns.lineplot(x='z',y=param, hue='ProjGalaxyID',data=df, palette=sns.color_palette('hls', df.ProjGalaxyID.nunique()), ax=ax1,legend=False, linewidth=0.8)
    ax0=ax1.twinx()
    sns.lineplot(x='z',y=param2, hue='ProjGalaxyID',data=df, palette=sns.color_palette('hls', df.ProjGalaxyID.nunique()),  ax=ax0)
    for i in range(df.ProjGalaxyID.nunique()):
        ax0.lines[i].set_linestyle('--')
    #plt.subplot(212)
    sns.lineplot(x='z',y='logsSFR', hue='ProjGalaxyID',data=df, palette=sns.color_palette('hls', df.ProjGalaxyID.nunique()), ax=ax2, legend=False)
    ax3=ax2.twinx()
    sns.lineplot(x='z',y='n_total', hue='ProjGalaxyID',data=df, palette=sns.color_palette('hls', df.ProjGalaxyID.nunique()),  ax=ax3, legend=True)
    for i in range(df.ProjGalaxyID.nunique()):
        ax3.lines[i].set_linestyle('--')

    plt.legend(bbox_to_anchor=(1.1,0.8), loc='center left')
    plt.xlim(0,1)
    plt.title('Evolution of '+param+' sized by'+param_size)
    #plt.tight_layout()
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/evolution of'+param+'and'+param2+'.png')
    
    plt.show()

def specificgalaxyplot(df, galaxyid, param1, param2, param3, param4):
    df2=df[df.ProjGalaxyID==galaxyid]
    df2=df2[df2.n_total>0]

    x = df2['z'].tolist()
    y_image=np.zeros(df2.z.nunique())
    y1= df2[param1].tolist()
    y2 = df2[param2].tolist()
    y3 = df2[param3].tolist()
    y4 = df2[param4].tolist()
    paths = df2['filename'].tolist()

    fig, (ax1,ax0) = plt.subplots(2,1, gridspec_kw={'height_ratios': [7.8, 1]}, sharex=True, figsize=(9,6))
    ax2=ax1.twinx()
    ax3=ax1.twinx()
    ax4=ax1.twinx()
    
    axes=[ax1,ax2,ax3,ax4]
    ax2.spines['right'].set_position(('axes', -0.25))
    ax3.spines['right'].set_position(('axes', -0.45))
    ax4.spines['right'].set_position(('axes', -0.65))
    axes[-1].set_frame_on(True)
    axes[-1].patch.set_visible(False)

    ax1.plot(x, y1,  'r', label=param1)
    ax1.yaxis.label.set_color('red')
    ax1.tick_params(axis='y', colors='red')
    ax1.set_ylabel(param1)

    ax2.plot(x, y2,  'b--', label=param2)
    ax2.yaxis.label.set_color('b')
    ax2.tick_params(axis='y', colors='b')
    ax2.set_ylabel(param2, labelpad=-40)

    ax3.plot(x, y3,  'g--', label=param3)
    ax3.yaxis.label.set_color('g')
    ax3.tick_params(axis='y', colors='g')
    ax3.set_ylabel(param3, labelpad=-45)

    ax4.plot(x, y4,  'y', label=param4)
    ax4.yaxis.label.set_color('y')
    ax4.tick_params(axis='y', colors='y')
    ax4.set_ylabel(param4, labelpad=-50) 

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    
    for x0, y0, path in zip(x, y_image,paths):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        ax0.add_artist(ab)
    ax0.yaxis.set_visible(False) 
    ax0.set_ylim(-0.01,0.01) 
    ax0.set_xlabel('z')
    #ax.plot(x, y, ax)
    plt.subplots_adjust(left=0.4,hspace=0)
    plt.title('Pictorial Evolution of Galaxy'+str(galaxyid))
    plt.draw()
    ax1.legend(lines, labels, bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=4)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/PictorialEvolutionGalaxy'+str(galaxyid)+'.png')
    plt.show()

def specificgalplotratesofvariabless(df, galaxyid):
    df=df[df.ProjGalaxyID==galaxyid]
    fig, (ax1,ax2,ax3) = plt.subplots(3,1, sharex=True, figsize=(8,6))
    ax0=ax1.twiny()
    ax0.set_xlabel('Lookback time (Gyr)')
    ax0.set_xticks(df.lbt)
    ax0.set_xlim(df.lbt.max(), df.lbt.min())
    ax1.plot(df.z,df.SFR, 'r' ,label='SFR')
    ax1.plot(df.z, df.dSIMdt,'purple', label='dSIMdt' )
    ax1b=ax1.twinx()
    ax1b.plot(df.z, df.dSFRdz, 'r--',label='dSFRdz')
    ax1b.set_ylabel('$\dfrac{dSFR}{dt}$')
    ax1.set_ylabel('$M_{\odot}yr^{-1}$')
    ax2.plot(df.z,df.BHAccretionrate,'brown', label='BHAccretionRate')
    ax2b=ax2.twinx()
    ax2b.plot(df.z,df.dBHmassdt, 'y', label='dBHmassdt')
    ax2b.set_ylabel('$\dfrac{dBHmass}{dt}$')
    ax2.set_ylabel('$M_{\odot}yr^{-1}$')
    ax3.plot(df.z,df.DiscToTotal, 'b', label='DiscToTotal')
    ax3.plot(df.z, df.dD2Tdz, 'b--',label='dD2Tdz')
    ax3.plot(df.z,df.n_total, 'g', label='n_total' )
    ax3b=ax3.twinx()
    ax3b.plot(df.z,df.dn_totaldz,'g--', label='dn_totaldz')
    ax3b.set_ylabel('$\dfrac{dn_total}{dt}$')
    ax3.set_xlabel('z')
    fig.legend(bbox_to_anchor=(1.01,0.5),loc='center right')
    ax0.set_title('Rate of Change of Variables for galaxy '+str(galaxyid))
    #plt.tight_layout()
    plt.subplots_adjust(hspace=0, right=0.6)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/RatesofVariables'+str(galaxyid)+'.png')
    plt.show()

def specificgalplotmasses(df, galaxyid):
    df=df[df.ProjGalaxyID==galaxyid]
    plt.plot(df.z,df.Starmass, label='Starmass')
    plt.plot(df.z, df.BHmass, label='BHmass')
    plt.plot(df.z, df.Gasmass, label='Gasmass')
    plt.plot(df.z, df.StellarInitialMass, label='StellarInitialMass')
    plt.xlabel('z')
    plt.yscale('log')
    plt.ylabel('$M_{\odot}$')
    plt.legend()
    plt.title('Mass Comparisons for galaxy '+str(galaxyid))
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Massesof'+str(galaxyid)+'.png')
    plt.show()

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, leaf_vs_root_factor = 0.5):

    '''
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.  

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).
    
    There are two basic approaches we think of to allocate the horizontal 
    location of a node.  
    
    - Top down: we allocate horizontal space to a node.  Then its ``k`` 
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a 
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.  
      
    We use use both of these approaches simultaneously with ``leaf_vs_root_factor`` 
    determining how much of the horizontal space is based on the bottom up 
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.   
    
    
    :Arguments: 
    
    **G** the graph (must be a tree)

    **root** the root node of the tree 
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be 
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root
    
    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, leftmost, width, leafdx = 0.2, vert_gap = 0.2, vert_loc = 0, 
                    xcenter = 0.5, rootpos = None, 
                    leafpos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if rootpos is None:
            rootpos = {root:(xcenter,vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            rootdx = width/len(children)
            nextx = xcenter - width/2 - rootdx/2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos(G,child, leftmost+leaf_count*leafdx, 
                                    width=rootdx, leafdx=leafdx,
                                    vert_gap = vert_gap, vert_loc = vert_loc-vert_gap, 
                                    xcenter=nextx, rootpos=rootpos, leafpos=leafpos, parent = root)
                leaf_count += newleaves

            leftmostchild = min((x for x,y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x,y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild+rightmostchild)/2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root]  = (leftmost, vert_loc)
        #pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
    #print(leaf_count)
        return rootpos, leafpos, leaf_count

    xcenter = width/2.
    if isinstance(G, nx.DiGraph):
        leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node)==0])
    elif isinstance(G, nx.Graph):
        leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node)==1 and node != root])
    rootpos, leafpos, leaf_count = _hierarchy_pos(G, root, 0, width, 
                                                    leafdx=width*1./leafcount, 
                                                    vert_gap=vert_gap, 
                                                    vert_loc = vert_loc, 
                                                    xcenter = xcenter)
    pos = {}
    for node in rootpos:
        pos[node] = (leaf_vs_root_factor*leafpos[node][0] + (1-leaf_vs_root_factor)*rootpos[node][0], leafpos[node][1]) 
        #pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x,y in pos.values())
    for node in pos:
        pos[node]= (pos[node][0]*width/xmax, pos[node][1])
    return pos

def plotmergertree(df, galaxyid, colourparam):
    df2=df[df.ProjGalaxyID==galaxyid]
    df2=df2.set_index('DescID')
    
    fig, ax=plt.subplots()
    G=nx.from_pandas_edgelist(df=df2, source='DescID', target='DescGalaxyID')
    G.add_nodes_from(nodes_for_adding=df2.DescID.tolist())
    df2=df2.reindex(G.nodes())
    tree=nx.bfs_tree(G,galaxyid)
    print(G)
   
    
    pos=hierarchy_pos(tree,galaxyid)
    print(pos)
    nx.draw_networkx(G, pos=pos, with_labels=True, font_size=9,node_color=df2[colourparam], cmap=plt.cm.plasma, vmin=df2[colourparam].min(), vmax=df2[colourparam].max(), ax=ax)
    sm=plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=df2[colourparam].min(), vmax=df2[colourparam].max()))
    sm.set_array([])
    
    ax.tick_params(left=True, labelleft=True)
    locs, labels = plt.yticks()
    print('locs={}, labels={}'.format(locs,labels))
    labels = np.linspace(df.z.max(), df.z.min(), len(labels))
    labels=np.around(labels,2)
    plt.yticks(locs, labels)
    plt.ylabel('z')
    cbar=plt.colorbar(sm).set_label(colourparam)
    plt.title('Galaxy Merger Tree for galaxy'+str(galaxyid))
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/MergerTreeforGalaxy'+str(galaxyid)+'.png')
    plt.show()

def plotmovinghistogram(df, histparam, binparam):
    #zmin=df.z.min()
    
    z0df=df[df.zrounded==0.]
    df=df[(df.zrounded==0.) | (df.zrounded==0.1) | (df.zrounded==0.2) | (df.zrounded==0.5)]
    z0df=z0df[['ProjGalaxyID', binparam]]
    #z0df['marker_bin']=pd.qcut(z0df[binparam], 6, labels=['vlow','low','medlow','medhigh','high','vhigh'])
    #z0df['marker_bin']=pd.qcut(z0df[binparam], 5, labels=['10','40','60','80','100'])
    #z0df['marker_bin']=pd.qcut(z0df[binparam],10, labels=['10','20','30','40','50','60','70','80','90','100'])
    z0df['marker_bin']=pd.qcut(z0df[binparam],3, labels=['10','90','100'])
    #z0df['marker_bin']=pd.qcut(z0df[binparam], [0.0, 0.05, 0.3,0.7,0.95,1.0], labels=['20','40','60','80','100'])
    df=pd.merge(df, z0df, on=['ProjGalaxyID'], how='left',  suffixes=('','_proj'))

    fig, axs =plt.subplots(4, 2, sharex=True, figsize=(9,6))
    fig.suptitle('Time evolution of historgram of '+histparam+' showing distribution of '+binparam)
    axs[0,0].set_title('0-10th percentile of '+binparam)
    axs[0,1].set_title('90-100th percentile of '+binparam)
    binedgs=np.linspace(df[histparam].min(), df[histparam].max(), 20)
    for i,zi in enumerate([0., 0.1, 0.2, 0.5]):
        #ax[i] = axs[i].twinx()
        zdf=df[df.zrounded==zi]
        lowdf=zdf[zdf.marker_bin=='10']
        highdf=zdf[zdf.marker_bin=='100']
        axs[i,0].hist(zdf[histparam], color="k", alpha=0.4,label='z='+str(zi), histtype='stepfilled', bins=binedgs)
        axs[i,1].hist(zdf[histparam], color="k", alpha=0.4,label='z='+str(zi), histtype='stepfilled', bins=binedgs)
        axs[i,0].hist(lowdf[histparam], color="r", alpha=0.5, histtype='step', bins=binedgs)
        axs[i,1].hist(highdf[histparam], color="b", alpha=0.5, histtype='step', bins=binedgs)
        
        """
        sns.distplot(zdf[histparam],  kde=False, color="k", ax=axs[i], norm_hist=False, label='z='+str(zi))
        #sns.kdeplot(zdf[histparam], ax=ax[i], color="k", label='')
        sns.distplot(lowdf[histparam],  kde=False, color="r", ax=axs[i], norm_hist=False)
        #sns.kdeplot(lowdf[histparam],  color="r", ax=axs[i], label='')
        sns.distplot(highdf[histparam], kde=False,  color="b", ax=axs[i], norm_hist=False)
        #sns.kdeplot(highdf[histparam],  color="b", ax=axs[i], label='')
        """
        axs[i,0].set_xlabel('')
        axs[i,1].set_xlabel('')
        axs[i,0].set_ylabel('')
        axs[i,1].legend()
        

    #sns.distplot(lowdf[histparam],  kde=False, color="r", ax=axs[3], norm_hist=False, label='20th percentile of'+binparam)
    #sns.distplot(highdf[histparam], kde=False,  color="b", ax=axs[3], norm_hist=False, label='80th percentile of'+binparam)
    axs[3,0].set_xlabel(histparam)
    axs[3,1].set_xlabel(histparam)
    plt.subplots_adjust(wspace=0, hspace=0)
    #handles, labels = axs[3].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='lower center')
    #plt.tight_layout()
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Histogramof'+histparam+'highlighted'+binparam+'.png')
    plt.show()

def plotbulgedisctransz(df, maxz, param, thresh, threshstep):
    B2B =[]
    D2D= []
    B2D=[]
    D2B=[]
    BDB=[]
    DBD=[]
    df=df[df.z<maxz]
    nmax=df.ProjGalaxyID.nunique()
    for id in df.ProjGalaxyID.unique():
        tempdf=df[df.ProjGalaxyID==id]
        tempdf=tempdf.sort_values('z').reset_index()
        if tempdf[param].min() > thresh-0.1:
            B2B.append(id)
        elif tempdf[param].max() < thresh+0.1:
            D2D.append(id)
        elif tempdf[param].iloc[tempdf.z.idxmax()]>thresh:
            if tempdf[param].iloc[0] <thresh-(threshstep):
                B2D.append(id)
            else:
                BDB.append(id)
        elif tempdf[param].iloc[tempdf.z.idxmax()]<thresh:
            if tempdf[param].iloc[0] >thresh+(threshstep):
                D2B.append(id)
            else:
                DBD.append(id)
    BDlist=[]
    fig, ax =plt.subplots(2, 6, sharey='row', sharex='row', figsize=(12,6))
    fig.suptitle('Time evolution'+param)
    for id in B2B:
        temp=df[df.ProjGalaxyID==id]
        ax[1,0].plot(temp.z, temp[param], 'k', linewidth=0.2)
    ax[1,0].plot([df.z.min(),df.z.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,0].bar(0,len(B2B), color='b')
    ax[0,0].text(-.1, 1, str(round(100*len(B2B)/nmax, 1)) +'%', fontsize=12, color='white')
    for id in D2D:
        temp=df[df.ProjGalaxyID==id]
        ax[1,1].plot(temp.z, temp[param], 'k', linewidth=0.2)
    ax[1,1].plot([df.z.min(),df.z.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,1].bar(0,len(D2D), color='b')
    ax[0,1].text(-.1, 1, ''+str(round(100*len(D2D)/nmax, 1))+'%', fontsize=12, color='white')
    for id in B2D:
        temp=df[df.ProjGalaxyID==id]
        ax[1,2].plot(temp.z, temp[param], 'k', linewidth=0.2)
    ax[1,2].plot([df.z.min(),df.z.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,2].bar(0,len(B2D), color='b')
    ax[0,2].text(-.1, 1, str(round(100*len(B2D)/nmax, 1))+'%', fontsize=12, color='white')
    for id in D2B:
        temp=df[df.ProjGalaxyID==id]
        ax[1,3].plot(temp.z, temp[param], 'k', linewidth=0.2)
    ax[1,3].plot([df.z.min(),df.z.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,3].bar(0,len(D2B), color='b')
    ax[0,3].text(-.1, 1, str(round(100*len(D2B)/nmax, 1))+'%', fontsize=12, color='white')
    for id in BDB:
        temp=df[df.ProjGalaxyID==id]
        ax[1,4].plot(temp.z, temp[param], 'k', linewidth=0.2)
    ax[1,4].plot([df.z.min(),df.z.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,4].bar(0,len(BDB), color='b')
    ax[0,4].text(-.1, 1, str(round(100*len(BDB)/nmax, 1)) +'%', fontsize=12, color='white')
    for id in DBD:
        temp=df[df.ProjGalaxyID==id]
        ax[1,5].plot(temp.z, temp[param], 'k', linewidth=0.2)
    ax[1,5].plot([df.z.min(),df.z.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,5].bar(0,len(DBD), color='b')
    ax[0,5].text(-.1, 1, str(round(100*len(DBD)/nmax, 1))+'%', fontsize=12, color='white')

    ax[0,0].set_title('B'),ax[0,1].set_title('D'),ax[0,2].set_title('BD'),ax[0,3].set_title('DB'),ax[0,4].set_title('BDB'),ax[0,5].set_title('DBD')
    ax[1,2].set_xlabel('z')
    ax[1,0].set_ylabel(param)
    ax[0,0].set_ylabel('count')

    #ax[0,1].xticks(locs, labels),ax[0,2].xticks(locs, labels),ax[0,3].xticks(locs, labels),ax[0,4].xticks(locs, labels), ax[0,5].xticks(locs, labels)
    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Evolution of'+str(param)+'thresh'+str(thresh)+'.png')
    plt.show()

def plotbulgedisctranscolour(df, maxz, param, colorparam, thresh, threshstep):
    B2B =[]
    D2D= []
    B2D=[]
    D2B=[]
    BDB=[]
    DBD=[]
    df=df[df.z<maxz]
    nmax=df.ProjGalaxyID.nunique()
    for id in df.ProjGalaxyID.unique():
        tempdf=df[df.ProjGalaxyID==id]
        tempdf=tempdf.sort_values('lbt').reset_index()
        if tempdf[param].min() > thresh-threshstep:
            B2B.append(id)
        elif tempdf[param].max() < thresh+threshstep:
            D2D.append(id)
        elif tempdf[param].iloc[0]>thresh:
            if tempdf[param].iloc[tempdf.lbt.idxmax()] <thresh-(threshstep):
                B2D.append(id)
            else:
                BDB.append(id)
        elif tempdf[param].iloc[0]<thresh:
            if tempdf[param].iloc[tempdf.lbt.idxmax()] >thresh+(threshstep):
                D2B.append(id)
            else:
                DBD.append(id)
    
    fig, ax =plt.subplots(2, 6, sharey='row', sharex='row', figsize=(12,6))
    fig.suptitle('Time evolution'+param)

    #Cmap=plt.get_cmap('RdBu')
    Cmap=mcol.LinearSegmentedColormap.from_list("cmop", ['tomato','cornflowerblue'])
    Norm=plt.Normalize(df[colorparam].min(),df[colorparam].max())

    for id in B2B:
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        y=temp[param].values
        t=temp[colorparam].values
        points=np.array([x,y]).T.reshape(-1,1,2)
        segments=np.concatenate([points[:-1], points[1:]], axis=1)
        lc=LineCollection(segments, cmap=Cmap, norm=Norm)
        lc.set_array(t)
        lc.set_linewidth(0.5)
        ax[1,0].add_collection(lc)
    #fig.colorbar(line, ax=ax[1,0])
        #ax[1,0].plot(temp.lbt, temp[param], 'k', linewidth=0.2)
    ax[1,0].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,0].bar(0,len(B2B), color='purple')
    ax[0,0].text(-.1, 1, str(round(100*len(B2B)/nmax, 1)) +'%', fontsize=12, color='white')
    for id in D2D:
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        y=temp[param].values
        t=temp[colorparam].values
        points=np.array([x,y]).T.reshape(-1,1,2)
        segments=np.concatenate([points[:-1], points[1:]], axis=1)
        lc=LineCollection(segments, cmap=Cmap, norm=Norm)
        lc.set_array(t)
        lc.set_linewidth(0.5)
        ax[1,1].add_collection(lc)
    ax[1,1].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,1].bar(0,len(D2D), color='purple')
    ax[0,1].text(-.1, 1, ''+str(round(100*len(D2D)/nmax, 1))+'%', fontsize=12, color='white')
    for id in B2D:
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        y=temp[param].values
        t=temp[colorparam].values
        points=np.array([x,y]).T.reshape(-1,1,2)
        segments=np.concatenate([points[:-1], points[1:]], axis=1)
        lc=LineCollection(segments, cmap=Cmap, norm=Norm)
        lc.set_array(t)
        lc.set_linewidth(0.5)
        ax[1,2].add_collection(lc)
    ax[1,2].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,2].bar(0,len(B2D), color='purple')
    ax[0,2].text(-.1, 1, str(round(100*len(B2D)/nmax, 1))+'%', fontsize=12, color='white')
    for id in D2B:
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        y=temp[param].values
        t=temp[colorparam].values
        points=np.array([x,y]).T.reshape(-1,1,2)
        segments=np.concatenate([points[:-1], points[1:]], axis=1)
        lc=LineCollection(segments, cmap=Cmap, norm=Norm)
        lc.set_array(t)
        lc.set_linewidth(0.5)
        ax[1,3].add_collection(lc)
    ax[1,3].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,3].bar(0,len(D2B), color='purple')
    ax[0,3].text(-.1, 1, str(round(100*len(D2B)/nmax, 1))+'%', fontsize=12, color='white')
    for id in BDB:
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        y=temp[param].values
        t=temp[colorparam].values
        points=np.array([x,y]).T.reshape(-1,1,2)
        segments=np.concatenate([points[:-1], points[1:]], axis=1)
        lc=LineCollection(segments, cmap=Cmap, norm=Norm)
        lc.set_array(t)
        lc.set_linewidth(0.5)
        ax[1,4].add_collection(lc)
    ax[1,4].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,4].bar(0,len(BDB), color='purple')
    ax[0,4].text(-.1, 1, str(round(100*len(BDB)/nmax, 1)) +'%', fontsize=12, color='white')
    for id in DBD:
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        y=temp[param].values
        t=temp[colorparam].values
        points=np.array([x,y]).T.reshape(-1,1,2)
        segments=np.concatenate([points[:-1], points[1:]], axis=1)
        lc=LineCollection(segments, cmap=Cmap, norm=Norm)
        lc.set_array(t)
        lc.set_linewidth(0.5)
        ax[1,5].add_collection(lc)
    ax[1,5].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,5].bar(0,len(DBD), color='purple')
    ax[0,5].text(-.1, 1, str(round(100*len(DBD)/nmax, 1))+'%', fontsize=12, color='white')

    ax[0,0].set_title('B'),ax[0,1].set_title('D'),ax[0,2].set_title('BD'),ax[0,3].set_title('DB'),ax[0,4].set_title('BDB'),ax[0,5].set_title('DBD')
    ax[1,2].set_xlabel('look back time (Gyr)')
    ax[1,0].set_ylabel(param)
    ax[0,0].set_ylabel('count')
    ax[1,0].set_xlim(df.lbt.min(), df.lbt.max())
    ax[1,0].set_ylim(df[param].min(), df[param].max())
    locs = ax[1,0].get_xticks()
    labels = [-item for item in locs]
    #ax[1,0].set_xticklabels(labels)
    #ax[0,1].xticks(locs, labels),ax[0,2].xticks(locs, labels),ax[0,3].xticks(locs, labels),ax[0,4].xticks(locs, labels), ax[0,5].xticks(locs, labels)
    plt.subplots_adjust(right=0.8, wspace=0.1, hspace=0)
    cbar_ax=fig.add_axes([0.85,0.15,0.05,0.8])
    sm=plt.cm.ScalarMappable(cmap=Cmap, norm=Norm)
    sm.set_array([])
    cbar=plt.colorbar(sm, cax=cbar_ax).set_label(colorparam)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Evolution of'+str(param)+'thresh'+str(thresh)+'zmax'+str(maxz)+'color.png')
    plt.show()

def plotbulgedisctranscolourmerger(df, maxz, param, colorparam, merger, thresh, threshstep, merge=False):
    B2B =[]
    D2D= []
    B2D=[]
    D2B=[]
    BDB=[]
    DBD=[]
    df=df[df.z<maxz]
    nmax=df.ProjGalaxyID.nunique()
    for id in df.ProjGalaxyID.unique():
        tempdf=df[df.ProjGalaxyID==id]
        tempdf=tempdf.sort_values('lbt').reset_index()
        if tempdf[param].min() > thresh-threshstep:
            B2B.append(id)
        elif tempdf[param].max() < thresh:
            D2D.append(id)
        elif tempdf[param].iloc[0]>thresh:
            if tempdf[param].iloc[tempdf.lbt.idxmax()] <thresh-(threshstep):
                B2D.append(id)
            else:
                BDB.append(id)
        elif tempdf[param].iloc[0]<thresh:
            if tempdf[param].iloc[tempdf.lbt.idxmax()] >thresh+(threshstep):
                D2B.append(id)
            else:
                DBD.append(id)
    
    fig, ax =plt.subplots(2, 6, sharey='row', sharex='row', figsize=(12,6))
    fig.suptitle('Time evolution'+param)

    #Cmap=plt.get_cmap('RdBu')
    Cmap=mcol.LinearSegmentedColormap.from_list("cmop", ['tomato','cornflowerblue'])
    Norm=plt.Normalize(df[colorparam].min(),df[colorparam].max())

    for id in B2B:
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        y=temp[param].values
        t=temp[colorparam].values
        points=np.array([x,y]).T.reshape(-1,1,2)
        segments=np.concatenate([points[:-1], points[1:]], axis=1)
        lc=LineCollection(segments, cmap=Cmap, norm=Norm)
        lc.set_array(t)
        lc.set_linewidth(0.5)
        ax[1,0].add_collection(lc)
        dfmergetemp=temp[temp.Starmassmergerfrac>0.]
        ax[1,0].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=10)
    #fig.colorbar(line, ax=ax[1,0])
        #ax[1,0].plot(temp.lbt, temp[param], 'k', linewidth=0.2)
    ax[1,0].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    
    ax[0,0].bar(0,len(B2B), color='purple')
    ax[0,0].text(-.1, 1, str(round(100*len(B2B)/nmax, 1)) +'%', fontsize=12, color='white')
    for id in D2D:
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        y=temp[param].values
        t=temp[colorparam].values
        points=np.array([x,y]).T.reshape(-1,1,2)
        segments=np.concatenate([points[:-1], points[1:]], axis=1)
        lc=LineCollection(segments, cmap=Cmap, norm=Norm)
        lc.set_array(t)
        lc.set_linewidth(0.5)
        ax[1,1].add_collection(lc)
        dfmergetemp=temp[temp.Starmassmergerfrac>0.]
        ax[1,1].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=10)
    ax[1,1].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,1].bar(0,len(D2D), color='purple')
    ax[0,1].text(-.1, 1, ''+str(round(100*len(D2D)/nmax, 1))+'%', fontsize=12, color='white')
    for id in B2D:
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        y=temp[param].values
        t=temp[colorparam].values
        points=np.array([x,y]).T.reshape(-1,1,2)
        segments=np.concatenate([points[:-1], points[1:]], axis=1)
        lc=LineCollection(segments, cmap=Cmap, norm=Norm)
        lc.set_array(t)
        lc.set_linewidth(0.5)
        ax[1,2].add_collection(lc)
        dfmergetemp=temp[temp.Starmassmergerfrac>0.]
        ax[1,2].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=10)
    ax[1,2].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,2].bar(0,len(B2D), color='purple')
    ax[0,2].text(-.1, 1, str(round(100*len(B2D)/nmax, 1))+'%', fontsize=12, color='white')
    for id in D2B:
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        y=temp[param].values
        t=temp[colorparam].values
        points=np.array([x,y]).T.reshape(-1,1,2)
        segments=np.concatenate([points[:-1], points[1:]], axis=1)
        lc=LineCollection(segments, cmap=Cmap, norm=Norm)
        lc.set_array(t)
        lc.set_linewidth(0.5)
        ax[1,3].add_collection(lc)
        dfmergetemp=temp[temp.Starmassmergerfrac>0.]
        ax[1,3].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=10)
    ax[1,3].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,3].bar(0,len(D2B), color='purple')
    ax[0,3].text(-.1, 1, str(round(100*len(D2B)/nmax, 1))+'%', fontsize=12, color='white')
    for id in BDB:
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        y=temp[param].values
        t=temp[colorparam].values
        points=np.array([x,y]).T.reshape(-1,1,2)
        segments=np.concatenate([points[:-1], points[1:]], axis=1)
        lc=LineCollection(segments, cmap=Cmap, norm=Norm)
        lc.set_array(t)
        lc.set_linewidth(0.5)
        ax[1,4].add_collection(lc)
        dfmergetemp=temp[temp.Starmassmergerfrac>0.]
        ax[1,4].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=10)
    ax[1,4].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,4].bar(0,len(BDB), color='purple')
    ax[0,4].text(-.1, 1, str(round(100*len(BDB)/nmax, 1)) +'%', fontsize=12, color='white')
    for id in DBD:
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        y=temp[param].values
        t=temp[colorparam].values
        points=np.array([x,y]).T.reshape(-1,1,2)
        segments=np.concatenate([points[:-1], points[1:]], axis=1)
        lc=LineCollection(segments, cmap=Cmap, norm=Norm)
        lc.set_array(t)
        lc.set_linewidth(0.5)
        ax[1,5].add_collection(lc)
        dfmergetemp=temp[temp.Starmassmergerfrac>0.]
        ax[1,5].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=10)
    ax[1,5].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,5].bar(0,len(DBD), color='purple')
    ax[0,5].text(-.1, 1, str(round(100*len(DBD)/nmax, 1))+'%', fontsize=12, color='white')

    ax[0,0].set_title('B'),ax[0,1].set_title('D'),ax[0,2].set_title('BD'),ax[0,3].set_title('DB'),ax[0,4].set_title('BDB'),ax[0,5].set_title('DBD')
    ax[1,2].set_xlabel('look back time (Gyr)')
    ax[1,0].set_ylabel(param)
    ax[0,0].set_ylabel('count')
    ax[1,0].set_xlim(df.lbt.min(), df.lbt.max())
    ax[1,0].set_ylim(df[param].min(), df[param].max())
    locs = ax[1,0].get_xticks()
    labels = [-item for item in locs]
    #ax[1,0].set_xticklabels(labels)
    #ax[0,1].xticks(locs, labels),ax[0,2].xticks(locs, labels),ax[0,3].xticks(locs, labels),ax[0,4].xticks(locs, labels), ax[0,5].xticks(locs, labels)
    plt.subplots_adjust(right=0.8, wspace=0.1, hspace=0)
    cbar_ax=fig.add_axes([0.85,0.15,0.05,0.8])
    sm=plt.cm.ScalarMappable(cmap=Cmap, norm=Norm)
    sm.set_array([])
    cbar=plt.colorbar(sm, cax=cbar_ax).set_label(colorparam)
    if merge==True:
        plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Evolution of'+str(param)+'thresh'+str(thresh)+'zmax'+str(maxz)+'colorby'+merger+'merge.png')
    else:
        plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Evolution of'+str(param)+'thresh'+str(thresh)+'zmax'+str(maxz)+'colorby'+merger+'.png')
    plt.show()
    plt.close()

def binvalue(df, paramx, paramy, binno):
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
        std=bindf[paramy].std() /2
        low=bindf[paramy].quantile(0.10)
        high=bindf[paramy].quantile(0.90)

        medianvals.append(med)
        stdvals.append(std)
        lowquart.append(low)
        uppquart.append(high)
    return medianvals, binedgs2, lowquart, uppquart, stdvals

def plotmovingquantiles(df, paramx, paramy, binparam):
    df['zrounded']=df.apply(lambda x: np.round(x.z, decimals=1), axis=1)
    #plt.hist(df.zrounded)
    z0df=df[df.zrounded==0.]
    df=df[(df.zrounded==0.) | (df.zrounded==0.1) | (df.zrounded==0.2) | (df.zrounded==0.5)]
    z0df=z0df[['ProjGalaxyID', binparam]]
    #z0df['marker_bin']=pd.qcut(z0df[binparam], 5, labels=['10','40','60','80','100'])
    z0df['marker_bin']=pd.qcut(z0df[binparam], 20, labels=['10','20','30','1','2','3','4','5','6','7','8','9','11','40','50','60','70','80','90','100'])
    df=pd.merge(df, z0df, on=['ProjGalaxyID'], how='left',  suffixes=('','_proj'))
    
    fig, axs =plt.subplots(4, 2, sharex=True, sharey=True, figsize=(9,6))
    fig.suptitle('Time evolution of '+paramx+paramy+' showing distribution of '+binparam)
    axs[0,0].set_title('10th percentile of '+binparam)
    axs[0,1].set_title('90th percentile of '+binparam)
    for i,zi in enumerate([0., 0.1, 0.2, 0.5]):
        #ax[i] = axs[i].twinx()
        zdf=df[df.zrounded==zi]
        medianvals, binedgs, lowquart, highquart, std=binvalue(zdf, paramx, paramy, 20)
        lowdf=zdf[zdf.marker_bin=='10']
        highdf=zdf[zdf.marker_bin=='100']
        axs[i,0].plot(binedgs, medianvals,color="k", label='z='+str(zi))
        axs[i,0].plot(binedgs, lowquart,"k--")
        axs[i,0].plot(binedgs, highquart,"k--")
        axs[i,0].fill_between(binedgs, lowquart, highquart, color='grey', alpha=0.4)
        axs[i,1].plot(binedgs, medianvals,color="k", label='z='+str(zi))
        axs[i,1].plot(binedgs, lowquart,"k--")
        axs[i,1].plot(binedgs, highquart,"k--")
        axs[i,1].fill_between(binedgs, lowquart, highquart, color='grey', alpha=0.4)
        axs[i,0].scatter(lowdf[paramx], lowdf[paramy],color="b", alpha=0.5)
        axs[i,1].scatter(highdf[paramx], highdf[paramy],color="r", alpha=0.5)

        axs[i,0].set_xlabel('')
        axs[i,1].set_xlabel('')
        #axs[i,1].set_yticks([])
        axs[i,0].set_ylabel(paramy)
        axs[i,1].legend()
        
    axs[3,0].set_xlabel(paramx)
    axs[3,1].set_xlabel(paramx)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Plotof'+paramx+paramy+'highlighted'+binparam+'.png')
    plt.show()

def plotmovingquantilesdemo(df, paramx, paramy, binparam):
    df['zrounded']=df.apply(lambda x: np.round(x.z, decimals=1), axis=1)
    z0df=df[df.zrounded==0.]
    df=df[(df.zrounded==0.) | (df.zrounded==0.1) | (df.zrounded==0.2) | (df.zrounded==0.5)]
    z0df=z0df[['ProjGalaxyID', binparam]]
    df['normisedcolor']=df.apply(lambda x: x.logsSFRpermass+np.abs(df.logsSFRpermass.min())/(df.logsSFRpermass.max()+np.abs(df.logsSFRpermass.min())), axis=1)
    fig, axs =plt.subplots(4, 1, sharex=True, sharey='col', figsize=(9,6))
    fig.suptitle('Time evolution of '+paramx+paramy+' showing distribution of sSFR per mass bin')
    Cmap=mcol.LinearSegmentedColormap.from_list("cmop", ['tomato','cornflowerblue'])
    Norm=plt.Normalize(df['logsSFRpermass'].min(),df['logsSFRpermass'].max())
    for i,zi in enumerate([0., 0.1, 0.2, 0.5]):
        #ax[i] = axs[i].twinx()
        zdf=df[df.zrounded==zi]
        medianvals, binedgs, lowquart, highquart, std=binvalue(zdf, paramx, paramy, 20)
        axs[i].plot(binedgs, medianvals,color="k", label='z='+str(zi))
        axs[i].plot(binedgs, lowquart,"k--")
        axs[i].plot(binedgs, highquart,"k--")
        axs[i].fill_between(binedgs, lowquart, highquart, color='grey', alpha=0.4)
        axs[i].scatter(zdf[paramx], zdf[paramy],c=zdf.normisedcolor.values, alpha=0.5, cmap=Cmap)
        axs[i].set_xlabel('')
        axs[i].set_ylabel(paramy)
        axs[i].legend()
        
    axs[3].set_xlabel(paramx)
    plt.subplots_adjust(right=0.8,hspace=0)
    cbar_ax=fig.add_axes([0.85,0.15,0.05,0.7])
    sm=plt.cm.ScalarMappable(cmap=Cmap, norm=Norm)
    sm.set_array([])
    cbar=plt.colorbar(sm, cax=cbar_ax).set_label('logsSFRpermass')
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Plotof'+paramx+paramy+'showingsSFRpermassbin.png')
    plt.show()

def plotmultivariateplot(df):
    names=['DiscToTotal', 'n_total', 'asymm','logsSFR','logmass', 'logBHmass', 'logDMmass']
    df=df[names]
    GTC = pygtc.plotGTC(df, paramNames=names)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/GTCplot.png')
    plt.show()

def categorybarchart(df, cat):
    zlist=[]
    alist=[]
    blist=[]
    dlist=[]
    for i in df.zrounded.unique():
        tempdf=df[df.zrounded==i]
        Anum=len(tempdf[tempdf[cat]=='A'])
        Bnum=len(tempdf[tempdf[cat]=='B'])
        Dnum=len(tempdf[tempdf[cat]=='D'])
        zlist.append(i)
        alist.append(Anum)
        blist.append(Bnum)
        dlist.append(Dnum)
    zarr=np.array(zlist)
    aarr=np.array(alist)
    barr=np.array(blist)
    darr=np.array(dlist)
    print(zlist, alist, blist, dlist)
    width=0.09
    plt.bar(zarr - width/3, aarr, width/3, label='Asymmetric')
    plt.bar(zarr, barr, width/3, label='Bulge')
    plt.bar(zarr + width/3, darr, width/3, label='Disc')
    plt.legend()
    plt.xlabel('z')
    plt.ylabel('no. of galaxies by'+cat)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/barchart'+cat+'.png')
    plt.show()

def calccatfrac2(cat, catfrac, typ, colormin):
    if cat == typ:
        return catfrac
    else:
        return colormin

def plotfrac(df, y, cat, color):
    fig, ax =plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9,5))
    
    dfA=df.copy()
    dfB=df.copy()
    dfD=df.copy()
    colormin=df[color].min()
    dfA['catfrac2']=dfA.apply(lambda x: calccatfrac2(x.categoryn, x.catDMfrac, 'A', colormin), axis=1)
    dfB['catfrac2']=dfB.apply(lambda x: calccatfrac2(x.categoryn, x.catDMfrac, 'B', colormin), axis=1)
    dfD['catfrac2']=dfD.apply(lambda x: calccatfrac2(x.categoryn, x.catDMfrac, 'D', colormin), axis=1)
    ABD=[dfA, dfB, dfD]
    Cmap=plt.cm.viridis
    
    Norm=plt.Normalize(df[color].min(),df[color].max())
    for i, dff in enumerate(ABD):
        dff=dff.drop_duplicates(['z',y])
        data=dff.pivot(y, 'z', 'catfrac2')
        print(data)
        ax[i].imshow(data, aspect='auto', cmap=Cmap, norm=Norm, origin='lower', extent=(df.z.min(), df.z.max(), df[y].min(), df[y].max()))
        ax[i].set_xlabel('z')

    ax[0].set_title('Asymmetrics'), ax[1].set_title('Bulges'), ax[2].set_title('Discs')
    plt.xlim(df.z.min(), df.z.max()+0.1), plt.ylim(df[y].min(),df[y].max() +0.1)
    ax[0].set_ylabel(''+y+' $M_{\odot}$')
    plt.subplots_adjust(right=0.8, wspace=0, hspace=0)
    cbar_ax=fig.add_axes([0.8,0.11,0.05,0.77])
    sm=plt.cm.ScalarMappable(cmap=Cmap, norm=Norm)
    sm.set_array([])
    cbar=plt.colorbar(sm, cax=cbar_ax).set_label('Fraction in'+color+' in each component')
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/evolvingfrac'+y+''+cat+'colouredby'+color+'.png')
    plt.show()

def colourscatter(df,x,y, column_colour, thresh):
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
    n=10
    py,y1=np.histogram(df[y], bins=n)
    y1=y1[:-1]+(y1[1]-y1[0])/2
    f=UnivariateSpline(y1,py,s=n)
    axleft.plot(f(y1), y1)

    px,x1=np.histogram(df[x], bins=n)
    x1=x1[:-1]+(x1[1]-x1[0])/2
    f=UnivariateSpline(x1,px,s=n)
    axtop.plot(x1, f(x1))

    axleft.set_xlabel('PDF')
    axtop.set_ylabel('PDF')
    axleft.set_ylabel(y)
    ax1.set_xlabel(x)

    ax1.scatter(df[x],df[y], c=df[column_colour], cmap=Cmap, norm=Norm, alpha=0.5, s=10)
    lowdf=df[df[column_colour]<thresh -0.1]
    highdf=df[df[column_colour]>thresh +0.1]
    dflist=[df, highdf, lowdf]
    cs=['k', 'r', 'b']
    for i,df in enumerate(dflist):
        medianvals, binedgs, lowquart, uppquart, std=binvalue(df, x, y, 10)
        ax1.errorbar(binedgs, medianvals, color=cs[i], yerr=(std), fmt='', capsize=0.5, elinewidth=0.5)

    plt.subplots_adjust(right=0.8, wspace=0, hspace=0)
    cbar_ax=fig.add_axes([0.85,0.15,0.05,0.8])
    sm=plt.cm.ScalarMappable(cmap=Cmap, norm=Norm)
    sm.set_array([])
    cbar=plt.colorbar(sm, cax=cbar_ax).set_label(column_colour)
    
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/plot'+x+''+y+'colouredby'+column_colour+'.png')
    plt.show()

def plotbulgetodisc(df, sim_name):
    df=df[df.logDMmass>8]
    df=df[df.num>12]
    df['Starmassmergerfrac']=df.apply(lambda x: zerotonancappedz(x.Starmassmergerfrac, x.z), axis=1)
    df['mergercol']=df.apply(lambda x: mergercolor(x.Starmassmergerfrac), axis=1)
    df['stargascol']=df.apply(lambda x: mergercolor2(x.Stargasmergerfrac, 'Stargasmergerfrac'), axis=1)
    df['vHsqrd']=df.apply(lambda x: divide(6.67*10e-11*x.M200, x.R200), axis=1)
    print(df.columns.values)
    mergerdf= mergerinvestigation(df)
    vdf=df.dropna(subset=['vHsqrd'])
    vdf=vdf[vdf.vHsqrd<40]

    
    plotbulgedisctranscolourmerger(df,1.5,'BulgeToTotal','logsSFRpermass','stargascol', 0.5,0.1)
    plotbulgedisctranscolourmerger(mergerdf,1.5,'BulgeToTotal','logsSFRpermass','stargascol', 0.5,0.1, merge=True)
    plotbulgedisctranscolourmerger(df,1.5,'BulgeToTotal','logsSFRpermass','mergercol', 0.5,0.1)
    plotbulgedisctranscolourmerger(mergerdf,1.5,'BulgeToTotal','logsSFRpermass','mergercol', 0.5,0.1, merge=True)
    """
    colourscatter(vdf, 'lbt', 'logmass', 'BulgeToTotal', 0.5)
    colourscatter(vdf, 'lbt', 'logDMmass', 'BulgeToTotal', 0.5)
    colourscatter(vdf, 'lbt', 'vHsqrd', 'BulgeToTotal', 0.5)
    """
    
    
    n2ddf=df[df.n2d>0]
    n2dmergerdf=mergerdf[mergerdf.n2d>0]
    n2dvdf=vdf[vdf.n2d>0]
    plotbulgedisctranscolourmerger(n2ddf,1.5,'n2d','logsSFRpermass','stargascol', 1.4,0.1)
    plotbulgedisctranscolourmerger(n2dmergerdf,1.5,'n2d','logsSFRpermass','stargascol', 1.4,0.1, merge=True)
    plotbulgedisctranscolourmerger(n2ddf,1.5,'n2d','logsSFRpermass','mergercol', 1.4,0.1)
    plotbulgedisctranscolourmerger(n2dmergerdf,1.5,'n2d','logsSFRpermass','mergercol', 1.4,0.1, merge=True)
    """
    colourscatter(n2dvdf, 'lbt', 'logmass', 'n2d', 1.4)
    colourscatter(n2dvdf, 'lbt', 'logDMmass', 'n2d', 1.4)
    colourscatter(n2dvdf, 'lbt', 'vHsqrd', 'n2d', 1.4)

    n_totaldf=df[df.n_total>0]
    n_totalmergerdf=mergerdf[mergerdf.n_total>0]
    n_totalvdf=vdf[vdf.n_total>0]
    plotbulgedisctranscolourmerger(n_totaldf,1.5,'n_total','logsSFRpermass','stargascol', 1.5,0.1)
    plotbulgedisctranscolourmerger(n_totalmergerdf,1.5,'n_total','logsSFRpermass','stargascol', 1.5,0.1, merge=True)
    plotbulgedisctranscolourmerger(n_totaldf,1.5,'n_total','logsSFRpermass','mergercol', 1.5,0.1)
    plotbulgedisctranscolourmerger(n_totalmergerdf,1.5,'n_total','logsSFRpermass','mergercol', 1.5,0.1, merge=True)
    colourscatter(n_totalvdf, 'lbt', 'logmass', 'n_total', 1.5)
    colourscatter(n_totalvdf, 'lbt', 'logDMmass', 'n_total', 1.5)
    colourscatter(n_totalvdf, 'lbt', 'vHsqrd', 'n_total', 1.5)
    """

    exit()

    #threeDplot(df, 'Starmassmergerfrac','z','n2d', 'logmass', 'logsSFR')
    
    plt.hist(df.Stargasmergerfrac)
    plt.show()
    plotmovinghistogram(df, 'n2d', 'Starmassmergerfrac')
    plotmovinghistogram(df, 'logsSFR', 'Starmassmergerfrac')
    plotmovinghistogram(df, 'n2d', 'Stargasmergerfrac')
    exit()
    df=df[df.n2d>0]
    df=df[df.z<3]
    df=df[df.asymm<0.5]
    print(df[['z','Gasmass', 'Gassmassmergerfrac']])
    
    
    plotbulgedisctranscolourmerger(df,1.,'n2d','logsSFRpermass',1.5,0.1)
    plotbulgedisctranscolour(df,1.,'n2d','logsDMmasspermass',1.5,0.1)
    plotbulgedisctranscolour(df,1.,'n2d','logDMEllipticitypermass',1.5,0.1)
    exit()
    
    plotmovingquantiles(df, 'logmass', 'logsSFR', 'n2d')
    plotmovingquantiles(df, 'logmass', 'logBHmass', 'n2d')
    plotmovingquantiles(df, 'logmass', 'logsBHmass', 'n2d')
    plotmovingquantiles(df, 'logmass', 'logDMmass', 'n2d')
    plotmovingquantiles(df, 'logmass', 'logsDMmass', 'n2d')
    #categorybarchart(df, 'categoryn2d')

    
    exit()
    colourscatter(df, 'logmass','SFR',  'n2d')
    colourscatter(df, 'logmass','sSFR',  'n2d')
    colourscatter(df, 'logmass','logSFR',  'n2d')

    exit()

    #plotfrac(df,'roundlogmass2', 'categoryn', 'catDMfrac')

    
    plotbulgedisctranscolour(df,0.6,'n2d','logsSFRpermass',1.5,0.1)
    exit()
    plotmovingquantiles(df, 'logmass', 'logsSFR', 'n2d')
    plotmovingquantiles(df, 'logmass', 'logBHmass', 'n2d')
    plotmovingquantiles(df, 'logmass', 'logsBHmass', 'n2d')
    plotmovingquantiles(df, 'logmass', 'logDMmass', 'n2d')
    plotmovingquantiles(df, 'logmass', 'logsDMmass', 'n2d')
    plotfrac(df,'roundlogmass2', 'categoryn2d', 'catfrac')
    plotfrac(df,'roundsSFR2', 'categoryn2d', 'catsfrfrac')
    plotfrac(df,'roundBHmass', 'categoryn2d', 'catBHfrac')
    plotfrac(df,'roundDMmass2', 'categoryn2d', 'catDMfrac')


    exit()
    
    #df=cleanandtransformdata(df)
    #df=df[df.sSFR>0]
    plotmultivariateplot(df)
    exit()
    df=df[['z','lbt','sSFR','DiscToTotal', 'BulgeToTotal','n_total','logmass', 'logBHmass', 'asymm', 'ProjGalaxyID', 'logsSFR', 'sSFRpermass', 'logsSFRpermass', 'logDMmass']]
    plotbulgedisctranscolour(df,0.6,'asymm','logsSFRpermass',0.25,0.05)
    #plotbulgedisctranscolour(df,0.6,'n_total','logsSFRpermass',1.5,0.1)
    #plotmovingquantiles(df, 'logmass', 'logsSFR', 'n_total')
    #plotmovingquantiles(df, 'logmass', 'logBHmass', 'n_total')
    #plotmovingquantiles(df, 'logmass', 'logDMmass', 'n_total')
    exit()

    
    plotmovingquantiles(df, 'logBHmass', 'logsSFR', 'n_total')
    plotmovingquantiles(df, 'logmass', 'logBHmass', 'n_total')

    plotmovingquantiles(df, 'logmass', 'logsSFR', 'asymm')
    plotmovingquantiles(df, 'logBHmass', 'logsSFR', 'asymm')
    plotmovingquantiles(df, 'logmass', 'logBHmass', 'asymm')
    
    
    plotmovinghistogram(df, 'logsSFR', 'asymm')
    exit()
    


    maxnum=df.num.max()
    print(maxnum)
    maxdf=df[df.num>20]
    ID=maxdf.ProjGalaxyID.unique()
    print(ID)
    
    ID=1459291
    ID=1250867
    ID=1573529
    plotmergertree(df, ID, 'Starmass')

    #specificgalaxyplot(maxdf, ID, 'DiscToTotal', 'n_total', 'logsSFR', 'loggasmass')

    #plotmovinghistogram(df, 'logsSFR', 'asymm')
    
    
    #print(df[['z','dz','lookbacktime','dlbt','StellarInitialMass', 'dSIM', 'dSIMdz', 'dSIMdt', 'SFR']])
    galaxyid=1430974
    specificgalaxyplot(df, galaxyid, 'BulgeToTotal', 'n_total', 'logsSFR', 'loggasmass')
    plotmergertree(df, galaxyid, 'logsSFR')

    plt.plot(df.z, df.BulgeToTotal)
    plt.plot(df.z, df.SFR)
    plt.show()
    #specificgalplotmasses(df, galaxyid)
    #specificgalplotratesofvariabless(df, galaxyid)
    specificgalaxyplot(df, galaxyid, 'BulgeToTotal', 'n_total', 'logsSFR', 'logBHmass')
    evolutionplot(df, 'BulgeToTotal', 'logmass', 'logBHmass')
    #df work
    
    
    evolutionplot(df, 'Starmass', 'Starmass')
    threeDplot(df, 'z','DiscToTotal','logBHmass', 'Starmass', 'logsSFR')
    exit()
    
    stackedhistogram(df, 'n_total','n_disc','n_bulge','n_bulge_exp')
    #subplothistograms(df, 'n_total','n_disc','n_bulge','n_disca','n_bulgea','n_bulge_exp')
    #colorbarplot(df, 'n_total', 'DiscToTotal', 'logmass', 'logsSFR', 'BHmass')
    threeDplot(df, 'dtototal','DiscToTotal','logBHmass', 'Starmass', 'logsSFR')

    exit()
    

    stackedhistogram(df, 'n_total','n_disc','n_bulge','n_bulge_exp')
    plt.close()
    subplothistograms(df, 'n_total','n_disc','n_bulge','n_bulge_exp')
    plt.close()
    
    df=df[df.zrounded==0.]
    df=df[df.logsSFR<0]
    colorbarplot(df, 'n2d','BulgeToTotal', 'logmass', 'logsSFR', 'logBHmass')
    colorbarplot(df, 'n2d','n_total', 'logmass', 'logsSFR', 'logBHmass')
    colorbarplot(df, 'n2d','logBHmass', 'logmass', 'logsSFR', 'logBHmass')
    colorbarplot(df, 'n2d','logsBHmass', 'logmass', 'logsSFR', 'logBHmass')
    colorbarplot(df, 'n2d','logDMmass', 'logmass', 'logsSFR', 'logBHmass')
    colorbarplot(df, 'n2d','logsDMmass', 'logmass', 'logsSFR', 'logBHmass')
    exit()

if __name__ == "__main__":
    sim_names=['RefL0050N0752']
    #query_type=mainbranch or allbranches
    for sim_name in sim_names:
        query_type='mainbranch'
        read_data=True
        if(read_data):
            print('........reading.......')
            df=pd.read_csv('evolvingEAGLEbulgediscmergedf'+sim_name+'total.csv')
        else:
            print('........writing.......')
            
            df=pd.read_csv('evolvingEAGLEimages'+query_type+'df'+sim_name+'.csv')
            df=df[df.z<3]
            df['num']= df.groupby('ProjGalaxyID')['ProjGalaxyID'].transform('count')
            df=df[df.num>17]
            print(df.shape)

            discbulgetemp=[]
            for filename in df['filename']:
                if filename == sim_name:
                    btdradius =btdintensity=star_count=hradius=bradius=disc_intensity=bulge_intensity=btotalintensity=btotalradius =0
                    n_total=n2d= n2d_error=n_disc=n_bulge=n_bulge_exp=n_total_error=n_disc_error=n_bulge_error=n_bulge_exp_error=con=r80=r20=asymm=asymmerror=0
                    discbulgetemp.append([filename, btdradius, btdintensity,n_total,n2d, n2d_error, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disc_error, n_bulge_error, n_bulge_exp_error, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius,con, r80, r20, asymm, asymmerror])

                else:
                    BGRimage=cv2.imread('evolvinggalaxyimagebin'+query_type+''+sim_name+'/'+filename)
                    btdradius, btdintensity, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius =findandlabelbulge(BGRimage, filename, sim_name)
                    #morph_c, morph_asymm, morph_sersic_n, morph_smoothness, morph_sersic_rhalf, morph_xc_asymmetry, morph_yc_asymmetry=runstatmorph(BGRimage)
                    n_total,n2d, n2d_error, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disc_error, n_bulge_error, n_bulge_exp_error,con, r80, r20, asymm, asymmerror=findsersicindex(BGRimage, bradius, hradius)
                    #discbulgetemp.append([filename, btdradius, btdintensity,n_total, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disc_error, n_bulge_error, n_bulge_exp_error, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius,con, r80, r20, asymm, morph_c, morph_asymm, morph_sersic_n, morph_smoothness, morph_sersic_rhalf, morph_xc_asymmetry, morph_yc_asymmetry])
                    discbulgetemp.append([filename, btdradius, btdintensity,n_total,n2d, n2d_error, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disc_error, n_bulge_error, n_bulge_exp_error, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius,con, r80, r20, asymm])

            #discbulgedf=pd.DataFrame(discbulgetemp, columns=['filename', 'btdradius', 'btdintensity','n_total','n_disc','n_bulge','n_bulge_exp', 'n_total_error', 'n_disc_error', 'n_bulge_error', 'n_bulge_exp_error', 'star_count', 'discradius', 'bulgeradius', 'disc_intensity', 'bulge_intensity', 'btotalintensity', 'btotalradius','con', 'r80', 'r20', 'asymm','asymmerror',  'morph_c', 'morph_asymm', 'morph_sersic_n', 'morph_smoothness', 'morph_sersic_rhalf', 'morph_xc_asymmetry', 'morph_yc_asymmetry'])
            discbulgedf=pd.DataFrame(discbulgetemp, columns=['filename', 'btdradius', 'btdintensity','n_total','n2d', 'n2d_error', 'n_disc','n_bulge','n_bulge_exp', 'n_total_error', 'n_disc_error', 'n_bulge_error', 'n_bulge_exp_error', 'star_count', 'discradius', 'bulgeradius', 'disc_intensity', 'bulge_intensity', 'btotalintensity', 'btotalradius','con', 'r80', 'r20', 'asymm','asymmerror'])
            
            df.filename.astype(str)
            discbulgedf.filename.astype(str)
            discbulgedf.to_csv('evolvingEAGLEbulgedisc'+query_type+'df'+sim_name+'TEMP.csv')
            
            #discbulgedf = pd.read_csv('evolvingEAGLEbulgedisc'+query_type+'df'+sim_name+'TEMP.csv')
            df=pd.merge(df, discbulgedf, on=['filename'], how='left').drop_duplicates()
            df.to_csv('evolvingEAGLEbulgedisc'+query_type+'df'+sim_name+'.csv')

        plotbulgetodisc(df, sim_name)


