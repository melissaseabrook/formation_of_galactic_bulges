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
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from astropy.cosmology import Planck13
import pylab
import networkx as nx
import matplotlib as mpl
import statmorph
import photutils
import scipy.ndimage as ndi

#sns.set_style('whitegrid')
def logx(x):
    if x !=0:
        return np.log10(x)
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
    imagecopy=image.copy()
    median=np.median(image)
    std=np.std(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred1 = cv2.GaussianBlur(gray, ksize=(7, 7), sigmaX=3,sigmaY=3)
    thresh1 = cv2.threshold(blurred1, median + 5*std, 255, cv2.THRESH_BINARY)[1]
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
        cv2.circle(imagecopy, (int(bcX), int(bcY)), int(bradius),(0, 0, 255), 1)
        print("bulge radius:{},  bulge centre({},{})".format(bradius, bcX,bcY))
        if numPixels > 20: 
            cv2.putText(imagecopy, "bulge", (x, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)
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
        cv2.circle(imagecopy, (int(hcX), int(hcY)), int(hradius),(255, 0, 0), 1)
        print("disc radius:{}, disc centre({},{})".format(hradius, hcX,hcY))
        if numPixels > 60: 
            cv2.putText(imagecopy, "disc", (x, y - 5),
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

    btdradius=bradius/hradius
    print("disc radius:bulge radius ={}".format(btdradius))
    print("star count ={}".format(count))
    disc_intensity, total_intensity=findlightintensity(image, hradius, (hcX,hcY))
    bulge_intensity, total_intensity=findlightintensity(image, bradius, (bcX,bcY))
    btdintensity= bulge_intensity/disc_intensity
    btotalintensity=bulge_intensity/total_intensity
    btotalradius=bradius/256
    print("disc intensity = {}, bulge intensity ={}, disc:bulge intensity ={}".format(disc_intensity, bulge_intensity, btdintensity))
    cv2.imwrite('evolvinggalaxygraphsbinmainbranch'+sim_name+'/BulgeDiscImages/opencvfindbulge'+imagefile, imagecopy)
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
    resid= np.abs(image_arr-image_arr180)
    asymm=(np.sum(resid))/(np.sum(np.abs(image_arr)))
    return asymm

def findsersicindex(image, bindex, dindex):
    image2=np.average(image, axis=2, weights=[0.2126,0.587,0.114])
    asymm=findassymetry(image2)
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
    return n_total, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disc_error, n_bulge_error, n_bulge_exp_error, con, r80, r20, asymm

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

def cleanandtransformdata(df):
    print(df.shape)
    df.sort_values(['z','ProjGalaxyID'], ascending=[False,True], inplace=True)
    df['lbt']=df.apply(lambda x: -round(Planck13.lookback_time(x.z).value, 1), axis=1)
    df['lbt2']=df.apply(lambda x: round(Planck13.lookback_time(x.z).value, 1), axis=1)

    df['lookbacktime']=df.apply(lambda x: -(Planck13.lookback_time(x.z).value)*(1e9), axis=1)
    df['dlbt']=df.groupby('ProjGalaxyID')['lookbacktime'].diff()
    df['dSFR']=df.groupby('ProjGalaxyID')['SFR'].diff()
    df['dBHmass']=df.groupby('ProjGalaxyID')['BHmass'].diff()
    df['dSIM']=df.groupby('ProjGalaxyID')['StellarInitialMass'].diff()
    df['dD2T']=df.groupby('ProjGalaxyID')['DiscToTotal'].diff()
    df['dn_total']=df.groupby('ProjGalaxyID')['n_total'].diff()
    df['dz']=df.groupby('ProjGalaxyID')['z'].diff()
    df['dz']=df.apply(lambda x: -x.dz, axis=1)
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
    #dftotal work
    print(df.ProjGalaxyID.nunique())
    df['BulgeToTotal']=df.apply(lambda x: (1-x.DiscToTotal), axis=1)
    df['logBHmass']=df.apply(lambda x: logx(x.BHmass), axis=1)
    df['logmass']=df.apply(lambda x: logx(x.Starmass), axis=1)
    df['loggasmass']=df.apply(lambda x: logx(x.Gasmass), axis=1)
    df['sSFR']=df.apply(lambda x: divide(x.SFR,x.Starmass), axis=1)
    df['logSFR']=df.apply(lambda x: logx(x.SFR), axis=1)
    df['logsSFR']=df.apply(lambda x: logx(x.sSFR), axis=1)
    df['dtototal']=df.apply(lambda x: (1-x.btdintensity), axis=1)
    df['dtbradius']=df.apply(lambda x: invertbtd(x.btdradius), axis=1)
    df['dtbintensity']=df.apply(lambda x: invertbtd(x.btdintensity), axis=1)

    
    """
    dftotal=df
    df=df[df.n_total_error<100]
    df['num_images']= df.groupby('ProjGalaxyID')['ProjGalaxyID'].transform('count')
    print(df.shape)
    df=df[df.n_total>0.05]
    df=df.reset_index()
    print(df.shape)
    """

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
    df['marker_bin']=pd.qcut(df.Z, [0,0.15,0.85,1], labels=['low','okay','high'])
    markers={"low":'^', "okay":'o', 'high':'s'}
    sm=plt.cm.ScalarMappable(cmap='autumn', norm=norm)
    sm.set_array([])
    ax=sns.relplot(x=x, y=y, size=column_size, sizes=(10,150), hue=column_colour, palette='autumn', style='marker_bin', markers=markers,data=df)
    ax._legend.remove()
    ax.fig.colorbar(sm).set_label(column_colour)
    plt.subplots_adjust(top=0.9)
    ax.fig.suptitle(''+x+' vs '+column_marker+', coloured by'+column_colour+', sized by'+column_size+', shaped by'+column_marker+'')
    ax.savefig('evolvinggalaxygraphsbinmainbranch'+sim_name+'/'+x+'vs'+y+'.png')
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
    df['zrounded']=df.apply(lambda x: np.round(x.z, decimals=1), axis=1)
    z0df=df[df.zrounded==0.]
    df=df[(df.zrounded==0.) | (df.zrounded==0.2) | (df.zrounded==0.5) | (df.zrounded==1.)]
    z0df=z0df[['ProjGalaxyID', binparam]]
    #z0df['marker_bin']=pd.qcut(z0df[binparam], 6, labels=['vlow','low','medlow','medhigh','high','vhigh'])
    z0df['marker_bin']=pd.qcut(z0df[binparam],10, labels=['10','20','30','40','50','60','70','80','90','100'])
    #z0df['marker_bin']=pd.qcut(z0df[binparam], [0.0, 0.05, 0.3,0.7,0.95,1.0], labels=['20','40','60','80','100'])
    df=pd.merge(df, z0df, on=['ProjGalaxyID'], how='left',  suffixes=('','_proj'))

    fig, axs =plt.subplots(4, 2, sharex=True, figsize=(9,6))
    fig.suptitle('Time evolution of historgram of '+histparam+' showing distribution of '+binparam)
    axs[0,0].set_title('0-10th percentile of '+binparam)
    axs[0,1].set_title('90-100th percentile of '+binparam)
    binedgs=np.linspace(df[histparam].min(), df[histparam].max(), 20)
    for i,zi in enumerate([0., 0.2, 0.5, 1.]):
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

def plotbulgedisctrans(df, maxz, param, thresh, threshstep):
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
            if tempdf[param].iloc[tempdf.lbt.idxmax()] <thresh-(threshstep/2):
                B2D.append(id)
            else:
                BDB.append(id)
        elif tempdf[param].iloc[0]<thresh:
            if tempdf[param].iloc[tempdf.lbt.idxmax()] >thresh+(threshstep/2):
                D2B.append(id)
            else:
                DBD.append(id)
    BDlist=[]
    fig, ax =plt.subplots(2, 6, sharey='row', sharex='row', figsize=(12,6))
    fig.suptitle('Time evolution'+param)
    for id in B2B:
        temp=df[df.ProjGalaxyID==id]
        ax[1,0].plot(temp.lbt, temp[param], 'k', linewidth=0.1)
        ax[1,0].plot([temp.lbt.min(),temp.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,0].bar(0,len(B2B), color='b')
    ax[0,0].text(-.1, 1, str(round(100*len(B2B)/nmax, 1)) +'%', fontsize=12, color='white')
    for id in D2D:
        temp=df[df.ProjGalaxyID==id]
        ax[1,1].plot(temp.lbt, temp[param], 'k', linewidth=0.1)
        ax[1,1].plot([temp.lbt.min(),temp.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,1].bar(0,len(D2D), color='b')
    ax[0,1].text(-.1, 1, ''+str(round(100*len(D2D)/nmax, 1))+'%', fontsize=12, color='white')
    for id in B2D:
        temp=df[df.ProjGalaxyID==id]
        ax[1,2].plot(temp.lbt, temp[param], 'k', linewidth=0.1)
        ax[1,2].plot([temp.lbt.min(),temp.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,2].bar(0,len(B2D), color='b')
    ax[0,2].text(-.1, 1, str(round(100*len(B2D)/nmax, 1))+'%', fontsize=12, color='white')
    for id in D2B:
        temp=df[df.ProjGalaxyID==id]
        ax[1,3].plot(temp.lbt, temp[param], 'k', linewidth=0.1)
        ax[1,3].plot([temp.lbt.min(),temp.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,3].bar(0,len(D2B), color='b')
    ax[0,3].text(-.1, 10, str(round(100*len(D2B)/nmax, 1))+'%', fontsize=12, color='black')
    for id in BDB:
        temp=df[df.ProjGalaxyID==id]
        ax[1,4].plot(temp.lbt, temp[param], 'k', linewidth=0.1)
        ax[1,4].plot([temp.lbt.min(),temp.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,4].bar(0,len(BDB), color='b')
    ax[0,4].text(-.1, 1, str(round(100*len(BDB)/nmax, 1)) +'%', fontsize=12, color='white')
    for id in DBD:
        temp=df[df.ProjGalaxyID==id]
        ax[1,5].plot(temp.lbt, temp[param], 'k', linewidth=0.1)
        ax[1,5].plot([temp.lbt.min(),temp.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,5].bar(0,len(DBD), color='b')
    ax[0,5].text(-.1, 1, str(round(100*len(DBD)/nmax, 1))+'%', fontsize=12, color='white')

    ax[0,0].set_title('B'),ax[0,1].set_title('D'),ax[0,2].set_title('BD'),ax[0,3].set_title('DB'),ax[0,4].set_title('BDB'),ax[0,5].set_title('DBD')
    ax[1,2].set_xlabel('look back time (Gyr)')
    ax[1,0].set_ylabel(param)
    ax[0,0].set_ylabel('count')
    locs = ax[1,0].get_xticks()
    print(locs)

    labels = [-item for item in locs]
    ax[1,0].set_xticklabels(labels)
    #ax[0,1].xticks(locs, labels),ax[0,2].xticks(locs, labels),ax[0,3].xticks(locs, labels),ax[0,4].xticks(locs, labels), ax[0,5].xticks(locs, labels)
    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.savefig('Evolution of'+str(param)+'thresh'+str(thresh)+'.png')
    plt.show()

def binvalue(df, paramx, paramy, binno):
    binedgs=np.linspace(df[paramx].min(), df[paramx].max(), binno)
    binedgs2=np.linspace(df[paramx].min(), df[paramx].max(), binno -1)
    medianvals=[]
    lowquart=[]
    uppquart=[]
    for i in range(0,binno -1):
        bindf=df[df[paramx]>binedgs[i]]
        bindf=bindf[bindf[paramx]<binedgs[i+1]]
        med=bindf[paramy].median()
        low=bindf[paramy].quantile(0.10)
        high=bindf[paramy].quantile(0.90)

        medianvals.append(med)
        lowquart.append(low)
        uppquart.append(high)
    return medianvals, binedgs2, lowquart, uppquart

def plotmovingquantiles(df, paramx, paramy, binparam):
    df['zrounded']=df.apply(lambda x: np.round(x.z, decimals=1), axis=1)
    z0df=df[df.zrounded==0.]
    df=df[(df.zrounded==0.) | (df.zrounded==0.2) | (df.zrounded==0.5) | (df.zrounded==1.)]
    z0df=z0df[['ProjGalaxyID', binparam]]
    #z0df['marker_bin']=pd.qcut(z0df[binparam], 5, labels=['20','40','60','80','100'])
    z0df['marker_bin']=pd.qcut(z0df[binparam], 20, labels=['10','20','30','1','2','3','4','5','6','7','8','9','11','40','50','60','70','80','90','100'])
    df=pd.merge(df, z0df, on=['ProjGalaxyID'], how='left',  suffixes=('','_proj'))
    
    fig, axs =plt.subplots(4, 2, sharex=True, sharey='col', figsize=(9,6))
    fig.suptitle('Time evolution of '+paramx+paramy+' showing distribution of '+binparam)
    axs[0,0].set_title('10th percentile of '+binparam)
    axs[0,1].set_title('90th percentile of '+binparam)
    for i,zi in enumerate([0., 0.2, 0.5, 1.]):
        #ax[i] = axs[i].twinx()
        zdf=df[df.zrounded==zi]
        medianvals, binedgs, lowquart, highquart=binvalue(zdf, paramx, paramy, 20)
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
        axs[i,0].scatter(lowdf[paramx], lowdf[paramy],color="r", alpha=0.5)
        axs[i,1].scatter(highdf[paramx], highdf[paramy],color="b", alpha=0.5)

        axs[i,0].set_xlabel('')
        axs[i,1].set_xlabel('')
        axs[i,1].set_yticks([])
        axs[i,0].set_ylabel(paramy)
        axs[i,1].legend()
        
    axs[3,0].set_xlabel(paramx)
    axs[3,1].set_xlabel(paramx)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Plotof'+paramx+paramy+'highlighted'+binparam+'.png')
    plt.show()


def plotbulgetodisc(df, sim_name):
    cleanandtransformdata(df)
    df=df[df.sSFR>0]
    
    plotmovingquantiles(df, 'logmass', 'logsSFR', 'asymm')
    exit()
    plotbulgedisctrans(df,2,'BulgeToTotal',0.6,0.1 )
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
    
    fig=plt.figure()
    ax0=fig.add_subplot(221)
    ax1=fig.add_subplot(222)
    ax2=fig.add_subplot(223)
    ax3=fig.add_subplot(224)
    markers={"low":'^', "okay":'o', 'high':'s'}
    g=sns.relplot(x='n_disc', y='DiscToTotal', size='mass', sizes=(10,150), hue='SFR', palette='autumn', style='BHBin', markers=markers,data=df, ax=ax0)
    g=sns.relplot(x='n_total', y='DiscToTotal', size='mass', sizes=(10,150), hue='SFR', palette='autumn', style='BHBin', markers=markers,data=df, ax=ax1)
    g=sns.relplot(x='n_bulge', y='DiscToTotal', size='mass', sizes=(10,150), hue='SFR', palette='autumn', style='BHBin', markers=markers,data=df,ax=ax2)
    g=sns.relplot(x='n_bulge_exp', y='DiscToTotal', size='mass', sizes=(10,150), hue='SFR', palette='autumn', style='BHBin', markers=markers,data=df, ax=ax3)
    fig.tight_layout()
    fig.savefig('evolvinggalaxygraphsbinmainbranch'+sim_name+'/DiscToTotalvsn_disc.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    sim_names=['RefL0050N0752']
    #query_type=mainbranch or allbranches
    for sim_name in sim_names:
        query_type='mainbranch'
        read_data=True
        if(read_data):
            print('........reading.......')
            df=pd.read_csv('evolvingEAGLEbulgedisc'+query_type+'df'+sim_name+'.csv')
        else:
            print('........writing.......')
            df=pd.read_csv('evolvingEAGLEimages'+query_type+'df'+sim_name+'.csv')
            discbulgetemp=[]
            for filename in df['filename']:
                if filename == sim_name:
                    btdradius =btdintensity=star_count=hradius=bradius=disc_intensity=bulge_intensity=btotalintensity=btotalradius =0
                    n_total=n_disc=n_bulge=n_bulge_exp=n_total_error=n_disc_error=n_bulge_error=n_bulge_exp_error=con=r80=r20=asymm=0
                    discbulgetemp.append([filename, btdradius, btdintensity,n_total, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disc_error, n_bulge_error, n_bulge_exp_error, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius,con, r80, r20, asymm])

                else:
                    BGRimage=cv2.imread('evolvinggalaxyimagebin'+query_type+''+sim_name+'/'+filename)
                    btdradius, btdintensity, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius =findandlabelbulge(BGRimage, filename, sim_name)
                    morph_c, morph_asymm, morph_sersic_n, morph_smoothness, morph_sersic_rhalf, morph_xc_asymmetry, morph_yc_asymmetry=runstatmorph(BGRimage)
                    n_total, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disc_error, n_bulge_error, n_bulge_exp_error,con, r80, r20, asymm=findsersicindex(BGRimage, bradius, hradius)
                    discbulgetemp.append([filename, btdradius, btdintensity,n_total, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disc_error, n_bulge_error, n_bulge_exp_error, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius,con, r80, r20, asymm, morph_c, morph_asymm, morph_sersic_n, morph_smoothness, morph_sersic_rhalf, morph_xc_asymmetry, morph_yc_asymmetry])
            discbulgedf=pd.DataFrame(discbulgetemp, columns=['filename', 'btdradius', 'btdintensity','n_total','n_disc','n_bulge','n_bulge_exp', 'n_total_error', 'n_disc_error', 'n_bulge_error', 'n_bulge_exp_error', 'star_count', 'discradius', 'bulgeradius', 'disc_intensity', 'bulge_intensity', 'btotalintensity', 'btotalradius','con', 'r80', 'r20', 'asymm',  'morph_c', 'morph_asymm', 'morph_sersic_n', 'morph_smoothness', 'morph_sersic_rhalf', 'morph_xc_asymmetry', 'morph_yc_asymmetry'])
            
            df.filename.astype(str)
            discbulgedf.filename.astype(str)
            print(discbulgedf)
            print(df)
            df=pd.merge(df, discbulgedf, on=['filename'], how='left').drop_duplicates()
            print(df)
            df.to_csv('evolvingEAGLEbulgedisc'+query_type+'df'+sim_name+'.csv')

        plotbulgetodisc(df, sim_name)


