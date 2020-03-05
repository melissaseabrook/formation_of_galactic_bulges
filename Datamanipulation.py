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
import statmorph
from astropy.modeling import models, fitting

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

def findmergers(dfall):
    dfall=dfall[['z', 'ProjGalaxyID', 'DescID', 'DescGalaxyID', 'Starmass', 'BHmass', 'DMmass', 'Gasmass', 'M200', 'R200']]
    for galaxyid in dfall.ProjGalaxyID.unique():
        df=dfall[dfall.ProjGalaxyID==galaxyid]
        G=nx.from_pandas_edgelist(df=df, source='DescGalaxyID', target='DescID', create_using=nx.DiGraph)
        G.add_nodes_from(nodes_for_adding=df.DescGalaxyID.tolist())
        df2=df[['DescGalaxyID', 'Starmass', 'BHmass', 'DMmass', 'Gasmass']]
        df2=df2.drop_duplicates(subset='DescGalaxyID')
        node_attr=df2.set_index('DescGalaxyID').to_dict('index')
        nx.set_node_attributes(G, node_attr)
        for mass_type in ['BHmass', 'DMmass']:
            merger=[]
            for node1 in G.nodes:
                pred=list(G.predecessors(node1))
                if len(pred)>1:
                    masses=[]
                    for predecessors in G.predecessors(node1):
                        masses.append(G.nodes[predecessors][mass_type])
                    merg=divide(sorted(masses, reverse=True)[1],max(masses))
                else:
                    merg=np.nan
                merger.append(merg)
            merg_dict={n:m for n,m in zip(list(G.nodes()), merger)}
            nx.set_node_attributes(G, merg_dict, mass_type+'mergerfrac')
            print('set node'+mass_type)

        Starmerger=[]
        Gasmerger=[]
        Stargas=[]
        for node1 in G.nodes:
            pred=list(G.predecessors(node1))
            if len(pred)>1:
                Gasmasses=[]
                Starmasses=[]
                for predecessors in G.predecessors(node1):
                    Starmasses.append(G.nodes[predecessors]['Starmass'])
                    Gasmasses.append(G.nodes[predecessors]['Gasmass'])
                starmerg=divide(sorted(Starmasses, reverse=True)[1],max(Starmasses))
                gasmerg=divide(sorted(Gasmasses, reverse=True)[1],max(Gasmasses))
                stargas=divide(np.sum(Gasmasses), np.sum(Starmasses))
            else:
                starmerg=np.nan
                gasmerg=np.nan
                stargas=np.nan
            Starmerger.append(starmerg)
            Gasmerger.append(gasmerg)
            Stargas.append(stargas)

        merg_dict={n:m for n,m in zip(list(G.nodes()), Starmerger)}
        nx.set_node_attributes(G, merg_dict, 'Starmassmergerfrac')
        print('set node starmass')

        merg_dict={n:m for n,m in zip(list(G.nodes()), Gasmerger)}
        nx.set_node_attributes(G, merg_dict, 'Gasmassmergerfrac')
        print('set node gas mass')

        merg_dict={n:m for n,m in zip(list(G.nodes()), Stargas)}
        nx.set_node_attributes(G, merg_dict, 'Stargasmergerfrac')

        print('set node gas star')
        df.append(df2, ignore_index=True)
        df3=pd.DataFrame.from_dict(G.nodes(), orient='index')
        df3['DescGalaxyID']=df3.index
        df3=df3[['DescGalaxyID', 'Starmassmergerfrac',  'BHmassmergerfrac',  'DMmassmergerfrac',  'Gasmassmergerfrac', 'Stargasmergerfrac']]
        dfall=pd.merge(dfall, df3, on=['DescGalaxyID'], how='left', inplace='True')

    #dftot['DescGalaxyID']=dftot.index
    dfall=dfall[['DescGalaxyID', 'Starmassmergerfrac',  'BHmassmergerfrac',  'DMmassmergerfrac',  'Gasmassmergerfrac', 'Stargasmergerfrac', 'M200', 'R200']]
    print(dfall)
    return dfall

def mergers(df):
    df[['Starmassmergerfrac',  'BHmassmergerfrac',  'DMmassmergerfrac',  'Gasmassmergerfrac', 'Stargasmergerfrac']]=df.apply(lambda x: find(x.ProjGalaxyID, x.DescGalaxyID, df), axis=1)
    return df

def find(galaxyid, node1, df):  
    temp=df[df.ProjGalaxyID==galaxyid]
    G=nx.from_pandas_edgelist(df=temp, source='DescGalaxyID', target='DescID', create_using=nx.DiGraph)
    G.add_nodes_from(nodes_for_adding=temp.DescGalaxyID.tolist())
    temp=temp.drop_duplicates(subset='DescGalaxyID')
    temp=temp[['DescGalaxyID', 'Starmass', 'BHmass', 'DMmass', 'Gasmass']]
    temp=temp.drop_duplicates(subset='DescGalaxyID')
    node_attr=temp.set_index('DescGalaxyID').to_dict('index')
    nx.set_node_attributes(G, node_attr)
    pred=list(G.predecessors(node1))

    if len(pred)>1:
        Gasmasses=[]
        Starmasses=[]
        BHmasses=[]
        DMmasses=[]
        for predecessor in G.predecessors(node1):
            Stardeep=[]
            Gasdeep=[]
            BHdeep=[]
            DMdeep=[]
            for nextneighbour in G.predecessors(predecessor):
                Stardeep.append(G.nodes[nextneighbour]['Starmass'])
                Gasdeep.append(G.nodes[nextneighbour]['Gasmass'])
                BHdeep.append(G.nodes[nextneighbour]['BHmass'])
                DMdeep.append(G.nodes[nextneighbour]['DMmass'])

            Stardeep.append(G.nodes[predecessor]['Starmass'])
            Gasdeep.append(G.nodes[predecessor]['Gasmass'])
            BHdeep.append(G.nodes[predecessor]['BHmass'])
            DMdeep.append(G.nodes[predecessor]['DMmass'])

            Starmasses.append(max(Stardeep))
            Gasmasses.append(max(Gasdeep))
            BHmasses.append(max(BHdeep))
            DMmasses.append(max(DMdeep))
            
        starmerg=divide(sorted(Starmasses, reverse=True)[1],max(Starmasses))
        gasmerg=divide(sorted(Gasmasses, reverse=True)[1],max(Gasmasses))
        BHmerg=divide(sorted(BHmasses, reverse=True)[1],max(BHmasses))
        DMmerg=divide(sorted(DMmasses, reverse=True)[1],max(DMmasses))
        stargas=divide(np.sum(Gasmasses), np.sum(Starmasses))
    else:
        starmerg=np.nan
        gasmerg=np.nan
        BHmerg=np.nan
        DMmerg=np.nan
        stargas=np.nan
    print(starmerg)
    return pd.Series([starmerg, gasmerg, BHmerg, DMmerg, stargas])

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
    df['sDMmass']=df.apply(lambda x: divide(x.DMmass,x.Starmass), axis=1)
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

def plotbulgetodisc(df, sim_name):
    print(df.columns.values)


if __name__ == "__main__":
    sim_names=['RefL0050N0752']
    for sim_name in sim_names:
        read_all_data =False
        read_main_branch_data=True
        read_all_branch_data=True
        if(read_all_data):
            print('........reading all data.......')
            df=pd.read_csv('evolvingEAGLEbulgediscmergedf'+sim_name+'.csv')
            
        else:
            if(read_main_branch_data):
                print('........reading mainbranch data.......')
                bulgedf=pd.read_csv('evolvingEAGLEbulgediscmainbranchdf'+sim_name+'.csv')
                bulgedf=cleanandtransformdata(bulgedf)
            else:
                print('........writing.......')
                df=pd.read_csv('evolvingEAGLEimagesmainbranchdf'+sim_name+'.csv')
                df['num']= df.groupby('ProjGalaxyID')['ProjGalaxyID'].transform('count')
                df=df[df.num>17]
                #df=df[df.num>5]
                print(df.shape)

                discbulgetemp=[]
                for filename in df['filename']:
                    if filename == sim_name:
                        btdradius =btdintensity=star_count=hradius=bradius=disc_intensity=bulge_intensity=btotalintensity=btotalradius =0
                        n_total=n2d= n2d_error=n_disc=n_bulge=n_bulge_exp=n_total_error=n_disc_error=n_bulge_error=n_bulge_exp_error=con=r80=r20=asymm=asymmerror=0
                        discbulgetemp.append([filename, btdradius, btdintensity,n_total,n2d, n2d_error, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disc_error, n_bulge_error, n_bulge_exp_error, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius,con, r80, r20, asymm, asymmerror])

                    else:
                        BGRimage=cv2.imread('evolvinggalaxyimagebinmainbranch'+sim_name+'/'+filename)
                        btdradius, btdintensity, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius =findandlabelbulge(BGRimage, filename, sim_name)
                        #morph_c, morph_asymm, morph_sersic_n, morph_smoothness, morph_sersic_rhalf, morph_xc_asymmetry, morph_yc_asymmetry=runstatmorph(BGRimage)
                        n_total,n2d, n2d_error, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disc_error, n_bulge_error, n_bulge_exp_error,con, r80, r20, asymm, asymmerror=findsersicindex(BGRimage, bradius, hradius)
                        #discbulgetemp.append([filename, btdradius, btdintensity,n_total, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disc_error, n_bulge_error, n_bulge_exp_error, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius,con, r80, r20, asymm, morph_c, morph_asymm, morph_sersic_n, morph_smoothness, morph_sersic_rhalf, morph_xc_asymmetry, morph_yc_asymmetry])
                        discbulgetemp.append([filename, btdradius, btdintensity,n_total,n2d, n2d_error, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disc_error, n_bulge_error, n_bulge_exp_error, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius,con, r80, r20, asymm])

                #discbulgedf=pd.DataFrame(discbulgetemp, columns=['filename', 'btdradius', 'btdintensity','n_total','n_disc','n_bulge','n_bulge_exp', 'n_total_error', 'n_disc_error', 'n_bulge_error', 'n_bulge_exp_error', 'star_count', 'discradius', 'bulgeradius', 'disc_intensity', 'bulge_intensity', 'btotalintensity', 'btotalradius','con', 'r80', 'r20', 'asymm','asymmerror',  'morph_c', 'morph_asymm', 'morph_sersic_n', 'morph_smoothness', 'morph_sersic_rhalf', 'morph_xc_asymmetry', 'morph_yc_asymmetry'])
                discbulgedf=pd.DataFrame(discbulgetemp, columns=['filename', 'btdradius', 'btdintensity','n_total','n2d', 'n2d_error', 'n_disc','n_bulge','n_bulge_exp', 'n_total_error', 'n_disc_error', 'n_bulge_error', 'n_bulge_exp_error', 'star_count', 'discradius', 'bulgeradius', 'disc_intensity', 'bulge_intensity', 'btotalintensity', 'btotalradius','con', 'r80', 'r20', 'asymm','asymmerror'])
                
                df.filename.astype(str)
                discbulgedf.filename.astype(str)
               
                discbulgedf.to_csv('evolvingEAGLEbulgediscmainbranchdf'+sim_name+'TEMP.csv')
                
                #discbulgedf = pd.read_csv('evolvingEAGLEbulgedisc'+query_type+'df'+sim_name+'TEMP.csv')
                bulgedf=pd.merge(df, discbulgedf, on=['filename'], how='left').drop_duplicates()
                bulgedf=cleanandtransformdata(bulgedf)
                bulgedf.to_csv('evolvingEAGLEbulgediscmainbranchdf'+sim_name+'.csv')

            if(read_all_branch_data):
                print('........reading merger data.......')
                mergedf=pd.read_csv('evolvingEAGLEmergerdf'+sim_name+'.csv')
                mergedf=mergedf[['DescGalaxyID', 'Starmassmergerfrac',  'BHmassmergerfrac',  'DMmassmergerfrac',  'Gasmassmergerfrac', 'Stargasmergerfrac']]
            else:
                print('........writing merger data.......')
                df=pd.read_csv('evolvingEAGLEimagesallbranchesdf'+sim_name+'.csv')
                print(df)
                df=df[df.Starmass>1e5]
                mergedf=mergers(df)
                #mergedf=pd.merge(df, mergerdf, on=['DescGalaxyID'], how='left').drop_duplicates()
                mergedf.to_csv('evolvingEAGLEmergerdf'+sim_name+'.csv')
                
            df2= pd.merge(bulgedf, mergedf, on=['DescGalaxyID'], how='left').drop_duplicates()
            dfAP=df=pd.read_csv('aperturemainbranchdf'+sim_name+'.csv')
            dfAP=dfAP[['DescGalaxyID','Starmass1', 'Gasmass1','SFR1','Starmass3','Gasmass3','SFR3', 'Starmass10', 'Gasmass10','SFR10']]
            df= pd.merge(df2, dfAP, on=['DescGalaxyID'], how='left').drop_duplicates()
            df.to_csv('evolvingEAGLEbulgediscmergedf'+sim_name+'.csv')
            print(df.columns.values)
            print(df[['z', 'Starmassmergerfrac']])
            #df2=pd.read_csv('evolvingEAGLEbulgediscmergedf'+sim_name+'.csv')
            #df3= df.append(df2, ignore_index=True)
            #df3.to_csv('evolvingEAGLEbulgediscmergedf'+sim_name+'total.csv')

        plotbulgetodisc(df, sim_name)


