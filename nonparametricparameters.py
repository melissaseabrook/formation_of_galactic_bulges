import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from imutils import contours
from skimage import measure
import imutils
import math
from scipy.optimize import curve_fit
from scipy import optimize
from scipy import stats
import seaborn as sns
import statmorph
import photutils
import scipy.ndimage as ndi

def findcenter(image):
    #finds coords of central bulge
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11,11), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    return maxVal, maxLoc

def SersicProfile(r, I_e, R_e, n):
	b=(1.992*n)-0.3271
	G=(r/R_e)**(1/n)
	return I_e*np.exp((-b*(G-1)))

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

def findeffectiveradius(radialprofile, r, nr, frac):
    totalbrightness=np.sum(radialprofile * 2 * np.pi *nr*r)
    centralbrightness=radialprofile[0]
    cumulativebrightness=np.cumsum(radialprofile * 2 * np.pi *nr*r)
    r_e_unnormalised=((np.abs((totalbrightness*frac) - cumulativebrightness)).argmin())
    r_e=r_e_unnormalised*(30.0/256)
    i_e= radialprofile[r_e_unnormalised]
    return i_e, r_e, centralbrightness, totalbrightness

def findlightintensity(radialpofile, radius):
    radius=int(radius)
    cumulativebrightness=np.sum(radialpofile[0:radius])
    return cumulativebrightness

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

def findconcentration(rad, r, nr):
    i20, r80, cb,tb=findeffectiveradius(rad, r, nr, 0.8)
    i20, r20, cb, tb=findeffectiveradius(rad, r, nr, 0.2)
    con=5*np.log10(r80/r20)
    return con, r80,r20

def findassymetry(image):
    image_arr = np.array(image)
    image_arr90 = np.rot90(image_arr)
    image_arr180 = np.rot90(image_arr90)
    resid= np.abs(image_arr-image_arr180)
    asymm=(np.sum(resid))/(np.sum(np.abs(image_arr)))
    return asymm

def findclumpiness():
    return

def petfunc(pr, rad, nr, r):
    cumulativebrightness80=np.sum(rad[0:int(0.8*pr)] * 2 * np.pi *nr[0:int(0.8*pr)]*r[0:int(0.8*pr)])
    cumulativebrightness125=np.sum(rad[0:int(1.25*pr)] * 2 * np.pi *nr[0:int(1.25*pr)]*r[0:int(1.25*pr)])
    cumulativebrightness100=np.sum(rad[0:int(pr)] * 2 * np.pi *nr[0:int(pr)]*r[0:int(pr)])
    numer=(cumulativebrightness125-cumulativebrightness80)/((np.pi)*((1.25**2)-(0.8**2))*((pr*30/256)**2))
    denom=cumulativebrightness100/(np.pi*((pr*30/256)**2))
    return (numer/denom)-0.2

def findpetrosianradius(rad, nr, r_arr, bhindex):
    petradbisect=optimize.bisect(petfunc, 0,np.sqrt(2)*15, args=(rad,nr, r_arr))
    petradnewton=optimize.newton(petfunc, bhindex, args=(rad,nr, r_arr))
    return petradbisect, petradnewton


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
    morph_c=morph.concentration
    morph_asymm=morph.asymmetry
    morph_sersic_n=morph.sersic_n
    morph_smoothness=morph.smoothness
    morph_sersic_rhalf=morph.sersic_rhalf*30/256
    morph_xc_asymmetry=morph.xc_asymmetry
    morph_yc_asymmetry=morph.yc_asymmetry
    return morph_c, morph_asymm, morph_sersic_n, morph_smoothness, morph_sersic_rhalf, morph_xc_asymmetry, morph_yc_asymmetry

def run_radial_profile(image, imagefile, sim_name):
    image2=np.average(image, axis=2, weights=[0.2126,0.587,0.114])
    #plots radius vs pixel intensity and its log
    maxVal, center = findcenter(image)
    rad, stdbins, r_arr, nr=radial_profile(image,center)
    bindex,hindex, (hcX,hcY), (bcX,bcY) = findbulge(image, imagefile)
    r=r_arr/256*30
    print(rad.shape, r_arr.shape, nr.shape)
    
    con, r80, r20=findconcentration(rad, r, nr)
    bhindex=int((bindex+hindex)/2)
    #petradbisect, petradnewton=findpetrosianradius(rad, nr, r, bhindex)
    max_r=np.sqrt(2)*15
    nr=nr[1:int(hindex)]
    r=r[1:int(hindex)]
    rad=rad[1:int(hindex)]
    stdbins=stdbins[1:int(hindex)]
    i_e, r_e, centralbrightness, totalbrightness= findeffectiveradius(rad, r, nr, 0.5) 
    b1=bindex*30/256
    h1=hindex*30/256
    bindex=int(bindex)
    hindex=int(hindex)
    bhindex=int((bindex+hindex)/2)
    
    asymm=findassymetry(image2)
    
    """
    median=np.median(image2)
    std=np.std(image2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred2 = cv2.GaussianBlur(gray, ksize=(15, 15), sigmaX=3,sigmaY=3)
    thresh2 = cv2.threshold(blurred2, median + std, 255, cv2.THRESH_BINARY)[1]
    thresh2 = cv2.dilate(thresh2, None, iterations=4)
    print(thresh2.data.shape)
    print(image2.shape)
    """
    n1, pcov1 = curve_fit(lambda x,n: SersicProfile(x, i_e, r_e, n), r, rad, p0=1, bounds=(0,8), sigma=stdbins, absolute_sigma=True)
    n1=n1[0]

    morph_c, morph_asymm, morph_sersic_n, morph_smoothness, morph_sersic_rhalf, morph_xc_asymmetry, morph_yc_asymmetry=runstatmorph(image)

"""
    
    print('xc_centroid = {}, statmorph:{}'.format(bcX, morph.xc_centroid))
    print('yc_centroid ={}, statmorph:{}'.format(bcY, morph.yc_centroid))
    print('r20 ={}, statmorph:{}'.format(r20, (morph.r20)*30/256))
    print('r80 ={}, statmorph:{}'.format(r80, (morph.r80)*30/256))
    print('C ={}, statmorph:{}'.format(con, morph.concentration))
    print('A ={}, statmorph:{}'.format(asymm, morph.asymmetry))
    print('sersic_amplitude ={}, statmorph:{}'.format(i_e, morph.sersic_amplitude))
    print('sersic_rhalf ={}, statmorph:{}'.format(r_e, (morph.sersic_rhalf)*30/256))
    print('sersic_n ={}, statmorph:{}'.format(n1, morph.sersic_n))


    print('ellipticity_centroid =', morph.ellipticity_centroid)
    print('elongation_centroid =', morph.elongation_centroid)
    print('orientation_centroid =', morph.orientation_centroid)
    print('xc_asymmetry =', morph.xc_asymmetry)
    print('yc_asymmetry =', morph.yc_asymmetry)
    print('ellipticity_asymmetry =', morph.ellipticity_asymmetry)
    print('elongation_asymmetry =', morph.elongation_asymmetry)
    print('orientation_asymmetry =', morph.orientation_asymmetry)
    print('rpetro_circ =', morph.rpetro_circ)
    print('rpetro_ellip =', morph.rpetro_ellip)
    print('rhalf_circ =', morph.rhalf_circ)
    print('rhalf_ellip =', morph.rhalf_ellip)

    print('Gini =', morph.gini)
    print('M20 =', morph.m20)
    print('F(G, M20) =', morph.gini_m20_bulge)
    print('S(G, M20) =', morph.gini_m20_merger)
    print('sn_per_pixel =', morph.sn_per_pixel)
    
    print('S =', morph.smoothness)
    
    print('sersic_xc =', morph.sersic_xc)
    print('sersic_yc =', morph.sersic_yc)
    print('sersic_ellip =', morph.sersic_ellip)
    print('sersic_theta =', morph.sersic_theta)
    print('sky_mean =', morph.sky_mean)
    print('sky_median =', morph.sky_median)
    print('sky_sigma =', morph.sky_sigma)
"""   
    print('con:{}, asymm:{}'.format(con, asymm))
    

if __name__ == "__main__":
    #sim_name=['RecalL0025N0752','']
    #imagefileRecalL0025N0752=['RecalL0025N0752galface_726306.png','']
    sim_name=['RecalL0025N0752', 'RefL0025N0376','RefL0050N0752']
    imagefileRecalL0025N0752=['RecalL0025N0752galface_646493.png','RecalL0025N0752galface_737885.png','RecalL0025N0752galface_746518.png','RecalL0025N0752galface_853401.png','RecalL0025N0752galface_4938.png','RecalL0025N0752galface_621500.png','RecalL0025N0752galface_726306.png','RecalL0025N0752galface_51604.png']
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
