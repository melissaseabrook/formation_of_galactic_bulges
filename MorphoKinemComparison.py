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
    cv2.imwrite('galaxygraphsbin'+sim_name+'/BulgeDiscImages/opencvfindbulge'+imagefile, imagecopy)
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

def SersicProfilea(r, I_e, R_e, n, a):
	b=(2*n)-(1/3)
	G=(r/R_e)**(1/n)
	return I_e*np.exp((-b*(G-1)))+a

def findsersicindex(image, bindex, dindex):
    if bindex>0:
        maxVal, center = findcenter(image)
        rad, r_arr,stdbins, nr=radial_profile(image,center)
        max_r=np.sqrt(2)*15
        r= np.linspace(0, max_r, num=len(rad))
        i_e, r_e, centralbrightness, totalbrightness= findeffectiveradius(rad, r, nr) 
        r=r_arr/256*30
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

        poptdisca, pcovdisca=curve_fit(SersicProfilea, r[bindex:dindex], rad[bindex:dindex], p0=(i_e, r_e, 1,0), bounds=((i_e-0.5,r_e-0.1,0.1,0), (i_e+0.5,r_e+0.1,2,20)), sigma=stdbins[bindex:dindex], absolute_sigma=True)
        n_disca=poptdisca[2]
        n_disca_error=pcovdisca[2,2]
        print("I_edisca={}, R_edisca={}, n_disca={}, adisca={}".format(poptdisca[0], poptdisca[1], n_disca, poptdisca[3]))
        isolated_discsima=SersicProfilea(r, poptdisca[0], poptdisca[1], n_disca,poptdisca[3])
        isolated_bulgea= rad - isolated_discsima
        isolated_bulgea[isolated_bulgea<0]=0
        i_ebulgea, r_ebulgea, centralbrightnessbulgea, totalbrightnessbulgea= findeffectiveradius(isolated_bulgea[0:bdindex], r[0:bdindex], nr[0:bdindex]) 
        poptbulgea, pcovbulgea=curve_fit(SersicProfilea, r[0:bindex], isolated_bulgea[0:bindex], p0=(i_ebulgea, r_ebulgea, 4,0), bounds=((i_ebulgea-1,r_ebulgea-0.1,0.01,0), (i_ebulgea+1,r_ebulgea+0.1,10,20)), sigma=stdbins[0:bindex], absolute_sigma=True)
        n_bulgea= poptbulgea[2]
        n_bulgea_error= pcovbulgea[2,2]
        print("I_ebulgea={}, R_ebulgea={}, n_bulgea={}, abulgea={}".format(poptbulgea[0],poptbulgea[1], n_bulgea, poptbulgea[2]))

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


    else:
        n_total=0
        n_bulge=0
        n_disc=0
        n_bulgea=0
        n_disca=0
        n_bulge_exp=0
        n_total_error=0
        n_bulge_error=0
        n_disc_error=0
        n_bulgea_error=0
        n_disca_error=0
        n_bulge_exp_error=0
    return n_total, n_disca, n_bulgea, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disca_error, n_bulgea_error, n_disc_error, n_bulge_error, n_bulge_exp_error

def drop_numerical_outliers(df, z_thresh):
    constrains=df.select_dtypes(include=[np.number]).apply(lambda x: np.abs(stats.zscore(x)) <z_thresh).all(axis=1)
    df.drop(df.index[~constrains], inplace=True)

def removeoutlierscolumn(df, column_name, sigma):
    df=df[np.abs(df[column_name]-df[column_name].mean())<=(sigma*df[column_name].std())]
    return df


def invert(var):
    if var != 0:
        return(1/var)*10
    else:
        return 0


def threeDplot(df, x,y,z, column_size, column_colour):
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
    ax.contourf(X,Y,hist, zdir='z', offset=minz, cmap=cm.YlOrRd, alpha=0.6)
    
    hist, binx, biny=np.histogram2d(df[z], df[x], bins=7)
    X = np.linspace(minx, maxx, hist.shape[0])
    Z = np.linspace(minz, maxz, hist.shape[1])
    X,Z=np.meshgrid(X,Z)
    ax.contourf(X,hist,Z, zdir='y', offset=maxy, cmap=cm.YlOrRd, alpha=0.6)

    hist, binx, biny=np.histogram2d(df[y], df[z], bins=7)
    Y = np.linspace(miny, maxy, hist.shape[0])
    Z = np.linspace(minz, maxz, hist.shape[1])
    Z,Y=np.meshgrid(Z,Y)
    ax.contourf(hist,Y,Z, zdir='x', offset=minx, cmap=cm.YlOrRd, alpha=0.6)
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
    ax.savefig('galaxygraphsbin'+sim_name+'/'+x+'vs'+y+'.png')
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
    plt.savefig('galaxygraphsbin'+sim_name+'/histogramofsersicindices.png')
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

def plotbulgetodisc(df, sim_name):

    print(df.shape)
    drop_numerical_outliers(df, 3)

    print(df.shape)
    df=df[df.n_total_error<10]
    print(df.shape)
    #df=df[df.n_bulge_exp_error<10]
    print(df.shape)
    #df=df[df.n_bulge_error<10]
    print(df.shape)

    df=df[df.n_total>0.05]
    df['sSFR']=df.apply(lambda x: (x.SFR/x.mass), axis=1)
    df=removeoutlierscolumn(df, 'sSFR', 2)
    df=removeoutlierscolumn(df, 'mass', 2)
    df=df[df.sSFR>0]
    print(df.shape)

    df['logsSFR']=df.apply(lambda x: np.log10(x.sSFR), axis=1)
    df['logBHmass']=df.apply(lambda x: np.log10(x.BHmass), axis=1)
    df['logmass']=df.apply(lambda x: np.log10(x.mass), axis=1)
    df['sSFR']=df.apply(lambda x: (x.SFR/x.mass), axis=1)
    df['logsSFR']=df.apply(lambda x: np.log10(x.sSFR), axis=1)
    df['logBHmass']=df.apply(lambda x: np.log10(x.BHmass), axis=1)
    df['logmass']=df.apply(lambda x: np.log10(x.mass), axis=1)

    #stackedhistogram(df, 'n_total','n_disc','n_bulge','n_bulge_exp')
    #subplothistograms(df, 'n_total','n_disc','n_bulge','n_disca','n_bulgea','n_bulge_exp')

    df['dtototal']=df.apply(lambda x: (1-x.btdintensity), axis=1)
    #colorbarplot(df, 'n_total', 'DiscToTotal', 'logmass', 'logsSFR', 'BHmass')
    threeDplot(df, 'n_total','DiscToTotal','logBHmass', 'mass', 'logsSFR')
    exit()
    
    df['dtbradius']=df.apply(lambda x: invertbtd(x.btdradius), axis=1)
    df['dtbintensity']=df.apply(lambda x: invertbtd(x.btdintensity), axis=1)
    df['ZBin']=pd.qcut(df.Z, 6)
    
    """
    size=100*(df.mass)/(df.mass.max())
    g=sns.PairGrid(df, vars=['DiscToTotal','dtbradius','dtbintensity','n_total','n_disca','n_disc','n_bulgea','n_bulge','SFR', 'BHmass'])
    g.map_diag(sns.kdeplot)
    g.hue_vals=df['mass']
    g.hue_names=df['mass'].unique()
    g.palette=sns.color_palette('Blues', len(g.hue_names))
    g.map_offdiag(plt.scatter, s=size)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('PairPlot for relationships between paramters, coloured by galaxy size')
    g.savefig('galaxygraphsbin'+sim_name+'/allbulgeparametersrelationships.png')
    plt.show()
    plt.close()
    
    size=100*(df.mass)/(df.mass.max())
    g=sns.PairGrid(df, x_vars=['DiscToTotal','SFR', 'BHmass'], y_vars=['n_total','n_disc','n_bulge','n_disca','n_bulgea'])
    g.hue_vals=df['mass']
    g.hue_names=df['mass'].unique()
    g.palette=sns.color_palette('Blues', len(g.hue_names))
    g.map(plt.scatter, s=size)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('PairPlot for relationships between paramters, coloured by galaxy size')
    g.savefig('galaxygraphsbin'+sim_name+'/selectedbulgeparametersrelationships.png')
    plt.show()
    plt.close()
   """ 
   
    stackedhistogram(df, 'n_total','n_disc','n_bulge','n_disca','n_bulgea','n_bulge_exp')
    plt.close()
    subplothistograms(df, 'n_total','n_disc','n_bulge','n_disca','n_bulgea','n_bulge_exp')
    plt.close()
    
    fig=plt.figure()
    ax0=fig.add_subplot(221)
    ax1=fig.add_subplot(222)
    ax2=fig.add_subplot(223)
    ax3=fig.add_subplot(224)
    markers={"low":'^', "okay":'o', 'high':'s'}
    g=sns.relplot(x='n_disc', y='DiscToTotal', size='mass', sizes=(10,150), hue='SFR', palette='autumn', style='BHBin', markers=markers,data=df, ax=ax0)
    g=sns.relplot(x='n_disca', y='DiscToTotal', size='mass', sizes=(10,150), hue='SFR', palette='autumn', style='BHBin', markers=markers,data=df, ax=ax1)
    g=sns.relplot(x='n_bulge', y='DiscToTotal', size='mass', sizes=(10,150), hue='SFR', palette='autumn', style='BHBin', markers=markers,data=df,ax=ax2)
    g=sns.relplot(x='n_bulgea', y='DiscToTotal', size='mass', sizes=(10,150), hue='SFR', palette='autumn', style='BHBin', markers=markers,data=df, ax=ax3)
    fig.tight_layout()
    fig.savefig('galaxygraphsbin'+sim_name+'/DiscToTotalvsn_disc.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    sim_name='RefL0050N0752'
    read_data=True
    if(read_data):
        print('.........reading.......')
        df=pd.read_csv('EAGLEbulgedisc'+sim_name+'.csv')
    else:
        print('.........writing.......')
        df=pd.read_csv('EAGLEimagesdf'+sim_name+'.csv')
        discbulgetemp=[]
        for filename in df['filename']:
            BGRimage=cv2.imread('galaxyimagebin'+sim_name+'/'+filename)
            btdradius, btdintensity, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius =findandlabelbulge(BGRimage, filename, sim_name)
            n_total, n_disca, n_bulgea, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disca_error, n_bulgea_error, n_disc_error, n_bulge_error, n_bulge_exp_error=findsersicindex(BGRimage, bradius, hradius)
            discbulgetemp.append([filename, btdradius, btdintensity,n_total, n_disca, n_bulgea, n_disc, n_bulge, n_bulge_exp, n_total_error, n_disca_error, n_bulgea_error, n_disc_error, n_bulge_error, n_bulge_exp_error, star_count, hradius, bradius, disc_intensity, bulge_intensity, btotalintensity, btotalradius])
        discbulgedf=pd.DataFrame(discbulgetemp, columns=['filename', 'btdradius', 'btdintensity','n_total','n_disca','n_bulgea','n_disc','n_bulge','n_bulge_exp', 'n_total_error', 'n_disca_error', 'n_bulgea_error', 'n_disc_error', 'n_bulge_error', 'n_bulge_exp_error', 'star_count', 'discradius', 'bulgeradius', 'disc_intensity', 'bulge_intensity', 'btotalintensity', 'btotalradius'])
        df.filename.astype(str)
        discbulgedf.filename.astype(str)
        df=pd.merge(df, discbulgedf, on=['filename'], how='outer')
        df.to_csv('EAGLEbulgedisc'+sim_name+'.csv')

    plotbulgetodisc(df, sim_name)


    


