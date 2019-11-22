import numpy as np
from PIL import Image
import cv2
from imutils import contours
from skimage import measure
import imutils
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats


class FeatureDetectionTools:
    def __init__(self,image,sigma,box_size):
        self.image=image
        self.sigma=sigma
        self.box_size=box_size

    def sigmaclip(self, image, sigma, box_size):
        #shows relative fluctuations in pixel intensities
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        median_background=np.median(gray)
        std_background=np.std(gray)
        print(median_background)
        print(median_background)
        print(gray.shape)
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
        
    def findlightintensity(self, image, radius, center):
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
        return cumulativebrightness

    def findcenter(self, image):
        #finds coords of central bulge
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11,11), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        return maxVal, maxLoc

    def findandlabelbulge(self, image, imagefile):
        #locates central bulge and diffuse halo, and marks this on the image
        imagecopy=image.copy()
        median=np.median(image)
        std=np.std(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred1 = cv2.GaussianBlur(gray, ksize=(11, 11), sigmaX=3,sigmaY=3)
        thresh1 = cv2.threshold(blurred1, median + 5*std, 255, cv2.THRESH_BINARY)[1]
        thresh1 = cv2.erode(thresh1, None, iterations=2)
        #thresh1 = cv2.dilate(thresh1, None, iterations=4)
        blurred2 = cv2.GaussianBlur(gray, ksize=(11, 11), sigmaX=3,sigmaY=3)
        thresh2 = cv2.threshold(blurred2, median +std, 255, cv2.THRESH_BINARY)[1]
        thresh2 = cv2.dilate(thresh2, None, iterations=4)
        #blurred3 = cv2.GaussianBlur(gray, ksize=(11, 11), sigmaX=2,sigmaY=2)
        #thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,61,0)
        thresh3= sigmaclip(image,4,8)
        cv2.imshow("thresh1", thresh1), cv2.waitKey(0)
        cv2.imshow("thresh2", thresh2), cv2.waitKey(0)
        cv2.imshow("thresh3", thresh3), cv2.waitKey(0)
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
        #find halo
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
        print("halo radius:{}, halo centre({},{})".format(hradius, hcX,hcY))
        if numPixels > 60: 
            cv2.putText(imagecopy, "halo", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)


        
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
        halotobulgeradius=hradius/bradius
        print("halo radius:bulge radius ={}".format(halotobulgeradius))
        print("star count ={}".format(count))

        halo_intensity=findlightintensity(image, hradius, (hcX,hcY))
        bulge_intensity=findlightintensity(image, bradius, (bcX,bcY))
        halotobulgeintensity= halo_intensity/bulge_intensity
        print("halo intensity = {}, bulge intensity ={}, halo:bulge intensity ={}".format(halo_intensity, bulge_intensity, halotobulgeintensity))
        cv2.imshow("Image", imagecopy)
        cv2.imwrite('galaxygraphsbinRecal/opencvfindbulge'+imagefile, imagecopy)
        cv2.waitKey(0)

    def central1kpcregion(self, image):
        #circles a 1.5kpc region around the cenral bulge
        width, height =image.shape[:2]
        #x=int(width/2)
        #y=int(height/2)
        radius = int(width*1.5/30)
        maxVal, maxLoc = findcenter(image)
        cv2.circle(image, maxLoc, radius, (255, 0, 0), 1)
        cv2.imshow("Central 1.5kpc Region", image)
        cv2.waitKey(0)

class Histograms:
    def __init__(self,image,image_file):
        self.image=image
        self.image_file=image_file

    def matplotlibcolourhistogram(self, image, image_file):
        #plots histogram of different color intensities
        plt.subplot(121)
        plt.imshow(image, cmap='Blues'), plt.axis('off')
        plt.colorbar(orientation='horizontal')
        plt.imshow(image, cmap='Greens'), plt.axis('off')
        plt.colorbar(orientation='horizontal')
        plt.imshow(image, cmap='Reds'), plt.axis('off')
        plt.colorbar(orientation='horizontal')
        plt.subplot(122)
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv2.calcHist([image],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        plt.xlabel('Pixel Values'), plt.ylabel('No pixels')
        plt.tight_layout()
        plt.savefig('galaxygraphsbinRecal/colorplthistogram'+image_file)
        plt.show()

    def matplotlibhistogram(self, image, image_file):
        #plots histogram of grayscale color intensitiesE
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        plt.subplot(211)
        plt.imshow(gray, cmap='gray'), plt.axis('off')
        plt.colorbar(orientation='horizontal')
        plt.subplot(212)
        plt.hist(gray.ravel(),256,[0,256])
        plt.xlabel('GrayScale Pixel Values'), plt.ylabel('No pixels')
        plt.xlim(0,256)
        plt.savefig('galaxygraphsbinRecal/grayplthistogram'+image_file)
        plt.show()

    def opencvhistogram(self, image, imagefile):
            #plots histogram for original and unsmaksed image
        # create a mask
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        mask = cv2.dilate(thresh, None, iterations=4)
        #mask = np.zeros(thresh.shape, dtype="uint8")
        masked_img = cv2.bitwise_and(image,image,mask = mask)
        # Calculate histogram with mask and without mask
        # Check third argument for mask
        hist_full = cv2.calcHist([image],[0],None,[256],[0,256])
        hist_mask = cv2.calcHist([image],[0],mask,[256],[0,256])
        plt.subplot(221), plt.imshow(image), plt.title('image'), plt.axis('off')
        plt.subplot(222), plt.imshow(mask,'gray'), plt.title('gray'), plt.axis('off')
        plt.subplot(223), plt.imshow(masked_img, 'gray'), plt.title('masked'), plt.axis('off')
        plt.subplot(224), plt.plot(hist_full,label='full'), plt.plot(hist_mask,label='masked'), plt.title('histogram'), plt.legend()
        plt.xlim([0,256])
        plt.tight_layout()
        plt.xlabel('ColourValue')
        plt.ylabel('Intensity')
        plt.savefig('galaxygraphsbinRecal/opencvhistogram'+imagefile)
        plt.show()

class RadialProfileTools:
    def __init__(self,image,image_file):
        self.image=image
        self.image_file=image_file

    def radial_profile(self, image, center):
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
        stdbins, bin_edges, binnumber=stats.binned_statistic(r.ravel(),image.ravel(), 'std', bins=len(radialprofile))
        stdbins[0]=stdbins[1]
        stdbins[stdbins<0.5]=0.5
        #meanbins, bin_edges, binnumber=stats.binned_statistic(r.ravel(),image.ravel(), 'mean', bins=len(radialprofile))
        return radialprofile, stdbins

    def findeffectiveradius(self, radialprofile, r):
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

    def findlightintensity(self, radialpofile, radius):
        radius=int(radius)
        cumulativebrightness=np.sum(radialpofile[0:radius])
        return cumulativebrightness


    def SersicProfile(self, r, I_e, R_e, n):
        b=np.exp(0.6950 + np.log(n) - (0.1789/n))
        G=(r/R_e)**(1/n)
        return I_e*np.exp((-b*(G-1)))

