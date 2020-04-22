"""Investigate the evolving population of galaxies"""

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
from matplotlib.lines import Line2D
import matplotlib.colors as mcol
import matplotlib.ticker as mticker
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
from astropy.cosmology import Planck13, z_at_value
import astropy.units as u
from astropy import constants as const
from astropy.modeling import models, fitting
import statmorph


def logx(x):
    #log base 10 of x
    if x !=0:
        if x>0:
            return np.log10(x)
        if x<0:
            return -np.log10(-x)
    else:
        return 0

def divide(x,y):
    #divide x by y
    if y !=0:
        return (x)/(y)
    else:
        return 0

def invert(var):
    #invert and multiply variable
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
    #convert zeroes to nans with a cap on z
    if z<0.001:
        return np.nan
    elif frac==0:
        return np.nan
    else:
        return frac

def roundx(x, dec=1):
    #round specifing decimal
    if x>0:
        return np.round(x, decimals=dec)
    elif x<0:
        return np.round(x, decimals=dec)
    else: 
        return np.nan

def threshtonan(x, thresh):
    #if x below threshold, return nan
    if x<thresh:
        return np.nan
    else:
        return x

def threshtonan2(x, y, thresh):
    #if x below threshold, return 0
    if x<thresh:
        return 0
    else:
        return y

def cattonan(x, y, cat):
    #if x is not in category, return 0
    if x==cat:
        return y
    else:
        return 0

def drop_numerical_outliers(df, z_thresh):
    #drop outliers below a treshold z
    constrains=df.select_dtypes(include=[np.number]).apply(lambda x: np.abs(stats.zscore(x)) <z_thresh).all(axis=1)
    df.drop(df.index[~constrains], inplace=True)

def removeoutlierscolumn(df, column_name, sigma):
    #remove outliers in specific column, sigma = std
    df=df[np.abs(df[column_name]-df[column_name].mean())<=(sigma*df[column_name].std())]
    return df

def getImage(path):
    #produce image path
    return OffsetImage(plt.imread('evolvinggalaxyimagebinmainbranch'+sim_name+'/'+path), zoom=0.15)

def categorise(asymm, param, thresh, athresh):
    #categories image based on asymmetry and morphological threshold.
    if asymm > athresh:
        return 'A'
    if param > thresh:
        return 'B'
    else:
        return 'D'

def classifymergermass(x):
    #classify mergers by stellar mass
    if x>0.2: 
        return 'major'
    elif x>0.01:
        return 'minor'
    elif x>0.0001:
        return 'accretion'

def classifymergergas(x):
    #classify mergers by fractional gas mass
    if x>0.35: 
        return 'gasrich'
    elif x<0.25:
        return 'gaspoor'
    else:
        return 'undefined'

def colourmergergas(x):
    #produce colour based on merger category
    if (x=='gasrich'):
        return 'g'
    elif (x=='undefined'):
        return 'orange'
    elif (x=='gaspoor'):
        return 'red'

def colourmergermass(x):
     #produce colour based on merger category
    if (x=='major'):
        return 'g'
    elif (x=='accretion'):
        return 'orange'
    elif (x=='minor'):
        return 'red'

def mediancolor(x, col):
    #colour parameter based on median
    if x>df[col].median():
        return 'g'
    else:
        return 'yellow'

def cutBHmass(x):
    #disregard unrealsitic BH masses
    if x>0:
        if x<4.5:
            return np.nan
        else:
            return x
    else:
        return x

def bulgetranslists(df, binparam, thresh, threshstep):
    #produce lists of galaxy transitions
    B2B =[]
    D2D= []
    B2D=[]
    D2B=[]
    BDB=[]
    DBD=[]
    nmax=df.ProjGalaxyID.nunique()
    for id in df.ProjGalaxyID.unique():
        tempdf=df[df.ProjGalaxyID==id]
        tempdf=tempdf.sort_values('lbt').reset_index()
        if tempdf[binparam].max() < thresh:
            D2D.append(id)
        elif tempdf[binparam].min() > thresh:
            B2B.append(id)
        elif tempdf[binparam].iloc[0]>thresh:
            if tempdf[binparam].iloc[tempdf.lbt.idxmax()] <thresh-(threshstep):
                B2D.append(id)
            else:
                BDB.append(id)
        elif tempdf[binparam].iloc[0]<thresh:
            if tempdf[binparam].iloc[tempdf.lbt.idxmax()] >thresh+(threshstep):
                D2B.append(id)
            else:
                DBD.append(id)
    return B2B, D2D, B2D, D2B, BDB, DBD

def bulgetrans(x, B2B, D2D, B2D, D2B, BDB, DBD):
    #assign transitioning category to each galaxy based on list
    if x in B2B:
        return 'B2B'
    elif x in D2D:
        return 'D2D'
    elif x in B2D:
        return 'B2D'
    elif x in D2B:
        return 'D2B'
    elif x in BDB:
        return 'BDB'
    elif x in DBD:
        return 'DBD'

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
    #produce 3D plot
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
    #produce a coloured scatter diagram with colourbar and varying marker sizes and shapes
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
    #prdoce plot of stacked histograms
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
    #produce individual histogram subplots
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
    #prodcue plot for evolution of single variable
    fig, (ax1,ax2, ax3)=plt.subplots(3,1, sharex=True)
    #plt.subplot(211)
    sns.scatterplot(x='z',y=param, hue='ProjGalaxyID',data=df, size=param_size, palette=sns.color_palette('hls', df.ProjGalaxyID.nunique()), legend=False, ax=ax1)
    sns.lineplot(x='z',y=param, hue='ProjGalaxyID',data=df, palette=sns.color_palette('hls', df.ProjGalaxyID.nunique()), ax=ax1,legend=False, linewidth=0.8)
    #ax0=ax1.twinx()
    sns.lineplot(x='z',y=param2, hue='ProjGalaxyID',data=df, palette=sns.color_palette('hls', df.ProjGalaxyID.nunique()),  ax=ax2, legend=False)
    sns.lineplot(x='z',y='n2d', hue='ProjGalaxyID',data=df, palette=sns.color_palette('hls', df.ProjGalaxyID.nunique()),  ax=ax3, legend=False)
    """
    for i in range(df.ProjGalaxyID.nunique()):
        ax0.lines[i].set_linestyle('--')
    """
    #plt.subplot(212)
    #sns.lineplot(x='z',y='logsSFR', hue='ProjGalaxyID',data=df, palette=sns.color_palette('hls', df.ProjGalaxyID.nunique()), ax=ax2, legend=False)
    #ax3=ax2.twinx()
    #sns.lineplot(x='z',y='n_total', hue='ProjGalaxyID',data=df, palette=sns.color_palette('hls', df.ProjGalaxyID.nunique()),  ax=ax3, legend=False)
    plt.legend(bbox_to_anchor=(1.1,0.8), loc='center left')
    plt.xlim(0,3)
    plt.title('Evolution of '+param+' sized by'+param_size)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/evolution of'+param+'and'+param2+'.png')
    plt.show()

def specificgalaxyplot(df, galaxyid, param1, param2, param3, param4):
    #plot 4 galaxy parameters alog with galaxy images for a specific galaxy
    df2=df[df.ProjGalaxyID==galaxyid]
    df2=df2[df2.n_total>0]
    df2=df2[['z', param1, param2, param3, param4, 'filename']]

    x = df2['z'].tolist()
    y_image=np.zeros(df2.z.nunique())
    y1= df2[param1].tolist()
    y2 = df2[param2].tolist()
    y3 = df2[param3].tolist()
    y4 = df2[param4].tolist()
    paths = df2['filename'].tolist()

    fig, (ax1,ax0) = plt.subplots(2,1, gridspec_kw={'height_ratios': [7.8, 1]}, sharex=True, figsize=(12,6))
    ax2=ax1.twinx()
    ax3=ax1.twinx()
    ax4=ax1.twinx()
    
    axes=[ax1,ax2,ax3,ax4]
    ax2.spines['right'].set_position(('axes', -0.2))
    ax3.spines['right'].set_position(('axes', -0.35))
    ax4.spines['right'].set_position(('axes', -0.5))
    axes[-1].set_frame_on(True)
    axes[-1].patch.set_visible(False)

    ax1.plot(x, y1,  'r', label=param1)
    ax1.yaxis.label.set_color('red')
    ax1.tick_params(axis='y', colors='red')
    ax1.set_ylabel(param1)

    ax2.plot(x, y2,  'b--', label=param2)
    ax2.yaxis.label.set_color('b')
    ax2.tick_params(axis='y', colors='b')
    ax2.set_ylabel(r'$n_{2d}$', labelpad=-40)

    ax3.plot(x, y3,  'g--', label=param3)
    ax3.yaxis.label.set_color('g')
    ax3.tick_params(axis='y', colors='g')
    ax3.set_ylabel(r'log(sSFR) [Gyr]', labelpad=-45)

    ax4.plot(x, y4,  'y', label=param4)
    ax4.yaxis.label.set_color('y')
    ax4.tick_params(axis='y', colors='y')
    ax4.set_ylabel(r'log($M_{*}$) [$M_{\odot}$]', labelpad=-50) 

    #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    
    for x0, y0, path in zip(x, y_image,paths):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        ax0.add_artist(ab)
    ax0.yaxis.set_visible(False) 
    ax0.set_ylim(-0.01,0.01) 
    ax0.set_xlabel('z')
    #ax.plot(x, y, ax)
    plt.subplots_adjust(left=0.4,hspace=0)
    plt.title('Pictorial Evolution of Galaxy '+str(galaxyid))
    plt.draw()
    #ax1.legend(lines, labels, bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=4)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/PictorialEvolutionGalaxy'+str(galaxyid)+'.png')
    plt.close()

def specificgalplotratesofvariabless(df, galaxyid):
    #plot rates of variables for a single galaxy
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
    #plot several masses for a specific galaxy
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

def plotmovinghistogram(df, histparam, binparam):
    #plot evolving histogram, highglighted by upper and lower qauntiles of binparam
    z0df=df[df.zrounded==0.]
    df=df[(df.zrounded==0.1) | (df.zrounded==0.5) | (df.zrounded==1.0) | (df.zrounded==2.0)]
    z0df=z0df[['ProjGalaxyID', binparam]]
    z0df['marker_bin']=pd.qcut(z0df[binparam],3, labels=['10','90','100'])
    #z0df['marker_bin']=pd.qcut(z0df[binparam], [0.0, 0.05, 0.3,0.7,0.95,1.0], labels=['20','40','60','80','100'])
    df=pd.merge(df, z0df, on=['ProjGalaxyID'], how='left',  suffixes=('','_proj'))

    fig, axs =plt.subplots(4, 2, sharex=True, figsize=(9,6))
    fig.suptitle('Time evolution of historgram of '+histparam+' showing distribution of '+binparam)
    axs[0,0].set_title('0-10th percentile of '+binparam)
    axs[0,1].set_title('90-100th percentile of '+binparam)
    binedgs=np.linspace(df[histparam].min(), df[histparam].max(), 20)
    for i,zi in enumerate([0.1, 0.5, 1.0, 2.0]):
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

def plotmovinghistogram4(df, histparam1, histparam2, histparam3, histparam4, binparam, label1, label2, label3, label4):
    #plot 4 moving histograms
    #z0df=df[df.zrounded==0.]
    df=df[(df.zrounded==0.1) | (df.zrounded==0.5) | (df.zrounded==1.) | (df.zrounded==1.5)]
    #z0df=z0df[['ProjGalaxyID', binparam]]
    #z0df['marker_bin']=pd.qcut(z0df[binparam],10, labels=['10','20','30','40','50','60','70','80','90','100'])
    #df=pd.merge(df, z0df, on=['ProjGalaxyID'], how='left',  suffixes=('','_proj'))
    fig, axs =plt.subplots(4, 4, sharex='col', sharey='row', figsize=(9,6))
    fig.suptitle('Time evolution of historgram of '+histparam1+' showing distribution of '+binparam)
    binedgs1=np.linspace(df[histparam1].min(), df[histparam1].max(), 30)
    binedgs2=np.linspace(df[histparam2].min(), df[histparam2].max(), 30)
    binedgs3=np.linspace(df[histparam3].min(), df[histparam3].max(), 30)
    binedgs4=np.linspace(df[histparam4].min(), df[histparam4].max(), 30)
    for i,zi in enumerate([0.1, 0.5, 1.0, 1.5]):
        
        zdf=df[df.zrounded==zi]
        #lowdf=zdf[zdf.marker_bin=='10']
        #highdf=zdf[zdf.marker_bin=='100']
        highdf=zdf[zdf[binparam]=='D2D']
        lowdf=zdf[(zdf[binparam]=='B2B')]


        axs[i,0].hist(zdf[histparam1], color="k", alpha=0.4,label='z='+str(zi), histtype='stepfilled', bins=binedgs1)
        axs[i,1].hist(zdf[histparam2], color="k",  alpha=0.4,histtype='stepfilled', bins=binedgs2)
        axs[i,2].hist(zdf[histparam3], color="k",  alpha=0.4,histtype='stepfilled', bins=binedgs3)
        axs[i,3].hist(zdf[histparam4], color="k",  alpha=0.4,histtype='stepfilled', bins=binedgs4)
        axs[i,0].hist(lowdf[histparam1], color="r", alpha=0.5, histtype='step', bins=binedgs1)
        axs[i,0].hist(highdf[histparam1], color="b", alpha=0.5, histtype='step', bins=binedgs1)
        axs[i,1].hist(lowdf[histparam2], color="r", alpha=0.5, histtype='step', bins=binedgs2)
        axs[i,1].hist(highdf[histparam2], color="b", alpha=0.5, histtype='step', bins=binedgs2)
        axs[i,2].hist(lowdf[histparam3], color="r", alpha=0.5, histtype='step', bins=binedgs3)
        axs[i,2].hist(highdf[histparam3], color="b", alpha=0.5, histtype='step', bins=binedgs3)
        axs[i,3].hist(lowdf[histparam4], color="r", alpha=0.5, histtype='step', bins=binedgs4, label='constant bulges')
        axs[i,3].hist(highdf[histparam4], color="b", alpha=0.5, histtype='step', bins=binedgs4,  label='constant discs')
        #axs[i,0].set_xlabel('')
        #axs[i,1].set_xlabel('')
        #axs[i,0].set_ylabel('')
        axs[i,0].set_ylabel('Count')
        axs[i,0].legend()
        
    axs[3,0].set_xlabel(label1)
    axs[3,1].set_xlabel(label2)
    axs[3,2].set_xlabel(label3)
    axs[3,3].set_xlabel(label4)
    plt.subplots_adjust(wspace=0, hspace=0)
    axs[3,3].legend()
    #handles, labels = axs[3].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='lower center')
    #plt.tight_layout()
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Histogramof'+histparam1+'highlighted'+binparam+'.png')
    plt.show()

def plotbulgedisctransz(df, maxz, param, thresh, threshstep):
    #plot bulge to dsc tranition diagram, showing z on x axis
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
    #prodcues coloured trnaition diagram
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
        if tempdf[param].min() > thresh:
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
    
    #calculate errors prodcued by varying the thresholds
    B2Bl, D2Dl, B2Dl, D2Bl, BDBl, DBDl = bulgetranslists(df, 'n2d',thresh - threshstep, threshstep)
    B2Bu, D2Du, B2Du, D2Bu, BDBu, DBDu = bulgetranslists(df, 'n2d', thresh + threshstep, threshstep)
    
    B2BU=np.abs(len(B2Bu)-len(B2B))
    B2BL=np.abs(len(B2B)-len(B2Bl))
    D2DU=np.abs(len(D2Du)-len(D2D))
    D2DL=np.abs(len(D2D)-len(D2Dl))
    B2DU=np.abs(len(B2Du)-len(B2D))
    B2DL=np.abs(len(B2D)-len(B2Dl))
    D2BU=np.abs(len(D2Bu)-len(D2B))
    D2BL=np.abs(len(D2B)-len(D2Bl))
    BDBU=np.abs(len(BDBu)-len(BDB))
    BDBL=np.abs(len(BDB)-len(BDBl))
    DBDU=np.abs(len(DBDu)-len(DBD))
    DBDL=np.abs(len(DBD)-len(DBDl))

    fig, ax =plt.subplots(2, 6, sharey='row', sharex='row', figsize=(12,6))
    fig.suptitle('Time evolution'+param)

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
    ax[1,0].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,0].bar(0,len(B2B), color='purple',  yerr=[[B2BL], [B2BU]],  capsize=5.0, alpha=0.8)
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
    ax[0,1].bar(0,len(D2D), color='purple', yerr=[[D2DL], [D2DU]],capsize=5.0, alpha=0.8)
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
    ax[0,2].bar(0,len(B2D), color='purple', yerr=[[B2DL], [B2DU]],capsize=5.0, alpha=0.8)
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
    ax[0,3].bar(0,len(D2B), color='purple', yerr=[[D2BL], [D2BU]], capsize=5.0, alpha=0.8)
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
    ax[0,4].bar(0,len(BDB), color='purple', yerr=[[BDBL], [BDBU]],capsize=5.0, alpha=0.8)
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
    ax[0,5].bar(0,len(DBD), color='purple', yerr=[[DBDL], [DBDU]], capsize=5.0, alpha=0.8)
    ax[0,5].text(-.1, 1, str(round(100*len(DBD)/nmax, 1))+'%', fontsize=12, color='white')

    ax[0,0].set_title('B'),ax[0,1].set_title('D'),ax[0,2].set_title('BD'),ax[0,3].set_title('DB'),ax[0,4].set_title('BDB'),ax[0,5].set_title('DBD')
    ax[1,2].set_xlabel('look back time (Gyr)')
    ax[1,0].set_ylabel(param)
    ax[0,0].set_ylabel('count')
    ax[0,0].set_ylim(0,80)
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
    cbar=plt.colorbar(sm, cax=cbar_ax).set_label(r'residual of sSFR per mass per z [$M_{\odot} Gyr^{-1}$]')
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Evolution of'+str(param)+'thresh'+str(thresh)+'zmax'+str(maxz)+'color2.png')
    plt.show()

def plotbulgedisctranscolourmerger(df, maxz, param, colorparam, merger, mergername, mergername2, thresh, threshstep, merge=True):
    #plot transition diagram showing positions of mergers.
    B2B, D2D, B2D, D2B, BDB, DBD = bulgetranslists(df, 'n2d', thresh, threshstep)
    df['transtypen2d']=df.apply(lambda x: bulgetrans(x.ProjGalaxyID, B2B, D2D, B2D, D2B, BDB, DBD), axis=1)

    B2B=df[df.transtypen2d=='B2B']
    D2D=df[df.transtypen2d=='D2D']
    B2D=df[df.transtypen2d=='B2D']
    D2B=df[df.transtypen2d=='D2B']
    BDB=df[df.transtypen2d=='BDB']
    DBD=df[df.transtypen2d=='DBD']

    fig, ax =plt.subplots(2, 6, sharey='row', sharex='row', figsize=(12,6))
    fig.suptitle('Time evolution'+param)

    #Cmap=plt.get_cmap('RdBu')
    Cmap=mcol.LinearSegmentedColormap.from_list("cmop", ['tomato','cornflowerblue'])
    Norm=plt.Normalize(df[colorparam].min(),df[colorparam].max())
    for id in B2B.ProjGalaxyID.unique():
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
        dfmergetemp=temp[temp[mergername]>0.]
        ax[1,0].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=5)
        
    ax[1,0].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    dfM=df[df[mergername]>0.]
    ax[0,0].bar((-1,0,1),\
        (len(dfM[(dfM['transtypen2d']=='B2B') & (dfM[merger]=='g')]),len(dfM[(dfM['transtypen2d']=='B2B') & (dfM[merger]=='orange')]),len(dfM[(dfM['transtypen2d']=='B2B') & (dfM[merger]=='red')])),\
             color=['g', 'orange', 'red'])
    ax[0,0].text(-1.3, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='B2B') & (dfM[merger]=='g')])/len(dfM[dfM['transtypen2d']=='B2B']))) +'%', fontsize=10, color='k')
    ax[0,0].text(-0.3, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='B2B') & (dfM[merger]=='orange')])/len(dfM[dfM['transtypen2d']=='B2B']))) +'%', fontsize=10, color='k')
    ax[0,0].text(0.7, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='B2B') & (dfM[merger]=='red')])/len(dfM[dfM['transtypen2d']=='B2B']))) +'%', fontsize=10, color='k')
   
    for id in D2D.ProjGalaxyID.unique():
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
        dfmergetemp=temp[temp[mergername]>0.]
        ax[1,1].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=5)
    ax[1,1].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)

    ax[0,1].bar((-1,0,1),\
        (len(dfM[(dfM['transtypen2d']=='D2D') & (dfM[merger]=='g')]),len(dfM[(dfM['transtypen2d']=='D2D') & (dfM[merger]=='orange')]),len(dfM[(dfM['transtypen2d']=='D2D') & (dfM[merger]=='red')])),\
             color=['g', 'orange', 'red'])
    ax[0,1].text(-1.3, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='D2D') & (dfM[merger]=='g')])/len(dfM[dfM['transtypen2d']=='D2D']))) +'%', fontsize=10, color='k')
    ax[0,1].text(-0.3, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='D2D') & (df[merger]=='orange')])/len(dfM[dfM['transtypen2d']=='D2D']))) +'%', fontsize=10, color='k')
    ax[0,1].text(0.7, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='D2D') & (df[merger]=='red')])/len(dfM[dfM['transtypen2d']=='D2D']))) +'%', fontsize=10, color='k')

    for id in B2D.ProjGalaxyID.unique():
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
        dfmergetemp=temp[temp[mergername]>0.]
        ax[1,2].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=5)
    ax[1,2].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    
    ax[0,2].bar((-1,0,1),\
        (len(dfM[(dfM['transtypen2d']=='B2D') & (dfM[merger]=='g')]),len(dfM[(dfM['transtypen2d']=='B2D') & (dfM[merger]=='orange')]),len(dfM[(dfM['transtypen2d']=='B2D') & (dfM[merger]=='red')])),\
             color=['g', 'orange', 'red'])
    ax[0,2].text(-1.3, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='B2D') & (dfM[merger]=='g')])/len(dfM[dfM['transtypen2d']=='B2D']))) +'%', fontsize=10, color='k')
    ax[0,2].text(-0.3, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='B2D') & (dfM[merger]=='orange')])/len(dfM[dfM['transtypen2d']=='B2D']))) +'%', fontsize=10, color='k')
    ax[0,2].text(0.7, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='B2D') & (dfM[merger]=='red')])/len(dfM[dfM['transtypen2d']=='B2D']))) +'%', fontsize=10, color='k')
   
    for id in D2B.ProjGalaxyID.unique():
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
        dfmergetemp=temp[temp[mergername]>0.]
        ax[1,3].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=5)
    ax[1,3].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    print(dfM[(dfM['transtypen2d']=='D2B') & (dfM[merger]=='g')])
    ax[0,3].bar((-1,0,1),\
        (len(dfM[(dfM['transtypen2d']=='D2B') & (dfM[merger]=='g')]),len(dfM[(dfM['transtypen2d']=='D2B') & (dfM[merger]=='orange')]),len(dfM[(dfM['transtypen2d']=='D2B') & (dfM[merger]=='red')])),\
             color=['g', 'orange', 'red'])
    ax[0,3].text(-1.3, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='D2B') & (dfM[merger]=='g')])/len(dfM[dfM['transtypen2d']=='D2B']))) +'%', fontsize=10, color='k')
    ax[0,3].text(-0.3, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='D2B') & (dfM[merger]=='orange')])/len(dfM[dfM['transtypen2d']=='D2B']))) +'%', fontsize=10, color='k')
    ax[0,3].text(0.7, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='D2B') & (dfM[merger]=='red')])/len(dfM[dfM['transtypen2d']=='D2B']))) +'%', fontsize=10, color='k')
   
    for id in BDB.ProjGalaxyID.unique():
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
        dfmergetemp=temp[temp[mergername]>0.]
        ax[1,4].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=5)
    ax[1,4].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    
    ax[0,4].bar((-1,0,1),\
        (len(dfM[(dfM['transtypen2d']=='BDB') & (dfM[merger]=='g')]),len(dfM[(dfM['transtypen2d']=='BDB') & (dfM[merger]=='orange')]),len(dfM[(dfM['transtypen2d']=='BDB') & (dfM[merger]=='red')])),\
             color=['g', 'orange', 'red'])
    ax[0,4].text(-1.3, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='BDB') & (dfM[merger]=='g')])/len(dfM[dfM['transtypen2d']=='BDB']))) +'%', fontsize=10, color='k')
    ax[0,4].text(-0.3, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='BDB') & (dfM[merger]=='orange')])/len(dfM[dfM['transtypen2d']=='BDB']))) +'%', fontsize=10, color='k')
    ax[0,4].text(0.7, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='BDB') & (dfM[merger]=='red')])/len(dfM[dfM['transtypen2d']=='BDB']))) +'%', fontsize=10, color='k')
   
    for id in DBD.ProjGalaxyID.unique():
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
        dfmergetemp=temp[temp[mergername]>0.]
        ax[1,5].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=5)

    ax[1,5].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    
    ax[0,5].bar((-1,0,1),\
        (len(dfM[(dfM['transtypen2d']=='DBD') & (dfM[merger]=='g')]),len(dfM[(dfM['transtypen2d']=='DBD') & (dfM[merger]=='orange')]),len(dfM[(dfM['transtypen2d']=='DBD') & (dfM[merger]=='red')])),\
             color=['g', 'orange', 'red'])
    ax[0,5].text(-1.3, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='DBD') & (dfM[merger]=='g')])/len(dfM[dfM['transtypen2d']=='DBD']))) +'%', fontsize=10, color='k')
    ax[0,5].text(-0.3, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='DBD') & (dfM[merger]=='orange')])/len(dfM[dfM['transtypen2d']=='DBD']))) +'%', fontsize=10, color='k')
    ax[0,5].text(0.7, 1, str(round(100*len(dfM[(dfM['transtypen2d']=='DBD') & (dfM[merger]=='red')])/len(dfM[dfM['transtypen2d']=='DBD']))) +'%', fontsize=10, color='k')
   
    
    legendelements=[]
    for color in df[merger].unique():
        temp=df[df[merger]==color]
        for colorname in temp[mergername2].unique():
            legendelements.append(Line2D([0],[0], marker='o', color='w', label=colorname, markerfacecolor=color, markersize=10))
    ax[1,0].legend(handles=legendelements, bbox_to_anchor=(-0.2, 1.1))
    
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
    cbar=plt.colorbar(sm, cax=cbar_ax).set_label(r'residual of sSFR per mass per z [$M_{\odot} Gyr^{-1}$]')
    if merge==True:
        plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Evolution of'+str(param)+'thresh'+str(thresh)+'zmax'+str(maxz)+'colorby'+str(mergername)+'merge.png')
    else:
        plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Evolution of'+str(param)+'thresh'+str(thresh)+'zmax'+str(maxz)+'colorby'+str(mergername)+'.png')
    plt.show()
    plt.close()

def binvalue(df, paramx, paramy, binno):
    #produce median, binedges, lowquart, uppquart, std for a parameter
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
    #plot moving quantiles, highlighted by extreme values in binparam
    df['zrounded']=df.apply(lambda x: np.round(x.z, decimals=1), axis=1)
    z0df=df[df.zrounded==0.]
    df=df[(df.zrounded==0.) | (df.zrounded==0.2) | (df.zrounded==0.5) | (df.zrounded==1.0)]
    z0df=z0df[['ProjGalaxyID', binparam]]
    z0df['marker_bin']=pd.qcut(z0df[binparam], 20, labels=['10','20','30','1','2','3','4','5','6','7','8','9','11','40','50','60','70','80','90','100'])
    df=pd.merge(df, z0df, on=['ProjGalaxyID'], how='left',  suffixes=('','_proj'))
    
    fig, axs =plt.subplots(4, 2, sharex=True, sharey=True, figsize=(9,6))
    fig.suptitle('Time evolution of '+paramx+paramy+' showing distribution of '+binparam)
    axs[0,0].set_title('10th percentile of '+binparam)
    axs[0,1].set_title('90th percentile of '+binparam)
    for i,zi in enumerate([0., 0.2, 0.5, 1.0]):
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
    #prodcue colouring scheme for transition diagrams.
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
    #produce GTC multivariate plot, highlighted with constant discs
    names=['n2d', 'logsSFR','logmass', 'logBHmass', 'logDMmass', 'loggasmass', 'logSFThermalEnergy', 'logsSFMass', 'logDMHalfMassRad']
    labels=['n2d', 'logsSFR','logM', r'log$M_{BH}$', r'log$M_{halo}$', r'log$M_{Gas}$', r'log$E_{gas,SF}$', r'log$\frac{M_{gas,SF}}{M}$', r'log$R_{halo, halfmass}$']
    truths=df[df.transtypen2d=='D2D']
    df=df[names]
    truths=truths[names]
    df=df.dropna()
    truths=truths.dropna()
    
    print(df[names])
    GTC = pygtc.plotGTC(df, paramNames=labels)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/evolvingGTCplot.png')
    plt.show()

def categorybarchart(df, cat, catl, catu):
    #plot barchart of STATIC galaxy categorisations
    zlist=[]
    alist=[]
    blist=[]
    dlist=[]

    alistl=[]
    blistl=[]
    dlistl=[]

    alistu=[]
    blistu=[]
    dlistu=[]

    lbtlist=[]

    for i in df.zrounded.unique():
        tempdf=df[df.zrounded==i]
        Anum=len(tempdf[tempdf[cat]=='A'])
        Bnum=len(tempdf[tempdf[cat]=='B'])
        Dnum=len(tempdf[tempdf[cat]=='D'])

        Anuml=len(tempdf[tempdf[catl]=='A'])
        Bnuml=len(tempdf[tempdf[catl]=='B'])
        Dnuml=len(tempdf[tempdf[catl]=='D'])

        Anumu=len(tempdf[tempdf[catu]=='A'])
        Bnumu=len(tempdf[tempdf[catu]=='B'])
        Dnumu=len(tempdf[tempdf[catu]=='D'])

        zlist.append(i)
        alist.append(Anum)
        blist.append(Bnum)
        dlist.append(Dnum)

        alistl.append(np.abs(Anum-Anuml))
        blistl.append(np.abs(Bnum-Bnuml))
        dlistl.append(np.abs(Dnum-Dnuml))

        alistu.append(np.abs(Anumu-Anum))
        blistu.append(np.abs(Bnumu-Bnum))
        dlistu.append(np.abs(Dnumu-Dnum))
        for j in tempdf.lbt.unique():
            lbtlist.append(-j)
    zarr=np.array(zlist)
    aarr=np.array(alist)
    barr=np.array(blist)
    darr=np.array(dlist)

    aarru=np.array(alistu)
    barru=np.array(blistu)
    darru=np.array(dlistu)

    aarrl=np.array(alistl)
    barrl=np.array(blistl)
    darrl=np.array(dlistl)

    lbtarr=np.array(lbtlist)
    print(zlist, alist, blist, dlist, lbtlist)
    width=0.09
    fig, ax =plt.subplots()
    ax0=ax.twiny()
    ax.bar(zarr +width/3, aarr, width/3, yerr=[aarrl,aarru], label='Asymmetric', color='green', ecolor='darkgreen',  capsize=0.5, alpha=0.5)
    ax.bar(zarr + 2*width/3, barr, width/3, yerr=[barrl,barru],label='Bulge',  color='red', ecolor='darkred',  capsize=0.5,  alpha=0.5)
    ax.bar(zarr + width, darr, width/3, yerr=[darrl,darru],label='Disc',  color='blue', ecolor='darkblue',  capsize=0.5,  alpha=0.5)

    
    ax.set_xlim(0,2.8)
    #ax0.cla()
    ax0.set_xlim(ax.get_xlim())
    #ax0.set_xticks(ax.get_xticks())
    ax0.set_xticklabels([lbtarr[15],lbtarr[10], lbtarr[6], lbtarr[4],lbtarr[2], lbtarr[0] ])
    ax0.set_xlabel('Lookback time [Gyr]')
    
    #ax0.set_xlim(df.lbt.max(), df.lbt.min())
    ax.legend()
    ax.set_xlabel('Redshift (z)')
    ax.set_ylabel(r'no. of galaxies by $n_{2d}$')
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/barchart'+cat+'.png')
    plt.show()

def calccatfrac2(cat, catfrac, typ, colormin):
    #calculate the fraction in each category per mass per redshift
    if cat == typ:
        return catfrac
    else:
        return colormin

def plotfrac(df, y, cat, color):
    #plot digitised scatter plot, as in Trayford, 2018.
    fig, ax =plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9,5))
    
    dfA=df.copy()
    dfB=df.copy()
    dfD=df.copy()
    colormin=df[color].min()
    dfA['catfrac2']=dfA.apply(lambda x: calccatfrac2(x.categoryn2d, x.catfrac, 'A', colormin), axis=1)
    dfB['catfrac2']=dfB.apply(lambda x: calccatfrac2(x.categoryn2d, x.catfrac, 'B', colormin), axis=1)
    dfD['catfrac2']=dfD.apply(lambda x: calccatfrac2(x.categoryn2d, x.catfrac, 'D', colormin), axis=1)
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
    plt.xlim(df.z.min(), df.z.max()+0.1), plt.ylim(10,df[y].max() + 0.2)
    ax[0].set_ylabel(r'$log(M_{*})$ [$M_{\odot}$]')
    plt.subplots_adjust(right=0.8, wspace=0, hspace=0)
    cbar_ax=fig.add_axes([0.8,0.11,0.05,0.77])
    sm=plt.cm.ScalarMappable(cmap=Cmap, norm=Norm)
    sm.set_array([])
    cbar=plt.colorbar(sm, cax=cbar_ax).set_label(r'$M_{*}$ fraction in each component')
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/evolvingfrac'+y+''+cat+'colouredby'+color+'.png')
    plt.show()

def colourscatter(df,x,y, column_colour, thresh):
    #produce coloured scatter diagram with PDF distributions
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
    n=15
    axleft.set_xlabel('PDF')
    axtop.set_ylabel('PDF')
    axleft.set_ylabel(r'$log(M_{gas} / M_{*})$')
    ax1.set_xlabel(r'$z$')

    ax1.scatter(df[x],df[y], c=df[column_colour], cmap=Cmap, norm=Norm, alpha=0.5, s=10)
    ax1.set_ylim(-4, 0.5)
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
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/plot'+x+''+y+'colouredby'+column_colour+'.png')
    plt.show()

def plotmergerrate(df, cat, time, mass, mergermassfrac):
    #plot merger rates - IN PROGRESS
    #df1=df[[time, cat]].groupby([time, cat]).size().unstack(fill_value=0)
    #PART 1
    df=df[df[time]>0]
    df11=df[[time, cat, mass, mergermassfrac]].groupby([time, cat]).agg({mass:['median','std'], mergermassfrac:['median','std'], cat:['count']})
    df11.columns=df11.columns.get_level_values(0)
    df11.columns=pd.io.parsers.ParserBase({'names':df11.columns})._maybe_dedup_names(df11.columns)
    df11=df11.rename({mass:'massmedian', mass+'.1':'massstd', mergermassfrac:'mergermassfracmedian', mergermassfrac+'.1':'mergermassfracstd', cat:'catcount'}, axis=1)
    df11=df11.reset_index(level=[0,1])

    df12=df[[time]].groupby([time]).agg({time:['count']})
    df12=df12.xs(time, axis=1, drop_level=True)
    df12=df12.rename({'count':'zcount'}, axis=1)
    df13=pd.merge(df11, df12, on=[time], how='left')
    df13['frac']=df13.apply(lambda x: x.catcount/x.zcount, axis=1)

    fig, ax= plt.subplots()
    accretiondf=df13[df13[cat]=='accretion']
    majordf=df13[df13[cat]=='major']
    minordf=df13[df13[cat]=='minor']
    ax.plot(majordf[time], majordf.frac, c='r', label=r'major ($\mu > 0.25$)')
    ax.plot(minordf[time], minordf.frac, c='b', label=r'major ($\mu > 0.1$)')
    ax.plot(accretiondf[time], accretiondf.frac, c='y', label='accretion')
    ax.legend()
    ax.set_xscale('log'), ax.set_yscale('log')
    formatter=mticker.ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(formatter)
    ax.set_ylabel(r'$\frac{dN_{Merger}}{dN_{Total} dt}  [Gyr]^{-1}$ ')
    ax.set_xlabel('LookBackTime [Gyr]')
    ax.set_xlim([df[time].min(),df[time].max()])
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/plotmergerrate'+ cat+time+'.png')
    plt.show()

    df['logmergermassfrac']=df.apply(lambda x: logx(x.Starmassmergerfrac), axis=1)
    df['time2']=df.apply(lambda x: -np.round(x.lbt/2, decimals=0)*2, axis=1)
    df['logmergermassfrac']=df.apply(lambda x: (roundx(x.logmergermassfrac/5, dec=1))*5, axis=1)

    df21=df[['time2', cat, mass, 'logmergermassfrac']].groupby(['logmergermassfrac', 'time2']).agg({mass:['median','std'], 'logmergermassfrac':['count']})
    df21.columns=df21.columns.get_level_values(0)
    df21.columns=pd.io.parsers.ParserBase({'names':df21.columns})._maybe_dedup_names(df21.columns)
    df21=df21.rename({mass:'massmedian', mass+'.1':'massstd', 'logmergermassfrac':'mergermassfraccount'}, axis=1)
    df21=df21.reset_index(level=[0,1])

    df22=df[['time2']].groupby(['time2']).agg({'time2':['count']})
    df22=df22.xs('time2', axis=1, drop_level=True)
    df22=df22.rename({'count':'zcount'}, axis=1)
    df23=pd.merge(df21, df22, on=['time2'], how='left')
    df23['frac']=df23.apply(lambda x: x.mergermassfraccount/x.zcount, axis=1)
    print(df23)
    
    fig, ax= plt.subplots()

    ninedf=df23[df23['time2']>8.1]
    sevendf=df23[(df23['time2']>6.1)& (df23['time2']>8.1)]
    fivedf=df23[(df23['time2']>4.1)& (df23['time2']<6.1)]
    threedf=df23[(df23['time2']>2.1)& (df23['time2']<4.1)]
    onedf=df23[df23['time2']<2.1]

    ax.plot(onedf['logmergermassfrac'], onedf.frac, c='r', label=r'$lbt \leq 2 [Gyr]$')
    ax.plot(threedf['logmergermassfrac'], threedf.frac, c='deeppink', label=r'$3 \leq lbt \leq 4 [Gyr]$')
    ax.plot(fivedf['logmergermassfrac'], fivedf.frac, c='darkviolet', label=r'$5 \leq lbt \leq 6 [Gyr]$')
    ax.plot(sevendf['logmergermassfrac'], sevendf.frac, c='b', label=r'$7 \leq lbt \leq 8 [Gyr]$')
    ax.plot(ninedf['logmergermassfrac'], ninedf.frac, c='dodgerblue', label=r'$1 \leq 9 lbt [Gyr]$')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylabel(r'$\frac{dN_{Merger}}{dN_{Total} dt}  [Gyr]^{-1}$ ')
    ax.set_xlabel(r'Log(Mass Ratio) log($\mu$)')
    ax.set_xlim([df23['logmergermassfrac'].min(),df23['logmergermassfrac'].max()])
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/plotmergerratepermassratio.png')
    plt.show()
    
    #PART 2

    df['logmergermassfrac']=df.apply(lambda x: logx(x.Starmassmergerfrac), axis=1)
    df['time2']=df.apply(lambda x: -np.round(x.lbt/2, decimals=0)*2, axis=1)
    df['logmergermassfrac']=df.apply(lambda x: (roundx(x.logmergermassfrac/5, dec=1))*5, axis=1)
    df['roundlogmass']=df.apply(lambda x: (roundx(x.logmass/1, dec=1))*1, axis=1)

    df21=df[['time2', cat, mass, 'logmergermassfrac', 'roundlogmass']].groupby(['time2','logmergermassfrac', 'roundlogmass']).agg({mass:['median','std'], 'logmergermassfrac':['count']})
    df21.columns=df21.columns.get_level_values(0)
    df21.columns=pd.io.parsers.ParserBase({'names':df21.columns})._maybe_dedup_names(df21.columns)
    df21=df21.rename({mass:'massmedian', mass+'.1':'massstd', 'logmergermassfrac':'mergermassfraccount'}, axis=1)
    df21=df21.reset_index(level=[0,1])
    df22=df[['time2', 'roundlogmass']].groupby(['time2', 'roundlogmass']).agg({'time2':['count']})
    df22=df22.xs('time2', axis=1, drop_level=True)
    df22=df22.rename({'count':'zcount'}, axis=1)
    df23=pd.merge(df21, df22, on=['time2', 'roundlogmass'], how='left')
    df23=df23.reset_index()
    df23['frac']=df23.apply(lambda x: x.mergermassfraccount/x.zcount, axis=1)

    fig, ax= plt.subplots(2,4, sharey='row', figsize=(12, 6))

    twodf=df23[df23['time2']==2.0]
    fourdf=df23[df23['time2']==4.0]
    sixdf=df23[df23['time2']==6.0]
    eightdf=df23[df23['time2']==8.0]

    for i, dff in enumerate([twodf,fourdf, sixdf, eightdf]):
        accretiondf=dff[(dff['logmergermassfrac']<logx(0.01))& (dff['logmergermassfrac']>logx(0.001))].sort_values(by=['roundlogmass'])
        accretion2df=dff[dff['logmergermassfrac']<logx(0.001)].sort_values(by=['roundlogmass'])
        print(accretiondf)
        majordf=dff[dff['logmergermassfrac']>logx(0.25)].sort_values(by=['roundlogmass'])
        minordf=dff[(dff['logmergermassfrac']<logx(0.25)) & (dff['logmergermassfrac']>logx(0.01))].sort_values(by=['roundlogmass'])
        
        ax[0,i].plot(majordf['roundlogmass'], majordf.frac,c='green', label=r'major ($\mu > 0.25$)')
        ax[0,i].plot(minordf['roundlogmass'], minordf.frac,c='limegreen', label=r'minor ($\mu > 0.1$)')
        ax[0,i].plot(accretiondf['roundlogmass'], accretiondf.frac,c='turquoise', label=r'accretion ($\mu < 0.01$)')
        ax[0,i].plot(accretion2df['roundlogmass'], accretion2df.frac,c= 'yellow', label=r'accretion ($\mu < 0.001$)')
        ax[0,3].legend(bbox_to_anchor=(1.1, 0.5))

        highmass=dff[dff['roundlogmass']> 10.7].sort_values(by=['logmergermassfrac'])
        midmass=dff[dff['roundlogmass']< 10.2].sort_values(by=['logmergermassfrac'])
        lowmass=dff[(dff['roundlogmass']> 10.2) & (dff['roundlogmass']< 10.7)].sort_values(by=['logmergermassfrac'])

        ax[1,i].plot(highmass['logmergermassfrac'], highmass.frac,c='r', label=r'$M_{*} > 10^{10.7}$')
        ax[1,i].plot(midmass['logmergermassfrac'], midmass.frac,c='magenta', label=r'$M_{*} > 10^{10.2}$')
        ax[1,i].plot(lowmass['logmergermassfrac'], lowmass.frac,c='darkviolet', label=r'$M_{*} < 10^{10.2}$')
        ax[1,3].legend(bbox_to_anchor=(1.1, 0.5))
        ax[0,i].set_title(r'Look Back Time $\approx$'+str((2*i) +2))
    
    plt.subplots_adjust(right=0.7, wspace=0, hspace=0.5)
    ax[0,0].set_yscale('log')
    ax[1,0].set_yscale('log')
    ax[0,0].set_ylabel(r'$\frac{dN_{Merger}}{dN_{Total} dt}  [Gyr]^{-1}$ ')
    ax[1,0].set_ylabel(r'$\frac{dN_{Merger}}{dN_{Total} dt}  [Gyr]^{-1}$ ')
    ax[0,0].set_xlabel(r'Log(Mass) log($M_{\odot}$)')

    #ax[].set_xlim([df23['logmergermassfrac'].min(),df23['logmergermassfrac'].max()])
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/plotmergerratepermassratio.png')
    plt.show()

def plotmergergasrate(df, cat, time):
    #plot gas merger rates -- IN PROGRESS
    #df1=df[['z', 'mergercategory']].groupby(['z', 'mergercategory']).agg({'mergercategory':['count']})
    #df1=df1.xs('mergercategory', axis=1, drop_level=True)
    df1=df[[time, cat]].groupby([time, cat]).size().unstack(fill_value=0)
    df1[time]=df1.index
    df1=df1.reset_index(drop=True)
    print(df1)
    df2=df[[time]].groupby([time]).agg({time:['count']})
    df2=df2.xs(time, axis=1, drop_level=True)
    df2=df2.rename({'count':'zcount'}, axis=1)
    print(df2)
    df3=pd.merge(df1, df2, on=[time], how='left')
    df3['accretionfrac']=df3.apply(lambda x: x.undefined/x.zcount, axis=1)
    df3['majorfrac']=df3.apply(lambda x: x.gasrich/x.zcount, axis=1)
    df3['minorfrac']=df3.apply(lambda x: x.gaspoor/x.zcount, axis=1)
    print(df3)
    plt.plot(df3[time], df3.majorfrac, c='r', label='gasrich')
    plt.plot(df3[time], df3.minorfrac, c='b', label='gaspoor')
    plt.plot(df3[time], df3.accretionfrac, c='y', label='undefined')
    plt.legend()
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/plotmergerrate'+ cat+time+'.png')
    plt.show()

def plotSFRcircles(df, histparam1, histparam2, histparam3, histparam4,histparam5,histparam6, binparam, vmin, vmax, hparam, thresh, threshstep, trans, label, label2):
    #plot SFR apertures, dispalying constant discs, bulges and transitions between the two
    z0df=df[df.zrounded==0.]
    df=df[(df.zrounded==0.1) | (df.zrounded==0.5) | (df.zrounded==1.0) | (df.zrounded==1.5)]
    z0df=z0df[['ProjGalaxyID', binparam]]
    z0df['marker_bin']=pd.qcut(z0df[binparam],8, labels=['10','30','40','50','70','80','90','100'])
    df=pd.merge(df, z0df, on=['ProjGalaxyID'], how='left',  suffixes=('','_proj'))

    fig, axs =plt.subplots(12, 5, figsize=(6,10), gridspec_kw={'width_ratios':[4,4,4,4,0.5], 'height_ratios':[4,1.5,1,4,1.5,1,4,1.5,1,4,1.5,1]})
    fig.suptitle('Time evolution of historgram of '+histparam1+' showing distribution of '+binparam)
    axs[0,0].set_title('Constant Discs')
    axs[0,1].set_title('B->D')
    axs[0,2].set_title('D->B')
    axs[0,3].set_title('Constant Bulges')
    Cmap=mcol.LinearSegmentedColormap.from_list("cmop", ['red','mediumorchid','blue'])
    #Cmap='seismic'
    norm=plt.Normalize(vmin,vmax)

    for i1,zi in enumerate([0.1, 0.5, 1.0, 1.5]):
        i=3*i1
        i2=i+1
        i3=i+2
        zdf=df[df.zrounded==zi]
        #lowdf=zdf[zdf.marker_bin=='10']
        Ddf=zdf[zdf[trans]=='D2D']
        B2Ddf=zdf[(zdf[trans]=='B2D')| (zdf[trans]=='DBD')]
        #B2Ddf=zdf[zdf.marker_bin=='10']
        #B2ddf=zdf
        D2Bdf=zdf[(zdf[trans]=='D2B')| (zdf[trans]=='BDB')]
        #D2Bdf=zdf[zdf.marker_bin=='100']
        #highdf=zdf[zdf.marker_bin=='100']
        Bdf=zdf[zdf[trans]=='B2B']
        minn=min([Ddf[histparam1].median(),Ddf[histparam2].median(),Ddf[histparam3].median(),Ddf[histparam4].median(),Ddf[histparam5].median(),Ddf[histparam6].median(),B2Ddf[histparam1].median(),B2Ddf[histparam2].median(),B2Ddf[histparam3].median(),B2Ddf[histparam4].median(),B2Ddf[histparam5].median(),B2Ddf[histparam6].median(),D2Bdf[histparam1].median(),D2Bdf[histparam2].median(),D2Bdf[histparam3].median(),D2Bdf[histparam4].median(),D2Bdf[histparam5].median(),D2Bdf[histparam6].median(), Bdf[histparam1].median(),Bdf[histparam2].median(),Bdf[histparam3].median(),Bdf[histparam4].median(),Bdf[histparam5].median(),Bdf[histparam6].median()])
        maxx=max([Ddf[histparam1].median(),Ddf[histparam2].median(),Ddf[histparam3].median(),Ddf[histparam4].median(),Ddf[histparam5].median(),Ddf[histparam6].median(),B2Ddf[histparam1].median(),B2Ddf[histparam2].median(),B2Ddf[histparam3].median(),B2Ddf[histparam4].median(),B2Ddf[histparam5].median(),B2Ddf[histparam6].median(),D2Bdf[histparam1].median(),D2Bdf[histparam2].median(),D2Bdf[histparam3].median(),D2Bdf[histparam4].median(),D2Bdf[histparam5].median(),D2Bdf[histparam6].median(), Bdf[histparam1].median(),Bdf[histparam2].median(),Bdf[histparam3].median(),Bdf[histparam4].median(),Bdf[histparam5].median(),Bdf[histparam6].median()])
        norm=plt.Normalize(minn, maxx)
        sm=plt.cm.ScalarMappable(cmap=Cmap, norm=norm)
        sm.set_array([])
        cbar=fig.colorbar(sm, cax=axs[i,4]).set_label(label)
        print(i1,i,i2,i3)
        binedgs1=np.linspace(df[hparam].min(), df[hparam].max(), 30)
        
        for j,df2 in enumerate([Ddf, B2Ddf, D2Bdf, Bdf]):
            axs[i,j].set_xlim(-30, 30)
            axs[i,j].set_ylim(-30, 30)
            axs[i3,j].remove()
            axs[i3,4].set_visible(False)
            axs[i2,4].set_visible(False)        
            
            #Cmap='seismic'
            color1=df2[histparam1].median()
            color3=df2[histparam2].median()
            color5=df2[histparam3].median()
            color10=df2[histparam4].median()
            color20=df2[histparam5].median()
            color30=df2[histparam6].median()

            circle1=plt.Circle((0,0), 2, color=Cmap(norm(color1)))
            circle3=plt.Circle((0,0), 5, color=Cmap(norm(color3)))
            circle5=plt.Circle((0,0), 8, color=Cmap(norm(color5)))
            circle10=plt.Circle((0,0), 14, color=Cmap(norm(color10)))
            circle20=plt.Circle((0,0), 22, color=Cmap(norm(color20)))
            circle30=plt.Circle((0,0), 30, color=Cmap(norm(color30)))
            axs[i,j].add_artist(circle30)
            axs[i,j].add_artist(circle20)
            axs[i,j].add_artist(circle10)
            axs[i,j].add_artist(circle5)
            axs[i,j].add_artist(circle3)
            axs[i,j].add_artist(circle1)
            axs[i,j].set_xlabel('')
            axs[i,j].set_ylabel('')
            
            axs[i2,j].hist(zdf[hparam], color="k", alpha=0.4,label='z='+str(zi), histtype='stepfilled', bins=binedgs1)
            axs[i2,0].hist(Ddf[hparam], color="b", alpha=0.5, histtype='step', bins=binedgs1)
            axs[i2,1].hist(B2Ddf[hparam], color="purple", alpha=0.5, histtype='step', bins=binedgs1)
            axs[i2,2].hist(D2Bdf[hparam], color="purple", alpha=0.5, histtype='step', bins=binedgs1)
            axs[i2,3].hist(Bdf[hparam], color="r", alpha=0.5, histtype='step', bins=binedgs1)
            axs[10,j].set_xlabel(label2)

            axs[i,j].set_xticks([])
            axs[i,1].set_yticks([])
            axs[i,2].set_yticks([])
            axs[i,3].set_yticks([])
            axs[i2,1].set_yticks([])
            axs[i2,2].set_yticks([])
            axs[i2,3].set_yticks([])

        axs[i2,0].set_ylabel('z='+str(zi))
        axs[i,0].set_ylabel('Extent')

    plt.subplots_adjust(right=0.8, wspace=0, hspace=0)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/'+histparam4+'circleplot4'+binparam+'.png')
    plt.show()

def plotSFRcircles3(df, histparam1, histparam2, histparam3, histparam4,histparam5,histparam6, binparam, vmin, vmax, hparam, thresh, threshstep, trans, label1, label2):
    #plot SFR aperture diagram dispaying 10th and 90th percentiles of bulges/discs
    z0df=df[df.zrounded==0.]
    df=df[(df.zrounded==0.1) | (df.zrounded==0.5) | (df.zrounded==1.0) | (df.zrounded==1.5)]
    z0df=z0df[['ProjGalaxyID', binparam]]
    z0df['marker_bin']=pd.qcut(z0df[binparam],8, labels=['10','30','40','50','70','80','90','100'])
    df=pd.merge(df, z0df, on=['ProjGalaxyID'], how='left',  suffixes=('','_proj'))

    fig, axs =plt.subplots(12, 4, figsize=(6,10), gridspec_kw={'width_ratios':[4,4,4,0.5], 'height_ratios':[4,1.5,1,4,1.5,1,4,1.5,1,4,1.5,1]})
    fig.suptitle('Time evolution of historgram of '+histparam1+' showing distribution of '+binparam)
    axs[0,0].set_title('10th')
    axs[0,1].set_title('Median')
    axs[0,2].set_title('90th')
 
    Cmap=mcol.LinearSegmentedColormap.from_list("cmop", ['red','mediumorchid','blue'])
    norm=plt.Normalize(vmin,vmax)

    for i1,zi in enumerate([0.1, 0.5, 1.0, 1.5]):
        i=3*i1
        i2=i+1
        i3=i+2
        zdf=df[df.zrounded==zi]
        Adf=zdf[zdf.marker_bin=='10']
        Bdf=zdf[zdf.marker_bin=='50']
        Cdf=zdf[zdf.marker_bin=='100']
        minn=min([Adf[histparam1].median(),Adf[histparam2].median(),Adf[histparam3].median(),Adf[histparam4].median(),Adf[histparam5].median(),Adf[histparam6].median(),Bdf[histparam1].median(),Bdf[histparam2].median(),Bdf[histparam3].median(),Bdf[histparam4].median(),Bdf[histparam5].median(),Bdf[histparam6].median(),Cdf[histparam1].median(),Cdf[histparam2].median(),Cdf[histparam3].median(),Cdf[histparam4].median(),Cdf[histparam5].median(),Cdf[histparam6].median()])
        maxx=max([Adf[histparam1].median(),Adf[histparam2].median(),Adf[histparam3].median(),Adf[histparam4].median(),Adf[histparam5].median(),Adf[histparam6].median(),Bdf[histparam1].median(),Bdf[histparam2].median(),Bdf[histparam3].median(),Bdf[histparam4].median(),Bdf[histparam5].median(),Bdf[histparam6].median(),Cdf[histparam1].median(),Cdf[histparam2].median(),Cdf[histparam3].median(),Cdf[histparam4].median(),Cdf[histparam5].median(),Cdf[histparam6].median()])
        norm=plt.Normalize(minn, maxx)
        sm=plt.cm.ScalarMappable(cmap=Cmap, norm=norm)
        sm.set_array([])
        cbar=fig.colorbar(sm, cax=axs[i,3]).set_label(label1)
        print(i1,i,i2,i3)
        binedgs1=np.linspace(df[hparam].min(), df[hparam].max(), 30)
        
        for j,df2 in enumerate([Adf, Bdf, Cdf]):
            axs[i,j].set_xlim(-30, 30)
            axs[i,j].set_ylim(-30, 30)
            axs[i3,j].remove()
            axs[i3,3].set_visible(False)
            axs[i2,3].set_visible(False)        
            
            #Cmap='seismic'
            color1=df2[histparam1].median()
            color3=df2[histparam2].median()
            color5=df2[histparam3].median()
            color10=df2[histparam4].median()
            color20=df2[histparam5].median()
            color30=df2[histparam6].median()

            circle1=plt.Circle((0,0), 2, color=Cmap(norm(color1)))
            circle3=plt.Circle((0,0), 5, color=Cmap(norm(color3)))
            circle5=plt.Circle((0,0), 8, color=Cmap(norm(color5)))
            circle10=plt.Circle((0,0), 14, color=Cmap(norm(color10)))
            circle20=plt.Circle((0,0), 22, color=Cmap(norm(color20)))
            circle30=plt.Circle((0,0), 30, color=Cmap(norm(color30)))
            axs[i,j].add_artist(circle30)
            axs[i,j].add_artist(circle20)
            axs[i,j].add_artist(circle10)
            axs[i,j].add_artist(circle5)
            axs[i,j].add_artist(circle3)
            axs[i,j].add_artist(circle1)
            axs[i,j].set_xlabel('')
            axs[i,j].set_ylabel('')
            
            axs[i2,j].hist(zdf[hparam], color="k", alpha=0.4,label='z='+str(zi), histtype='stepfilled', bins=binedgs1)
            axs[i2,0].hist(Adf[hparam], color="b", alpha=0.5, histtype='step', bins=binedgs1)
            axs[i2,1].hist(Bdf[hparam], color="purple", alpha=0.5, histtype='step', bins=binedgs1)
            axs[i2,2].hist(Cdf[hparam], color="r", alpha=0.5, histtype='step', bins=binedgs1)
            axs[10,j].set_xlabel(label2)

            axs[i,j].set_xticks([])
            axs[i,1].set_yticks([])
            axs[i,2].set_yticks([])
            axs[i2,1].set_yticks([])
            axs[i2,2].set_yticks([])

        axs[i2,0].set_ylabel('z='+str(zi))
        axs[i,0].set_ylabel('Extent')

    plt.subplots_adjust(right=0.8, wspace=0, hspace=0)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/'+histparam4+'circleplot3'+binparam+'.png')
    plt.show()

def evolutionplot3(df, param0, param1, param2, colorparam, thresh, merger, mergername, mergername2):
    #plot evolution diagram for 3 parameters
    fig, ax =plt.subplots(3, 4, sharey='row',sharex=True, figsize=(12,6))
    Cmap=mcol.LinearSegmentedColormap.from_list("cmop", ['tomato','cornflowerblue'])
    Norm=plt.Normalize(df[colorparam].min(),df[colorparam].max())
    B2B=df[df.transtypen2d=='B2B']
    D2D=df[df.transtypen2d=='D2D']
    B2D=df[df.transtypen2d=='B2D']
    D2B=df[df.transtypen2d=='D2B']

    for id in B2B.ProjGalaxyID.unique():
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        t=temp[colorparam].values
        for i, param in enumerate([param0, param1, param2]):
            y=temp[param].values
            points=np.array([x,y]).T.reshape(-1,1,2)
            segments=np.concatenate([points[:-1], points[1:]], axis=1)
            lc=LineCollection(segments, cmap=Cmap, norm=Norm)
            lc.set_array(t)
            lc.set_linewidth(0.5)
            ax[i,0].add_collection(lc)
            dfmergetemp=temp[temp[mergername]>0.]
            ax[i,0].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=3)
    ax[0,0].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'k--', linewidth=1)
    ax[1,0].plot([df.lbt.min(),df.lbt.max()], [12.2,12.2], 'k--', linewidth=1)
    #ax[2,0].plot([df.lbt.min(),df.lbt.max()], [0,0], 'k--', linewidth=1)

    for id in D2D.ProjGalaxyID.unique():
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        t=temp[colorparam].values
        for i, param in enumerate([param0, param1, param2]):
            y=temp[param].values
            points=np.array([x,y]).T.reshape(-1,1,2)
            segments=np.concatenate([points[:-1], points[1:]], axis=1)
            lc=LineCollection(segments, cmap=Cmap, norm=Norm)
            lc.set_array(t)
            lc.set_linewidth(0.5)
            ax[i,1].add_collection(lc)
            dfmergetemp=temp[temp[mergername]>0.]
            ax[i,1].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=3)
    ax[0,1].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'k--', linewidth=1)
    ax[1,1].plot([df.lbt.min(),df.lbt.max()], [12.2,12.2], 'k--', linewidth=1)
    #ax[2,1].plot([df.lbt.min(),df.lbt.max()], [0,0], 'k--', linewidth=1)

    for id in B2D.ProjGalaxyID.unique():
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        t=temp[colorparam].values
        for i, param in enumerate([param0, param1, param2]):
            y=temp[param].values
            points=np.array([x,y]).T.reshape(-1,1,2)
            segments=np.concatenate([points[:-1], points[1:]], axis=1)
            lc=LineCollection(segments, cmap=Cmap, norm=Norm)
            lc.set_array(t)
            lc.set_linewidth(0.5)
            ax[i,2].add_collection(lc)
            dfmergetemp=temp[temp[mergername]>0.]
            ax[i,2].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=3)
    ax[0,2].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'k--', linewidth=1)
    ax[1,2].plot([df.lbt.min(),df.lbt.max()], [12.2,12.2], 'k--', linewidth=1)
    #ax[2,2].plot([df.lbt.min(),df.lbt.max()], [0,0], 'k--', linewidth=1)

    for id in D2B.ProjGalaxyID.unique():
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        t=temp[colorparam].values
        for i, param in enumerate([param0, param1, param2]):
            y=temp[param].values
            points=np.array([x,y]).T.reshape(-1,1,2)
            segments=np.concatenate([points[:-1], points[1:]], axis=1)
            lc=LineCollection(segments, cmap=Cmap, norm=Norm)
            lc.set_array(t)
            lc.set_linewidth(0.5)
            ax[i,3].add_collection(lc)
            dfmergetemp=temp[temp[mergername]>0.]
            ax[i,3].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=3)
    ax[0,3].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'k--', linewidth=1)
    ax[1,3].plot([df.lbt.min(),df.lbt.max()], [12.2,12.2], 'k--', linewidth=1)
    #ax[2,3].plot([df.lbt.min(),df.lbt.max()], [0,0], 'k--', linewidth=1)
    
    legendelements=[]
    for color in df[merger].unique():
        temp=df[df[merger]==color]
        for colorname in temp[mergername2].unique():
            legendelements.append(Line2D([0],[0], marker='o', color='w', label=colorname, markerfacecolor=color, markersize=10))
    ax[1,0].legend(handles=legendelements, bbox_to_anchor=(-0.2, 1.1))
    
    ax[0,0].set_title('B'),ax[0,1].set_title('D'),ax[0,2].set_title('B2D'), ax[0,3].set_title('D2B')
    ax[2,2].set_xlabel('look back time (Gyr)')
    ax[0,0].set_ylabel(r'n2d'), ax[1,0].set_ylabel(r'log($M_{halo})$ [$M_{\odot}$]'), ax[2,0].set_ylabel(r'log($\frac{M_{halo}}{dt})$ [$M_{\odot}Gyr^{-1}$]')
    ax[0,0].set_xlim(df.lbt.min(), df.lbt.max())
    plt.subplots_adjust(right=0.8, wspace=0.1, hspace=0)
    cbar_ax=fig.add_axes([0.85,0.15,0.05,0.8])
    sm=plt.cm.ScalarMappable(cmap=Cmap, norm=Norm)
    sm.set_array([])
    cbar=plt.colorbar(sm, cax=cbar_ax).set_label(r'residual of sSFR per mass per z [$M_{\odot} Gyr^{-1}$]')
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Evolution of'+str(param0)+str(param1)+str(param2)+str(mergername)+'.png')
    plt.show()
    plt.close()

def evolutionplot4(df, param0, name0, param1,name1, param2, name2,param3,name3, colorparam, thresh, merger, mergername, mergername2):
    #plot evolution diagram for 4 parameters
    fig, ax =plt.subplots(4, 4, sharey='row',sharex=True, figsize=(12,6))
    Cmap=mcol.LinearSegmentedColormap.from_list("cmop", ['tomato','cornflowerblue'])
    Norm=plt.Normalize(df[colorparam].min(),df[colorparam].max())
    B2B=df[df.transtypen2d=='B2B']
    D2D=df[df.transtypen2d=='D2D']
    B2D=df[df.transtypen2d=='B2D']
    D2B=df[df.transtypen2d=='D2B']


    for id in B2B.ProjGalaxyID.unique():
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        t=temp[colorparam].values
        for i, param in enumerate([param0, param1, param2, param3]):
            y=temp[param].values
            points=np.array([x,y]).T.reshape(-1,1,2)
            segments=np.concatenate([points[:-1], points[1:]], axis=1)
            lc=LineCollection(segments, cmap=Cmap, norm=Norm)
            lc.set_array(t)
            lc.set_linewidth(0.5)
            ax[i,0].add_collection(lc)
            dfmergetemp=temp[temp[mergername]>0.]
            ax[i,0].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=3)
    ax[0,0].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'k--', linewidth=1)
    ax[1,0].plot([df.lbt.min(),df.lbt.max()], [5.1,5.1], 'k--', linewidth=1)
    #ax[2,0].plot([df.lbt.min(),df.lbt.max()], [0.3,0.3], 'k--', linewidth=1)
    ax[3,0].plot([df.lbt.min(),df.lbt.max()], [0.2*1e8,0.2*1e8], 'k--', linewidth=1)
    

    for id in D2D.ProjGalaxyID.unique():
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        t=temp[colorparam].values
        for i, param in enumerate([param0, param1, param2,param3]):
            y=temp[param].values
            points=np.array([x,y]).T.reshape(-1,1,2)
            segments=np.concatenate([points[:-1], points[1:]], axis=1)
            lc=LineCollection(segments, cmap=Cmap, norm=Norm)
            lc.set_array(t)
            lc.set_linewidth(0.5)
            ax[i,1].add_collection(lc)
            dfmergetemp=temp[temp[mergername]>0.]
            ax[i,1].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=3)
    ax[0,1].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'k--', linewidth=1)
    ax[1,1].plot([df.lbt.min(),df.lbt.max()], [5.1,5.1], 'k--', linewidth=1)
    #ax[2,1].plot([df.lbt.min(),df.lbt.max()], [0.3,0.3], 'k--', linewidth=1)
    ax[3,1].plot([df.lbt.min(),df.lbt.max()], [0.2*1e8,0.2*1e8], 'k--', linewidth=1)
    

    for id in B2D.ProjGalaxyID.unique():
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        t=temp[colorparam].values
        for i, param in enumerate([param0, param1, param2, param3]):
            y=temp[param].values
            points=np.array([x,y]).T.reshape(-1,1,2)
            segments=np.concatenate([points[:-1], points[1:]], axis=1)
            lc=LineCollection(segments, cmap=Cmap, norm=Norm)
            lc.set_array(t)
            lc.set_linewidth(0.5)
            ax[i,2].add_collection(lc)
            dfmergetemp=temp[temp[mergername]>0.]
            ax[i,2].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=3)
    ax[0,2].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'k--', linewidth=1)
    ax[1,2].plot([df.lbt.min(),df.lbt.max()], [5.1,5.1], 'k--', linewidth=1)
    #ax[2,2].plot([df.lbt.min(),df.lbt.max()], [0.3,0.3], 'k--', linewidth=1)
    ax[3,2].plot([df.lbt.min(),df.lbt.max()], [0.2*1e8,0.2*1e8], 'k--', linewidth=1)
    
    for id in D2B.ProjGalaxyID.unique():
        temp=df[df.ProjGalaxyID==id]
        x=temp.lbt.values
        t=temp[colorparam].values
        for i, param in enumerate([param0, param1, param2,param3]):
            y=temp[param].values
            points=np.array([x,y]).T.reshape(-1,1,2)
            segments=np.concatenate([points[:-1], points[1:]], axis=1)
            lc=LineCollection(segments, cmap=Cmap, norm=Norm)
            lc.set_array(t)
            lc.set_linewidth(0.5)
            ax[i,3].add_collection(lc)
            dfmergetemp=temp[temp[mergername]>0.]
            ax[i,3].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=3)
    ax[0,3].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'k--', linewidth=1)
    ax[1,3].plot([df.lbt.min(),df.lbt.max()], [5.1,5.1], 'k--', linewidth=1)
    #ax[2,3].plot([df.lbt.min(),df.lbt.max()], [0.3,0.3], 'k--', linewidth=1)
    ax[3,3].plot([df.lbt.min(),df.lbt.max()], [0.2*1e8,0.2*1e8], 'k--', linewidth=1)
    

    legendelements=[]
    for color in df[merger].unique():
        temp=df[df[merger]==color]
        for colorname in temp[mergername2].unique():
            legendelements.append(Line2D([0],[0], marker='o', color='w', label=colorname, markerfacecolor=color, markersize=10))
    ax[0,0].legend(handles=legendelements, bbox_to_anchor=(-0.2, 1.1))
    
    ax[0,0].set_title('B'),ax[0,1].set_title('D'),ax[0,2].set_title('B2D'), ax[0,3].set_title('D2B')
    ax[3,2].set_xlabel('look back time (Gyr)')
    ax[0,0].set_ylabel(name0) 
    ax[1,0].set_ylabel(name1) 
    ax[2,0].set_ylabel(name2)
    ax[3,0].set_ylabel(name3)
    ax[0,0].set_xlim(df.lbt.min(), df.lbt.max())
    plt.subplots_adjust(right=0.8, wspace=0.1, hspace=0)
    cbar_ax=fig.add_axes([0.85,0.15,0.05,0.8])
    sm=plt.cm.ScalarMappable(cmap=Cmap, norm=Norm)
    sm.set_array([])
    cbar=plt.colorbar(sm, cax=cbar_ax).set_label(r'residual of sSFR per mass per z [$M_{\odot} Gyr^{-1}$]')
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Evolution of'+str(param0)+str(param1)+str(param2)+str(param3)+str(mergername)+'.png')
    #plt.show()
    plt.close()

def evolutionplot2(df, param1, param2, param3):
    #plot evolution diagram for 3 paramters using seaborn
    fig, ax=plt.subplots(3,4, sharex=True)

    for i,typ in enumerate(['B2B', 'D2D', 'B2D', 'D2B']):
        temp=df[df.transtypen2d==typ]
        sns.lineplot(x='z',y=param1, hue='ProjGalaxyID',data=temp, palette=sns.color_palette('hls', temp.ProjGalaxyID.nunique()), ax=ax[0,i], legend=False, linewidth=0.8)
        sns.lineplot(x='z',y=param2, hue='ProjGalaxyID',data=temp, palette=sns.color_palette('hls', temp.ProjGalaxyID.nunique()), ax=ax[1,i], legend=False, linewidth=0.8)
        sns.lineplot(x='z',y=param3, hue='ProjGalaxyID',data=temp, palette=sns.color_palette('hls', temp.ProjGalaxyID.nunique()), ax=ax[2,i], legend=False, linewidth=0.8)
        ax[0,i].set_title(typ)
        ax[2,i].set_xlabel('z')
    ax[0,0].set_ylabel(param1)
    ax[0,1].set_ylabel(param2)
    ax[0,2].set_ylabel(param3)
    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.legend(bbox_to_anchor=(1.1,0.8), loc='center left')
    plt.xlim(0,3)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/evolution of'+param1+param2+param3+'.png')
    plt.show()

def calchalorate(mhalo, z, zdiff):
    #calc halo growth rate as descibed in Correa, 2015
    rate= 7e10*(mhalo/ 10e12)*(0.51 +(0.75*z))*((abs(zdiff))**1.5)
    return rate

def transformdata(df):
    #reoganise and manipulate data
    df=df[df.z<3]
    df['lbtrounded']=df.apply(lambda x: -np.round(x.lbt, decimals=0), axis=1)
    #f=df[df.num>12]
    df['Starmassmergerfrac']=df.apply(lambda x: zerotonancappedz(x.Starmassmergerfrac, x.z), axis=1)
    df['Starmergercategory']=df.apply(lambda x: classifymergermass(x.Starmassmergerfrac), axis=1)
    df['starmasscol']=df.apply(lambda x: colourmergermass(x.Starmergercategory), axis=1)

    df['DMmassmergerfrac']=df.apply(lambda x: zerotonancappedz(x.DMmassmergerfrac, x.z), axis=1)
    df['DMmergercategory']=df.apply(lambda x: classifymergermass(x.DMmassmergerfrac), axis=1)

    df['Stargasmergerfrac']=df.apply(lambda x: zerotonancappedz(x.Stargasmergerfrac, x.z), axis=1)
    df['stargasmergercategory']=df.apply(lambda x: classifymergergas(x.Stargasmergerfrac), axis=1)
    df['stargascol']=df.apply(lambda x: colourmergergas(x.stargasmergercategory), axis=1)

    df['sSFR1']=df.apply(lambda x: divide(x.SFR1,x.Starmass1), axis=1)
    df['logsSFR1']=df.apply(lambda x: logx(x.sSFR1), axis=1)
    df['sSFR3']=df.apply(lambda x: divide(x.SFR3,x.Starmass3), axis=1)
    df['logsSFR3']=df.apply(lambda x: logx(x.sSFR3), axis=1)
    df['sSFR5']=df.apply(lambda x: divide(x.SFR5,x.Starmass5), axis=1)
    df['logsSFR5']=df.apply(lambda x: logx(x.sSFR5), axis=1)
    df['sSFR10']=df.apply(lambda x: divide(x.SFR10,x.Starmass10), axis=1) 
    df['logsSFR10']=df.apply(lambda x: logx(x.sSFR10), axis=1)
    df['sSFR20']=df.apply(lambda x: divide(x.SFR20,x.Starmass20), axis=1) 
    df['logsSFR20']=df.apply(lambda x: logx(x.sSFR20), axis=1)
    
    df['SFR3m']=df.apply(lambda x: x.SFR3- x.SFR1, axis=1)
    df['Starmass3m']=df.apply(lambda x: x.Starmass3- x.Starmass1, axis=1)
    df['sSFR3m']=df.apply(lambda x: divide(x.SFR3m,x.Starmass3m), axis=1)
    df['logsSFR3m']=df.apply(lambda x: logx(x.sSFR3m), axis=1)

    df['SFR5m']=df.apply(lambda x: x.SFR5- x.SFR3, axis=1)
    df['Starmass5m']=df.apply(lambda x: x.Starmass5- x.Starmass3, axis=1)
    df['sSFR5m']=df.apply(lambda x: divide(x.SFR5m,x.Starmass5m), axis=1)
    df['logsSFR5m']=df.apply(lambda x: logx(x.sSFR5m), axis=1)

    df['SFR10m']=df.apply(lambda x: x.SFR10- x.SFR5, axis=1)
    df['Starmass10m']=df.apply(lambda x: x.Starmass10- x.Starmass5, axis=1)
    df['sSFR10m']=df.apply(lambda x: divide(x.SFR10m,x.Starmass10m), axis=1)
    df['logsSFR10m']=df.apply(lambda x: logx(x.sSFR10m), axis=1)

    df['SFR20m']=df.apply(lambda x: x.SFR20- x.SFR10, axis=1)
    df['Starmass20m']=df.apply(lambda x: x.Starmass20- x.Starmass10, axis=1)
    df['sSFR20m']=df.apply(lambda x: divide(x.SFR20m,x.Starmass20m), axis=1)
    df['logsSFR20m']=df.apply(lambda x: logx(x.sSFR20m), axis=1)

    df['SFR30m']=df.apply(lambda x: x.SFR- x.SFR20, axis=1)
    df['Starmass30m']=df.apply(lambda x: x.Starmass- x.Starmass20, axis=1)
    df['sSFR30m']=df.apply(lambda x: divide(x.SFR30m,x.Starmass30m), axis=1)
    df['logsSFR30m']=df.apply(lambda x: logx(x.sSFR30m), axis=1)

    """
    df['logsSFR1A']=df.apply(lambda x: logx(x.SFR1), axis=1)
    df['sSFR3mA']=df.apply(lambda x: divide(x.SFR3m, 3**3 -1), axis=1)
    df['logsSFR3mA']=df.apply(lambda x: logx(x.sSFR3mA), axis=1)
    df['sSFR10mA']=df.apply(lambda x: divide(x.SFR10m,10**3 - 3**3), axis=1)
    df['logsSFR10mA']=df.apply(lambda x: logx(x.sSFR10mA), axis=1)
    df['sSFR30mA']=df.apply(lambda x: divide(x.SFR30m, 30**3 - 10**3), axis=1)
    df['logsSFR30mA']=df.apply(lambda x: logx(x.sSFR30mA), axis=1)
    """
    B2B, D2D, B2D, D2B, BDB, DBD = bulgetranslists(df, 'n2d', 1.4, 0.1)
    df['transtypen2d']=df.apply(lambda x: bulgetrans(x.ProjGalaxyID, B2B, D2D, B2D, D2B, BDB, DBD), axis=1)

    B2B, D2D, B2D, D2B, BDB, DBD = bulgetranslists(df, 'BulgeToTotal', 0.5, 0.05)
    df['transtypeBTD']=df.apply(lambda x: bulgetrans(x.ProjGalaxyID, B2B, D2D, B2D, D2B, BDB, DBD), axis=1)
    
    df['n2d']=df.apply(lambda x: threshtonan(x.n2d, 0.4), axis=1)
    df['n_total']=df.apply(lambda x: threshtonan(x.n_total, 0.4), axis=1)
    df['logBHmass']=df.apply(lambda x: threshtonan(x.logBHmass, 0.01), axis=1)
    df['logDMmass']=df.apply(lambda x: threshtonan(x.logDMmass, 1), axis=1)
    #df.to_csv('evolvingEAGLEbulgediscmergedf'+sim_name+'.csv')
    """
    df['DMdensity200Area']=df.apply(lambda x: divide(x.M200DM,(4*np.pi * (x.R200DM ))), axis=1)
    df['DMHalfMassdensityArea']=df.apply(lambda x: divide((x.DMmass *0.5), (4 * np.pi * ((x.DMHalfMassRad)))), axis=1)
    df['DM500densityArea']=df.apply(lambda x: divide(x.M500DM, (4 * np.pi * ((x.R500DM)))), axis=1)
    df['DM200meandensityArea']=df.apply(lambda x: divide(x.M200DMmean, (4 * np.pi * ((x.R200DMmean)))), axis=1)
    
    df['logM200DM']=df.apply(lambda x: logx(x.M200DM), axis=1)
    df['logM500DM']=df.apply(lambda x: logx(x.M500DM), axis=1)
    df['logM200DMmean']=df.apply(lambda x: logx(x.M200DMmean), axis=1)
    df['logDMMass100AP']=df.apply(lambda x: logx(x.DMMass100AP), axis=1)

    df['logR200DM']=df.apply(lambda x: logx(x.R200DM), axis=1)
    df['logR500DM']=df.apply(lambda x: logx(x.R500DM), axis=1)
    df['logR200DMmean']=df.apply(lambda x: logx(x.R200DMmean), axis=1)
    df['logDMHalfMassRad']=df.apply(lambda x: logx(x.DMHalfMassRad), axis=1)

    df['Stargasmergerfrac']=df.apply(lambda x: threshtonan2(x.Starmassmergerfrac, x.Stargasmergerfrac, 0.01), axis=1)
    df=df.sort_values(by=['z'], ascending=False)
    df['logsBHmass']=df.apply(lambda x: cutBHmass(x.logBHmass), axis=1)
    df['dlbt']=df.groupby('ProjGalaxyID')['lookbacktime'].diff()
    df['dz']=df.groupby('ProjGalaxyID')['z'].diff()

    df['dlogBHmass']=df.groupby('ProjGalaxyID')['logBHmass'].diff()
    df['dBHmass']=df.groupby('ProjGalaxyID')['BHmass'].diff()
    df['dlogBHmassdt']=df.apply(lambda x: (x.dlogBHmass)/(x.dlbt), axis=1)
    df['dBHmassdt']=df.apply(lambda x: (x.dBHmass)/(x.dlbt), axis=1)

    
    df['dM200DM']=df.groupby('ProjGalaxyID')['M200DM'].diff()
    df['dM200DMdt']=df.apply(lambda x: (x.dM200DM)/(x.dlbt), axis=1)
    df['logdM200DMdt']=df.apply(lambda x: logx(x.dM200DMdt), axis=1)
    df['logDM200density']=df.apply(lambda x: logx(x.DM200density), axis=1)

    df['dM500DM']=df.groupby('ProjGalaxyID')['M500DM'].diff()
    df['dM500DMdt']=df.apply(lambda x: (x.dM500DM)/(x.dlbt), axis=1)
    df['logdM500DMdt']=df.apply(lambda x: logx(x.dM500DMdt), axis=1)
    df['logDM500density']=df.apply(lambda x: logx(x.DM500density), axis=1)

    df['dM100APDM']=df.groupby('ProjGalaxyID')['DMMass100AP'].diff()
    df['dM100APDMdt']=df.apply(lambda x: (x.dM100APDM)/(x.dlbt), axis=1)
    df['logdM100APDMdt']=df.apply(lambda x: logx(x.dM100APDMdt), axis=1)
    df['logDM100APdensity']=df.apply(lambda x: logx(x.DMAPdensity), axis=1)

    df['dM200meanDM']=df.groupby('ProjGalaxyID')['M200DMmean'].diff()
    df['dM200meanDMdt']=df.apply(lambda x: (x.dM200meanDM)/(x.dlbt), axis=1)
    df['logdM200meanDMdt']=df.apply(lambda x: logx(x.dM200meanDMdt), axis=1)
    df['logDM200meandensity']=df.apply(lambda x: logx(x.DM200meandensity), axis=1)

    df['dMDM']=df.groupby('ProjGalaxyID')['DMmass'].diff()
    df['dMDMdt']=df.apply(lambda x: (x.dMDM)/(x.dlbt), axis=1)
    df['logdMDMdt']=df.apply(lambda x: logx(x.dMDMdt), axis=1)
    df['logDMHalfMassdensity']=df.apply(lambda x: logx(x.DMHalfMassdensity), axis=1)


    df['dR200DM']=df.groupby('ProjGalaxyID')['R200DM'].diff()
    df['dR200DMdt']=df.apply(lambda x: (x.dR200DM)/(x.dlbt), axis=1)
    df['logdR200DMdt']=df.apply(lambda x: logx(x.dR200DMdt), axis=1)

    df['dR500DM']=df.groupby('ProjGalaxyID')['R500DM'].diff()
    df['dR500DMdt']=df.apply(lambda x: (x.dR500DM)/(x.dlbt), axis=1)
    df['logdR500DMdt']=df.apply(lambda x: logx(x.dR500DMdt), axis=1)

    df['dR200meanDM']=df.groupby('ProjGalaxyID')['R200DMmean'].diff()
    df['dR200meanDMdt']=df.apply(lambda x: (x.dR200meanDM)/(x.dlbt), axis=1)
    df['logdR200meanDMdt']=df.apply(lambda x: logx(x.dR200meanDMdt), axis=1)
   
    df['dRhalfDM']=df.groupby('ProjGalaxyID')['DMHalfMassdensity'].diff()
    df['dRhalfDMdt']=df.apply(lambda x: (x.dRhalfDM)/(x.dlbt), axis=1)
    df['logdRhalfDMdt']=df.apply(lambda x: logx(x.dRhalfDMdt), axis=1)

    #df['DMmassdtcorrea']=df.apply(lambda x: calchalorate(x.DMmass, x.z, x.dz), axis=1)
    #df['logDMmassdtcorrea']=df.apply(lambda x: logx(x.DMmassdtcorrea), axis=1)
    for name in ['DM200density', 'DMdensity200Area','DMHalfMassdensity','DMAPdensity', 'DM200meandensity','DMHalfMassdensityArea',  'DM200meandensityArea']:
        grouped=df[['z', name]].groupby(['z']).agg({name:['median', 'std']})
        grouped=grouped.xs(name, axis=1, drop_level=True)
        df=pd.merge(df, grouped, on=['z'], how='left').drop_duplicates()
        print(df.columns.values)
        df=df.rename({'median':name+'_median', 'std':name+'_std'}, axis=1)
        print(df.columns.values)
    print(df.DM200density_std)
    print(df.columns.values)
    df['residDM200density']=df.apply(lambda x: divide((x.DM200density-x.DM200density_median),x.DM200density_std) , axis=1)
    df['residDMdensity200Area']=df.apply(lambda x: divide((x.DMdensity200Area-x.DMdensity200Area_median),x.DMdensity200Area_std) , axis=1)
    df['residDMHalfMassdensity']=df.apply(lambda x: divide((x.DMHalfMassdensity-x.DMHalfMassdensity_median),x.DMHalfMassdensity_std) , axis=1)
    df['residDMHalfMassdensityArea']=df.apply(lambda x: divide((x.DMHalfMassdensityArea-x.DMHalfMassdensityArea_median),x.DMHalfMassdensityArea_std) , axis=1)
    df['residDM200meandensity']=df.apply(lambda x: divide((x.DM200meandensity-x.DM200meandensity_median),x.DM200meandensity_std) , axis=1)
    df['residDM200meandensityArea']=df.apply(lambda x: divide((x.DM200meandensityArea-x.DM200meandensityArea_median),x.DM200meandensityArea_std) , axis=1)
    df['residDMAPdensity']=df.apply(lambda x: divide((x.DMAPdensity-x.DMAPdensity_median),x.DMAPdensity_std) , axis=1)
    """
    #df['logDMEllipticitypermass']=df.apply(lambda x: logx(x.DMEllipticitypermass), axis=1)
    df['Concentration5200']=df.apply(lambda x: (divide(x.DMMass5,x.M200DM))**(1/3), axis=1)
    df['Concentration530']=df.apply(lambda x: (divide(x.DMMass5,x.DMMass30))**(1/3), axis=1)
    df['Concentration30200']=df.apply(lambda x: (divide(x.DMMass30,x.M200DM))**(1/3), axis=1)

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
    df['M200DM']=df.apply(lambda x: 10**(x.logM200DM), axis=1)
    df['M500DM']=df.apply(lambda x: 10**(x.logM500DM), axis=1)
    df['M200DMmean']=df.apply(lambda x: 10**(x.logM200DMmean), axis=1)
    df['DMMass100AP']=df.apply(lambda x: 10**(x.logDMMass100AP), axis=1)
    df['R200DM']=df.apply(lambda x: 10**(x.logR200DM), axis=1)
    df['R500DM']=df.apply(lambda x: 10**(x.logR500DM), axis=1)
    df['DMHalfMassRad']=df.apply(lambda x: 10**(x.logDMHalfMassRad), axis=1)
    df['Hz']=df.apply(lambda x: Planck13.H(x.z).value, axis=1)
    df['Concentration5100']=df.apply(lambda x: (divide(x.DMMass5,x.DMMass100AP))**(1/3), axis=1)
    df['Concentration30100']=df.apply(lambda x: (divide(x.DMMass30,x.DMMass100AP))**(1/3), axis=1)
    df['Concentration530']=df.apply(lambda x: (divide(x.DMMass5, x.DMMass30))**(1/3), axis=1)
    df['Concentration500200']=df.apply(lambda x: (divide(x.M500DM,x.M200DM))**(1/3), axis=1)
    df['Concentration2500200']=df.apply(lambda x: (divide(x.M2500DM,x.M200DM))**(1/3), axis=1)
    df['Concentration2500500']=df.apply(lambda x: (divide(x.M2500DM,x.M500DM))**(1/3), axis=1)
    df['Concentration500200R']=df.apply(lambda x: divide(x.R500DM,x.R200DM), axis=1)
    df['Concentration2500200R']=df.apply(lambda x: divide(x.R2500DM,x.R200DM), axis=1)
    df['Concentration2500500R']=df.apply(lambda x: divide(x.R2500DM,x.R500DM), axis=1)
    df['Concentration200halfR']=df.apply(lambda x: divide(x.R200DM,x.DMHalfMassRad), axis=1)
    df['Concentration500halfR']=df.apply(lambda x: divide(x.R500DM,x.DMHalfMassRad), axis=1)
    df['Concentration2500halfR']=df.apply(lambda x: divide(x.R2500DM,x.DMHalfMassRad), axis=1)
    df['vhsqrd2500200']=df.apply(lambda x:  x.Concentration2500200 * (x.M200DM * x.R200DM), axis=1 )
    df['vh2500200']=df.apply(lambda x:  np.sqrt(x.vhsqrd2500200), axis=1 )
    df0=df.sort_values(by='z').drop_duplicates('ProjGalaxyID')
    df0=df0[['ProjGalaxyID', 'vh2500200']]
    df=pd.merge(df, df0, on=['ProjGalaxyID'], how='left', suffixes=('', '_0')).drop_duplicates()
    df1=df.sort_values(by='z', ascending=False).drop_duplicates('ProjGalaxyID')
    df1=df1[['ProjGalaxyID', 'vh2500200']]
    df=pd.merge(df, df1, on=['ProjGalaxyID'], how='left', suffixes=('', '_1')).drop_duplicates()
    df['vh2500200_norm0']=df.apply(lambda x:  divide(x.vh2500200, x.vh2500200_0), axis=1 )
    df['vh2500200_norm1']=df.apply(lambda x:  divide(x.vh2500200, x.vh2500200_1), axis=1 )
    df['DMHalfMassdensity200']=df.apply(lambda x: divide((x.logM200DM *0.5), (4/3 * np.pi * ((x.DMHalfMassRad)**3))), axis=1)
    df['logDMHalfMassdensity200']=df.apply(lambda x: logx(x.DMHalfMassdensity200), axis=1)
    df['num']= df.groupby('ProjGalaxyID')['ProjGalaxyID'].transform('count')

    df['logSFThermalEnergy']=df.apply(lambda x: logx(x.SFThermalEnergy), axis=1)
    df['logNSFThermalEnergy']=df.apply(lambda x: logx(x.NSFThermalEnergy), axis=1)
    df['logSFMass']=df.apply(lambda x: logx(x.SFMass), axis=1)
    df['logNSFMass']=df.apply(lambda x: logx(x.NSFMass), axis=1)

    df['sSFThermalEnergy']=df.apply(lambda x: divide(x.SFThermalEnergy, x.Gasmass), axis=1)
    df['sNSFThermalEnergy']=df.apply(lambda x: divide(x.NSFThermalEnergy, x.Gasmass), axis=1)
    df['sSFMass']=df.apply(lambda x: divide(x.SFMass, x.Gasmass), axis=1)
    df['sNSFMass']=df.apply(lambda x: divide(x.NSFMass, x.Gasmass), axis=1)

    df['logsSFThermalEnergy']=df.apply(lambda x: logx(x.sSFThermalEnergy), axis=1)
    df['logsNSFThermalEnergy']=df.apply(lambda x: logx(x.sNSFThermalEnergy), axis=1)
    df['logsSFMass']=df.apply(lambda x: logx(x.sSFMass), axis=1)
    df['logsNSFMass']=df.apply(lambda x: logx(x.sNSFMass), axis=1)

    df['logSFTemp']=df.apply(lambda x: logx(x.SF_MassWeightedTemperature), axis=1)
    df['sSFTemp']=df.apply(lambda x: divide(x.SF_MassWeightedTemperature, (x.SFMass +x.NSFMass)), axis=1)
    df['logsSFTemp']=df.apply(lambda x: logx(x.sSFTemp), axis=1)

    df['logSFMass']=df.apply(lambda x: logx(x.SFMass), axis=1)
    df['sSFMass']=df.apply(lambda x: divide(x.SFMass, (x.SFMass +x.NSFMass)), axis=1)
    df['logsSFMass']=df.apply(lambda x: logx(x.sSFMass), axis=1)

    df['logsSFR1A']=df.apply(lambda x: logx(x.SFR1), axis=1)
    df['sSFR3mA']=df.apply(lambda x: divide(x.SFR3m, (3**3) -1), axis=1)
    df['logsSFR3mA']=df.apply(lambda x: logx(x.sSFR3mA), axis=1)
    df['sSFR5mA']=df.apply(lambda x: divide(x.SFR5m, (5**3) -(3**3)), axis=1)
    df['logsSFR5mA']=df.apply(lambda x: logx(x.sSFR5mA), axis=1)
    df['sSFR10mA']=df.apply(lambda x: divide(x.SFR10m,(10**3) - (5**3)), axis=1)
    df['logsSFR10mA']=df.apply(lambda x: logx(x.sSFR10mA), axis=1)
    df['sSFR20mA']=df.apply(lambda x: divide(x.SFR20m,(20**3) - (10**3)), axis=1)
    df['logsSFR20mA']=df.apply(lambda x: logx(x.sSFR20mA), axis=1)
    df['sSFR30mA']=df.apply(lambda x: divide(x.SFR30m, (30**3) - (20**3)), axis=1)
    df['logsSFR30mA']=df.apply(lambda x: logx(x.sSFR30mA), axis=1)

    df['sGas1']=df.apply(lambda x: divide(x.Gasmass1,x.Starmass1), axis=1)
    df['logsGas1']=df.apply(lambda x: logx(x.sGas1), axis=1)
    
    df['Gas3m']=df.apply(lambda x: x.Gasmass3- x.Gasmass1, axis=1)
    df['sGas3m']=df.apply(lambda x: divide(x.Gas3m,x.Starmass3m), axis=1)
    df['logsGas3m']=df.apply(lambda x: logx(x.sGas3m), axis=1)

    df['Gas5m']=df.apply(lambda x: x.Gasmass5- x.Gasmass3, axis=1)
    df['sGasm']=df.apply(lambda x: divide(x.Gas5m,x.Starmass5m), axis=1)
    df['logsGas5m']=df.apply(lambda x: logx(x.sGasm), axis=1)

    df['Gas10m']=df.apply(lambda x: x.Gasmass10- x.Gasmass5, axis=1)
    df['sGas10m']=df.apply(lambda x: divide(x.Gas10m,x.Starmass10m), axis=1)
    df['logsGas10m']=df.apply(lambda x: logx(x.sGas10m), axis=1)

    df['Gas20m']=df.apply(lambda x: x.Gasmass20- x.Gasmass10, axis=1)
    df['sGas20m']=df.apply(lambda x: divide(x.Gas20m,x.Starmass20m), axis=1)
    df['logsGas20m']=df.apply(lambda x: logx(x.sGas20m), axis=1)

    df['Gas30m']=df.apply(lambda x: x.Gasmass- x.Gasmass20, axis=1)
    df['sGas30m']=df.apply(lambda x: divide(x.Gas30m,x.Starmass30m), axis=1)
    df['logsGas30m']=df.apply(lambda x: logx(x.sGas30m), axis=1)
    df.to_csv('evolvingEAGLEbulgediscmergedf'+sim_name+'.csv')

    return df
    #df['vHsqrd']=df.apply(lambda x: divide(6.67*10e-11*x.M200, x.R200), axis=1)


def plotbulgetodisc(df, sim_name):
    #plot different graphs.
    #df=transformdata(df)
    print(df.ProjGalaxyID.unique())
    print(df.columns.values)

    plotfrac(df,'roundlogmass2', 'categoryn2d', 'catfrac')
    plotbulgedisctranscolour(df,1,'n2d','logsSFRpermass',1.5,0.1)
    specificgalaxyplot(df, 1298133, 'BulgeToTotal', 'n2d', 'logsSFR', 'logmass')
    #df['catn2d15']=df.apply(lambda x: categorise(x.asymm, x.n2d, 1.5, 0.3),axis=1)
    #df['catn2d14']=df.apply(lambda x: categorise(x.asymm, x.n2d, 1.4, 0.31),axis=1)
    #df['catn2d16']=df.apply(lambda x: categorise(x.asymm, x.n2d, 1.6, 0.29),axis=1)
    #categorybarchart(df, 'catn2d15', 'catn2d14', 'catn2d16')
    
    plotbulgedisctranscolourmerger(df,1.5,'n2d','logsSFRpermass','starmasscol','Starmassmergerfrac', 'Starmergercategory', 1.5, 0.1)
    evolutionplot4(df, 'n2d',r'n2d', 'logDMHalfMassdensity',r'log($\rho_{halo, mass_{\frac{1}{2}}}$) [$M_{\odot} pkpc^{-3}$]', \
         'Concentration2500200',r'$(\frac{M_2500}{M_{200}})^{(1/3)}$', 'vh2500200',r'$v_h [G M_{\odot} (pkpc)^{-1}]$',\
             'logsSFRpermass', 1.4, 'stargascol', 'Stargasmergerfrac', 'stargasmergercategory')

    plotmovinghistogram4(df, 'logSFTemp', 'logsSFTemp', 'logSFMass', 'logsSFMass', 'transtypen2d',\
        r'log($T_{SF, weighted}$) [$K$]',  r'log($\frac{T_{SF, weighted}}{M_{total}}$ [KM_{\odot}^-1])', r'log($M_{gas, SF}$) [$M_{\odot}$]', r'log($\frac{M_{gas, SF}}{M_{gas,tot}})$')

    plotmultivariateplot(df)
    plotSFRcircles(df,'logsSFR1','logsSFR3m','logsSFR5m','logsSFR10m','logsSFR20m','logsSFR30m', 'n2d', -12,  -7, 'logsBHmass', 1.5 , 0.1, 'transtypen2d', r'Log(sSFR) [$Gyr^{-1}$]', r'Log($M_{BH}/M_{*}$) [$M_{\odot}$]')
    plotSFRcircles(df,'Gasmass1','Gas3m','Gas5m','Gas10m','Gas20m','Gas30m', 'n2d', -12,  -7, 'logsBHmass', 1.5, 0.1, 'transtypen2d', r'Log($M_{gas}$) [$M_{\odot}$]', r'Log($E_{SF}$) [$M_{\odot} (km/s)^2$]')
    plotmovinghistogram4(df, 'logSFThermalEnergy', 'logNSFThermalEnergy', 'logsSFMass', 'logsNSFMass', 'n2d')
    #plotmovingquantiles(df, 'logmass', 'dDMmassdt', 'n2d')
    #evolutionplot3(df, 'BulgeToTotal','logBHmass','logdBHmassdt',  'logsSFRpermass', 0.5, 'stargascol', 'Stargasmergerfrac', 'stargasmergercategory')
    #evolutionplot2(df, 'logBHmass', 'logdBHmassdt', 'n2d')
    plotmergerrate(df, 'Starmergercategory', 'lbtrounded', 'Starmass', 'Starmassmergerfrac')
    #plotmergergasrate(df, 'stargasmergercategory', 'lbtrounded')
    mergerdf= mergerinvestigation(df)
    plotbulgedisctranscolourmerger(df,1.5,'BulgeToTotal','logsSFRpermass','stargascol','stargasmergerfrac','stargasmergercategory', 0.5,0.1)
    plotbulgedisctranscolourmerger(mergerdf,1.5,'BulgeToTotal','logsSFRpermass','stargascol', 'stargasmergerfrac','stargasmergercategory',0.5,0.1, merge=True)
    colourscatter(df, 'logmass','logSFR',  'n2d')
    #specificgalplotmasses(df, galaxyid)
    #specificgalplotratesofvariabless(df, galaxyid)
    threeDplot(df, 'z','DiscToTotal','logBHmass', 'Starmass', 'logsSFR')
    stackedhistogram(df, 'n_total','n_disc','n_bulge','n_bulge_exp')
    
if __name__ == "__main__":
    sim_names=['RefL0050N0752']
    #query_type=mainbranch or allbranches
    for sim_name in sim_names:
        query_type='mainbranch'
        read_data=True
        if(read_data):
            #read data
            print('........reading.......')
            df=pd.read_csv('evolvingEAGLEbulgediscmergedf'+sim_name+'.csv')
        else:
            print('........writing.......')
            #calculate variables from galaxy images
            df=pd.read_csv('evolvingEAGLEimages'+query_type+'df'+sim_name+'.csv')
            df=df[df.z<3]
            df['num']= df.groupby('ProjGalaxyID')['ProjGalaxyID'].transform('count')
            df=df[df.num>17]
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


