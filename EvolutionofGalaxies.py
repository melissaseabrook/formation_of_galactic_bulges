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

def roundx(x, dec=1):
    if x>0:
        return np.round(x, decimals=dec)
    elif x<0:
        return np.round(x, decimals=dec)
    else: 
        return np.nan

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

def classifymergermass(x):
    if x>0.25: 
        return 'major'
    elif x>0.01:
        return 'minor'
    else:
        return 'accretion'

def classifymergergas(x):
    if x>0.5: 
        return 'gasrich'
    elif x<0.2:
        return 'gaspoor'
    else:
        return 'undefined'

def colourmergergas(x):
    if (x=='gasrich'):
        return 'g'
    elif (x=='undefined'):
        return 'orange'
    elif (x=='gaspoor'):
        return 'red'

def colourmergermass(x):
    if (x=='major'):
        return 'g'
    elif (x=='accretion'):
        return 'orange'
    elif (x=='minor'):
        return 'red'

def mediancolor(x, col):
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

def plotmovinghistogram(df, histparam, binparam):
    #zmin=df.z.min()
    
    z0df=df[df.zrounded==0.]
    df=df[(df.zrounded==0.1) | (df.zrounded==0.5) | (df.zrounded==1.0) | (df.zrounded==2.0)]
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
    for i,zi in enumerate([0.1, 0.5, 1.0, 2.0]):
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

def plotmovinghistogram4(df, histparam1, histparam2, histparam3, histparam4, binparam):
    z0df=df[df.zrounded==0.]
    df=df[(df.zrounded==0.1) | (df.zrounded==0.5) | (df.zrounded==1.) | (df.zrounded==2.)]
    z0df=z0df[['ProjGalaxyID', binparam]]
    z0df['marker_bin']=pd.qcut(z0df[binparam],10, labels=['10','20','30','40','50','60','70','80','90','100'])
    df=pd.merge(df, z0df, on=['ProjGalaxyID'], how='left',  suffixes=('','_proj'))
    fig, axs =plt.subplots(4, 4, sharex='col', sharey='row', figsize=(9,6))
    fig.suptitle('Time evolution of historgram of '+histparam1+' showing distribution of '+binparam)
    binedgs1=np.linspace(df[histparam1].min(), df[histparam1].max(), 30)
    binedgs2=np.linspace(df[histparam2].min(), df[histparam2].max(), 30)
    binedgs3=np.linspace(df[histparam3].min(), df[histparam3].max(), 30)
    binedgs4=np.linspace(df[histparam4].min(), df[histparam4].max(), 30)
    for i,zi in enumerate([0.1, 0.5, 1.0, 2.0]):
        
        zdf=df[df.zrounded==zi]
        lowdf=zdf[zdf.marker_bin=='10']
        highdf=zdf[zdf.marker_bin=='100']
        axs[i,0].hist(zdf[histparam1], color="k", alpha=0.4,label='z='+str(zi), histtype='stepfilled', bins=binedgs1)
        axs[i,1].hist(zdf[histparam2], color="k", alpha=0.4,label='z='+str(zi), histtype='stepfilled', bins=binedgs2)
        axs[i,2].hist(zdf[histparam3], color="k", alpha=0.4,label='z='+str(zi), histtype='stepfilled', bins=binedgs3)
        axs[i,3].hist(zdf[histparam4], color="k", alpha=0.4,label='z='+str(zi), histtype='stepfilled', bins=binedgs4)
        axs[i,0].hist(lowdf[histparam1], color="r", alpha=0.5, histtype='step', bins=binedgs1)
        axs[i,0].hist(highdf[histparam1], color="b", alpha=0.5, histtype='step', bins=binedgs1)
        axs[i,1].hist(lowdf[histparam2], color="r", alpha=0.5, histtype='step', bins=binedgs2)
        axs[i,1].hist(highdf[histparam2], color="b", alpha=0.5, histtype='step', bins=binedgs2)
        axs[i,2].hist(lowdf[histparam3], color="r", alpha=0.5, histtype='step', bins=binedgs3)
        axs[i,2].hist(highdf[histparam3], color="b", alpha=0.5, histtype='step', bins=binedgs3)
        axs[i,3].hist(lowdf[histparam4], color="r", alpha=0.5, histtype='step', bins=binedgs4)
        axs[i,3].hist(highdf[histparam4], color="b", alpha=0.5, histtype='step', bins=binedgs4)
        #axs[i,0].set_xlabel('')
        #axs[i,1].set_xlabel('')
        #axs[i,0].set_ylabel('')
        axs[i,0].set_ylabel('Count')
        axs[i,3].legend()
        
    axs[3,0].set_xlabel(histparam1)
    axs[3,1].set_xlabel(histparam2)
    axs[3,2].set_xlabel(histparam3)
    axs[3,3].set_xlabel(histparam4)
    plt.subplots_adjust(wspace=0, hspace=0)
    #handles, labels = axs[3].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='lower center')
    #plt.tight_layout()
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Histogramof'+histparam1+'highlighted'+binparam+'.png')
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

def plotbulgedisctranscolourmerger(df, maxz, param, colorparam, merger, mergername, thresh, threshstep, merge=True):
    B2B =[]
    D2D= []
    B2D=[]
    D2B=[]
    BDB=[]
    DBD=[]
    if (merger==True):
        df=df[df.Starmassmergerfrac>0.]
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
        ax[1,0].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=5)
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
        ax[1,1].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=5)
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
        ax[1,2].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=5)
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
        ax[1,3].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=5)
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
        ax[1,4].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=5)
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
        ax[1,5].scatter(dfmergetemp.lbt, dfmergetemp[param], c=dfmergetemp[merger], s=5)

    ax[1,5].plot([df.lbt.min(),df.lbt.max()], [thresh,thresh], 'r--', linewidth=1)
    ax[0,5].bar(0,len(DBD), color='purple')
    ax[0,5].text(-.1, 1, str(round(100*len(DBD)/nmax, 1))+'%', fontsize=12, color='white')

    legendelements=[]
    for color in df[merger].unique():
        temp=df[df[merger]==color]
        for colorname in temp[mergername].unique():
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
    cbar=plt.colorbar(sm, cax=cbar_ax).set_label(colorparam)
    if merge==True:
        plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Evolution of'+str(param)+'thresh'+str(thresh)+'zmax'+str(maxz)+'colorby'+str(mergername)+'merge.png')
    else:
        plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Evolution of'+str(param)+'thresh'+str(thresh)+'zmax'+str(maxz)+'colorby'+str(mergername)+'.png')
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
    df=df[(df.zrounded==0.) | (df.zrounded==0.2) | (df.zrounded==0.5) | (df.zrounded==1.0)]
    z0df=z0df[['ProjGalaxyID', binparam]]
    #z0df['marker_bin']=pd.qcut(z0df[binparam], 5, labels=['10','40','60','80','100'])
    z0df['marker_bin']=pd.qcut(z0df[binparam], 20, labels=['10','20','30','1','2','3','4','5','6','7','8','9','11','40','50','60','70','80','90','100'])
    df=pd.merge(df, z0df, on=['ProjGalaxyID'], how='left',  suffixes=('','_proj'))
    
    fig, axs =plt.subplots(4, 2, sharex=True, sharey=True, figsize=(9,6))
    fig.suptitle('Time evolution of '+paramx+paramy+' showing distribution of '+binparam)
    axs[0,0].set_title('10th percentile of '+binparam)
    axs[0,1].set_title('90th percentile of '+binparam)
    for i,zi in enumerate([0., 0.2, 0.5, 1.0]):
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

def plotmergerrate(df, cat, time, mass, mergermassfrac):
    #df1=df[[time, cat]].groupby([time, cat]).size().unstack(fill_value=0)
    """
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
    """
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

def transformdata(df):
    df=df[df.logDMmass>5]
    df['lbtrounded']=df.apply(lambda x: -np.round(x.lbt, decimals=0), axis=1)
    df['zplus1']=df.apply(lambda x: x.zrounded +1, axis=1)
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
    df['sSFR10']=df.apply(lambda x: divide(x.SFR10,x.Starmass10), axis=1) 
    df['logsSFR10']=df.apply(lambda x: logx(x.sSFR10), axis=1)
    
    df['SFR3m']=df.apply(lambda x: x.SFR3- x.SFR1, axis=1)
    df['Starmass3m']=df.apply(lambda x: x.Starmass3- x.Starmass1, axis=1)
    df['sSFR3m']=df.apply(lambda x: divide(x.SFR3m,x.Starmass3m), axis=1)
    df['logsSFR3m']=df.apply(lambda x: logx(x.sSFR3m), axis=1)

    df['SFR10m']=df.apply(lambda x: x.SFR10- x.SFR3, axis=1)
    df['Starmass10m']=df.apply(lambda x: x.Starmass10- x.Starmass3, axis=1)
    df['sSFR10m']=df.apply(lambda x: divide(x.SFR10m,x.Starmass10m), axis=1)
    df['logsSFR10m']=df.apply(lambda x: logx(x.sSFR10m), axis=1)

    df['SFR30m']=df.apply(lambda x: x.SFR- x.SFR10, axis=1)
    df['Starmass30m']=df.apply(lambda x: x.Starmass- x.Starmass10, axis=1)
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

    return df
    #df['vHsqrd']=df.apply(lambda x: divide(6.67*10e-11*x.M200, x.R200), axis=1)

def plotSFRcircles(df, histparam1, histparam2, histparam3, histparam4, binparam, vmin, vmax, hparam):
    z0df=df[df.zrounded==0.]
    df=df[(df.zrounded==0.1) | (df.zrounded==0.5) | (df.zrounded==1.) | (df.zrounded==2.)]
    z0df=z0df[['ProjGalaxyID', binparam]]
    z0df['marker_bin']=pd.qcut(z0df[binparam],8, labels=['10','30','40','50','70','80','90','100'])
    df=pd.merge(df, z0df, on=['ProjGalaxyID'], how='left',  suffixes=('','_proj'))


    fig, axs =plt.subplots(12, 4, figsize=(6,10), gridspec_kw={'width_ratios':[4,4,4,0.5], 'height_ratios':[4,1.5,1,4,1.5,1,4,1.5,1,4,1.5,1]})

    fig.suptitle('Time evolution of historgram of '+histparam1+' showing distribution of '+binparam)
    axs[0,0].set_title('10th Percentile')
    axs[0,1].set_title('Median')
    axs[0,2].set_title('90th Percentile')
    maxcol=max(abs(df[histparam4].min()), abs(df[histparam4].max()))
    Cmap=mcol.LinearSegmentedColormap.from_list("cmop", ['red','mediumorchid','blue'])
    #Cmap='seismic'
    norm=plt.Normalize(vmin,vmax)

    for i1,zi in enumerate([0.1, 0.5, 1.0, 2.0]):
        i=3*i1
        i2=i+1
        i3=i+2
        zdf=df[df.zrounded==zi]
              
        lowdf=zdf[zdf.marker_bin=='10']
        middf=zdf[zdf.marker_bin=='50']
        highdf=zdf[zdf.marker_bin=='100']
        minn=min([lowdf[histparam1].median(),lowdf[histparam2].median(),lowdf[histparam3].median(),lowdf[histparam4].median(),middf[histparam1].median(),middf[histparam2].median(),middf[histparam3].median(),middf[histparam4].median(),highdf[histparam1].median(),highdf[histparam2].median(),highdf[histparam3].median(),highdf[histparam4].median()])
        maxx=max([lowdf[histparam1].median(),lowdf[histparam2].median(),lowdf[histparam3].median(),lowdf[histparam4].median(),middf[histparam1].median(),middf[histparam2].median(),middf[histparam3].median(),middf[histparam4].median(),highdf[histparam1].median(),highdf[histparam2].median(),highdf[histparam3].median(),highdf[histparam4].median()])
        norm=plt.Normalize(minn, maxx)
        sm=plt.cm.ScalarMappable(cmap=Cmap, norm=norm)
        sm.set_array([])
        cbar=fig.colorbar(sm, cax=axs[i,3]).set_label('logsSFR')
        print(i1,i,i2,i3)
        binedgs1=np.linspace(df[hparam].min(), df[hparam].max(), 30)
        
        for j,df2 in enumerate([lowdf, middf, highdf]):
            axs[i,j].set_xlim(0, 60)
            axs[i,j].set_ylim(0, 60)
            axs[i3,j].remove()
            axs[i3,3].set_visible(False)
            axs[i2,3].set_visible(False)        

            """
            color1=np.round(abs(df2[histparam1].median())/maxcol, decimals=1)
            color3=np.round(abs(df2[histparam2].median())/maxcol, decimals=1)
            color10=np.round(abs(df2[histparam3].median())/maxcol, decimals=1)
            color30=np.round(abs(df2[histparam4].median())/maxcol, decimals=1)
            """
            
            #Cmap='seismic'
            color1=df2[histparam1].median()
            color3=df2[histparam2].median()
            color10=df2[histparam3].median()
            color30=df2[histparam4].median()

            circle1=plt.Circle((30,30), 2, color=Cmap(norm(color1)))
            circle3=plt.Circle((30,30), 5, color=Cmap(norm(color3)))
            circle10=plt.Circle((30,30), 15, color=Cmap(norm(color10)))
            circle30=plt.Circle((30,30), 30, color=Cmap(norm(color30)))
            axs[i,j].add_artist(circle30)
            axs[i,j].add_artist(circle10)
            axs[i,j].add_artist(circle3)
            axs[i,j].add_artist(circle1)
            axs[i,j].set_xlabel('')
            axs[i,j].set_ylabel('')
            
            axs[i2,j].hist(zdf[hparam], color="k", alpha=0.4,label='z='+str(zi), histtype='stepfilled', bins=binedgs1)
            axs[i2,0].hist(lowdf[hparam], color="b", alpha=0.5, histtype='step', bins=binedgs1)
            axs[i2,1].hist(middf[hparam], color="purple", alpha=0.5, histtype='step', bins=binedgs1)
            axs[i2,2].hist(highdf[hparam], color="r", alpha=0.5, histtype='step', bins=binedgs1)
            axs[10,j].set_xlabel(hparam)

            axs[i,j].set_xticks([])
            axs[i,1].set_yticks([])
            axs[i,2].set_yticks([])
            axs[i2,1].set_yticks([])
            axs[i2,2].set_yticks([])

            axs[i,1].set_yticks([])
            axs[i,2].set_yticks([])
        
        axs[i2,0].set_ylabel('z='+str(zi))
        axs[i,0].set_ylabel('Radius')
        axs[i,2].legend()
        
    plt.subplots_adjust(right=0.8, wspace=0, hspace=0)
    
    #handles, labels = axs[3].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='lower center')
    #plt.tight_layout()
    #plt.subplots_adjust(right=0.8)
    #cbar_ax=fig.add_axes([0.85,0.15,0.05,0.8])
    
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/'+histparam4+'circleplot'+binparam+'.png')
    plt.show()

def cutBHmass(x):
    if x>0:
        if x<4.5:
            return np.nan
        else:
            return x
    else:
        return x

def plotSFRpergalaxy(df):
    #zmin=df.z.min()
    
    z0df=df[df.zrounded==0.]
    df=df[(df.zrounded==0.1) | (df.zrounded==0.5) | (df.zrounded==1.0) | (df.zrounded==2.0)]
    

    fig, axs =plt.subplots(4, 2, sharex=True, figsize=(9,6))
    fig.suptitle('Time evolution of historgram of  showing distribution of ')
    axs[0,0].set_title('0-10th percentile of ')
    axs[0,1].set_title('90-100th percentile of ')
    for i,zi in enumerate([0.1, 0.5, 1.0, 2.0]):
        #ax[i] = axs[i].twinx()
        zdf=df[df.zrounded==zi]
        temp=[]
        for galid in zdf.ProjGalaxyID.unique():
            temp=zdf[zdf.ProjGalaxyID == galid]
            axs[i,0].plot([1, 3, 10, 30],[temp.SFR1.values, temp.SFR3.values, temp.SFR10.values, temp.SFR.values])
            axs[i,1].plot([1, 3, 10, 30],[temp.SFR1.values, temp.SFR3m.values, temp.SFR10m.values, temp.SFR30m.values])
        
        
        axs[i,0].set_xlabel('')
        axs[i,1].set_xlabel('')
        axs[i,0].set_ylabel('')
        axs[i,1].legend()
        

    #plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Histogramof'+histparam+'highlighted'+binparam+'.png')
    plt.show()

def plotbulgetodisc(df, sim_name):
    #df=df[df.n2d>0.5]
    df=df[df.z<3]
    df=transformdata(df)
    df=df[df.logBHmass>0]
    #df=df[df.logsSFR30m<0]
    #df=df[df.logsSFR3m<0]
    #df=df[df.logsSFR1<0]
    #df=df[df.logsSFR10m<0]
    
    plotSFRcircles(df,'logsSFR1','logsSFR3m','logsSFR10m','logsSFR30m', 'n2d', -12,  -7, 'logsDMmass')
    exit()
    
    plotmovinghistogram4(df,'logsSFR1','logsSFR3m','logsSFR10m','logsSFR30m', 'n2d')
   
    exit()

    
    """
    df['logsBHmass']=df.apply(lambda x: cutBHmass(x.logBHmass), axis=1)
    df['dBHmass']=df.groupby('ProjGalaxyID')['BHmass'].diff()
    df['dlbt']=df.groupby('ProjGalaxyID')['lookbacktime'].diff()
    df['dBHmassdt']=df.apply(lambda x: (x.dBHmass)/(x.dlbt), axis=1)
    df['dDMmass']=df.groupby('ProjGalaxyID')['DMmass'].diff()
    df['dDMmassdt']=df.apply(lambda x: (x.dDMmass)/(x.dlbt), axis=1)
    plotmovingquantiles(df, 'logmass', 'dDMmassdt', 'n2d')
    plotmovingquantiles(df, 'logmass', 'dBHmassdt', 'n2d')
    """
    

    print(df.columns.values)
    exit()

    plotmergerrate(df, 'Starmergercategory', 'lbtrounded', 'Starmass', 'Starmassmergerfrac')
    exit()

    plotbulgedisctranscolourmerger(df,1.5,'BulgeToTotal','logsSFRpermass','stargascol','stargasmergercategory', 0.5,0.05)
    plotbulgedisctranscolourmerger(df,1.5,'BulgeToTotal','logsSFRpermass','starmasscol','Starmergercategory', 0.5,0.05)
    plotbulgedisctranscolourmerger(df,1.5,'n2d','logsSFRpermass','stargascol','stargasmergercategory', 1.4, 0.1)
    plotbulgedisctranscolourmerger(df,1.5,'n2d','logsSFRpermass','starmasscol','Starmergercategory', 1.4, 0.1)

    #plotmergerrate(df, 'Starmergercategory', 'lbtrounded')
    #plotmergerrate(df, 'DMmergercategory', 'lbtrounded')
    #plotmergergasrate(df, 'stargasmergercategory', 'lbtrounded')
    #plotSFRcircles(df,'SFR1','SFR3','SFR10','SFR', 'n2d', 0, 11)
    #plotSFRcircles(df,'SFR1','SFR3m','SFR10m','SFR30m', 'n2d', 0, 5)
    #plotSFRcircles(df,'logsSFR1','logsSFR3','logsSFR10','logsSFR', 'n2d',  -11, -8)
    #plotmovinghistogram4(df,'logsSFR1','logsSFR3m','logsSFR10m','logsSFR30m', 'n2d')
    plotSFRcircles(df,'SFR1','SFR3m','SFR10m','SFR30m', 'n2d', 0,  4)
    #plotSFRcircles(df,'logsSFR1','logsSFR3m','logsSFR10m','logsSFR30m', 'n2d', -12,  -7)
    #plotSFRcircles(df,'logsSFR1A','logsSFR3mA','logsSFR10mA','logsSFR30mA', 'n2d', -6,  1)
    exit()
    #plotmovinghistogram(df, 'Starmassmergerfrac', 'n2d')
    df=df[df.logsSFR1<-1]
    df=df[df.logsSFR<-1]
    #plotmovinghistogram4(df,'SFR1','SFR3','SFR10','SFR', 'n2d')
    #plotmovinghistogram4(df,'sSFR1','sSFR3','sSFR10','sSFR', 'n2d')
    #plotmovinghistogram4(df,'logsSFR1','logsSFR3','logsSFR10','logsSFR', 'n2d')
    plotmovinghistogram4(df,'SFR1','SFR3m','SFR10m','SFR30m', 'n2d')
    plotmovinghistogram4(df,'logsSFR1','logsSFR3m','logsSFR10m','logsSFR30m', 'n2d')
    #plotSFRcircles(df,'logsSFR1','logsSFR3m','logsSFR10m','logsSFR30m', 'n2d')
    plotbulgedisctranscolourmerger(df,1.5,'BulgeToTotal','logsSFRpermass','stargascol','stargasmergercategory', 0.5,0.1)
    exit()
    print(df.columns.values)
    mergerdf= mergerinvestigation(df)
    #vdf=df.dropna(subset=['vHsqrd'])
    #vdf=vdf[vdf.vHsqrd<40]

    
    plotbulgedisctranscolourmerger(df,1.5,'BulgeToTotal','logsSFRpermass','stargascol','stargasmergercategory', 0.5,0.1)
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
    #n2dvdf=vdf[vdf.n2d>0]
    plotbulgedisctranscolourmerger(n2ddf,1.5,'n2d','logsSFRpermass','stargascol', 1.4,0.1)
    plotbulgedisctranscolourmerger(n2dmergerdf,1.5,'n2d','logsSFRpermass','stargascol', 1.4,0.1, merge=True)
    plotbulgedisctranscolourmerger(n2ddf,1.5,'n2d','logsSFRpermass','mergercol', 1.4,0.1)
    plotbulgedisctranscolourmerger(n2dmergerdf,1.5,'n2d','logsSFRpermass','mergercol', 1.4,0.1, merge=True)
    """
    colourscatter(n2dvdf, 'lbt', 'logmass', 'n2d', 1.4)
    colourscatter(n2dvdf, 'lbt', 'logDMmass', 'n2d', 1.4)
    colourscatter(n2dvdf, 'lbt', 'vHsqrd', 'n2d', 1.4)
    """

    exit()

    #threeDplot(df, 'Starmassmergerfrac','z','n2d', 'logmass', 'logsSFR')
    
    plt.hist(df.Stargasmergerfrac)
    plt.show()
    plotmovinghistogram(df, 'n2d', 'Starmassmergerfrac')
    plotmovinghistogram(df, 'logsSFR', 'Starmassmergerfrac')
    plotmovinghistogram(df, 'n2d', 'Stargasmergerfrac')
    exit()

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
    #plotfrac(df,'roundlogmass2', 'categoryn', 'catDMfrac')

    plotbulgedisctranscolour(df,0.6,'n2d','logsSFRpermass',1.5,0.1)
    plotmovingquantiles(df, 'logmass', 'logsSFR', 'n2d')
    plotmovingquantiles(df, 'logmass', 'logBHmass', 'n2d')
    plotmovingquantiles(df, 'logmass', 'logsBHmass', 'n2d')
    plotmovingquantiles(df, 'logmass', 'logDMmass', 'n2d')
    plotmovingquantiles(df, 'logmass', 'logsDMmass', 'n2d')
    plotfrac(df,'roundlogmass2', 'categoryn2d', 'catfrac')
    plotfrac(df,'roundsSFR2', 'categoryn2d', 'catsfrfrac')
    plotfrac(df,'roundBHmass', 'categoryn2d', 'catBHfrac')
    plotfrac(df,'roundDMmass2', 'categoryn2d', 'catDMfrac')
    
    #df=cleanandtransformdata(df)
    #df=df[df.sSFR>0]
    plotmultivariateplot(df)
    plotbulgedisctranscolour(df,0.6,'asymm','logsSFRpermass',0.25,0.05)
    #plotbulgedisctranscolour(df,0.6,'n_total','logsSFRpermass',1.5,0.1)
    #plotmovingquantiles(df, 'logmass', 'logsSFR', 'n_total')
    #plotmovingquantiles(df, 'logmass', 'logBHmass', 'n_total')
    #plotmovingquantiles(df, 'logmass', 'logDMmass', 'n_total')

    plotmovingquantiles(df, 'logBHmass', 'logsSFR', 'n_total')
    plotmovingquantiles(df, 'logmass', 'logBHmass', 'n_total')
    plotmovinghistogram(df, 'logsSFR', 'asymm')
    
    maxnum=df.num.max()
    #galaxyid=1430974
    #specificgalaxyplot(maxdf, galaxyid, 'DiscToTotal', 'n_total', 'logsSFR', 'loggasmass')
    #specificgalaxyplot(df, galaxyid, 'BulgeToTotal', 'n_total', 'logsSFR', 'loggasmass')
    #specificgalplotmasses(df, galaxyid)
    #specificgalplotratesofvariabless(df, galaxyid)
    evolutionplot(df, 'BulgeToTotal', 'logmass', 'logBHmass')
    threeDplot(df, 'z','DiscToTotal','logBHmass', 'Starmass', 'logsSFR')
    exit()
    
    stackedhistogram(df, 'n_total','n_disc','n_bulge','n_bulge_exp')
    #subplothistograms(df, 'n_total','n_disc','n_bulge','n_disca','n_bulgea','n_bulge_exp')
    #colorbarplot(df, 'n_total', 'DiscToTotal', 'logmass', 'logsSFR', 'BHmass')
    threeDplot(df, 'dtototal','DiscToTotal','logBHmass', 'Starmass', 'logsSFR')

    exit()
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
            df=pd.read_csv('evolvingEAGLEbulgediscmergedf'+sim_name+'.csv')
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


