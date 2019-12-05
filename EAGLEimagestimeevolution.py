import eagleSqlTools as sql
import numpy as np
import matplotlib .pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import urllib.request
import re
import os
import pandas as pd
# Array of chosen simulations. entries refer to simulation name and comoving box length
dir_base=os.getcwd()
con = sql.connect("hwg083", password="KJHjqm91")
def download_image(url, filename, sim_name, querytype):
    print(filename)
    if url == '':
        return('')
    else:
        urllib.request.urlretrieve(url, 'evolvinggalaxyimagebin'+''+querytype+''+sim_name+'/'+filename)
        local_path_image=os.path.join(dir_base, 'evolvinggalaxyimagebin'+''+querytype+''+sim_name+'/'+filename)
        return(local_path_image)

# MW properties: M* = 5 10^10 solar masses, SFR = 3.5 solar masses per year, 
# Mh =- 1.1 10^12 solar masses:
# see https://www.annualreviews.org/doi/10.1146/annurev-astro-081915-023441
#Uses eagleSQLTools module to connect to database for username and password
#If username and password not given, module will prompt for it.
def getdata(mySims, querytype):
    for sim_name in mySims:
        print(sim_name)
        if querytype =='mainbranch':
            Query = "SELECT \
                    ref.GalaxyID as ProjGalaxyID, \
                    SH.GalaxyID as DescGalaxyID, \
                    SH.Redshift as z, \
                    SH.Stars_Metallicity as Z, \
                    SH.Image_Face as face, \
                    SH.HalfMassRad_Star as HalfMassRadius, \
                    AP.VelDisp as VelDisp, \
                    AP.Mass_Star as Starmass, \
                    AP.Mass_BH as BHmass, \
                    AP.Mass_DM as DMmass, \
                    AP.Mass_Gas as Gasmass, \
                    AP.SFR as SFR, \
                    SH.StellarInitialMass as StellarInitialMass, \
                    SH.BlackHoleMassAccretionRate as BHAccretionrate, \
                    SH.Vmax as Vmax, \
                    SH.VmaxRadius as Vmaxradius, \
                    MK.DiscToTotal as DiscToTotal,\
                    MK.DispAnisotropy as DispAnisotropy, \
                    MK.DMEllipticity as DMEllipticity, \
                    MK.Ellipticity as StellarEllipticity, \
                    MK.KappaCoRot as StellarCoRotKE, \
                    MK.MedOrbitCircu as MedOrbitCircu, \
                    MK.RotToDispRatio as RotToDispRatio, \
                    MK.Triaxiality as Triaxiality \
                FROM \
                    %s_Subhalo as SH, \
                    %s_Subhalo as ref, \
                    %s_Aperture as AP, \
                    %s_MorphoKinem as MK \
                WHERE \
                    ref.MassType_Star between 1.0e10 and 8.0e10 and \
                    ref.StarFormationRate between 0.1 and 15 and \
                    ref.MassType_BH between 1.0e6 and 1.0e7 and \
                    ref.SnapNum=28 and \
                    ((SH.SnapNum > ref.SnapNum and ref.GalaxyID between SH.GalaxyID and SH.TopLeafID) or (SH.SnapNum <= ref.SnapNum and SH.GalaxyID between ref.GalaxyID and ref.TopLeafID)) and \
                    SH.GalaxyID = AP.GalaxyID and \
                    SH.GalaxyID = MK.GalaxyID and \
                    AP.ApertureSize = 30 and\
                    ref.Image_face IS NOT null\
                ORDER BY \
                    SH.Redshift"%( sim_name , sim_name, sim_name, sim_name)

        elif querytype == 'allbranches':
            Query = "SELECT \
                    ref.GalaxyID as ProjGalaxyID, \
                    SH.GalaxyID as DescGalaxyID, \
                    SH.Redshift as z, \
                    SH.Stars_Metallicity as Z, \
                    SH.Image_Face as face, \
                    SH.HalfMassRad_Star as HalfMassRadius, \
                    AP.VelDisp as VelDisp, \
                    AP.Mass_Star as Starmass, \
                    AP.Mass_BH as BHmass, \
                    AP.Mass_DM as DMmass, \
                    AP.Mass_Gas as Gasmass, \
                    AP.SFR as SFR, \
                    SH.StellarInitialMass as StellarInitialMass, \
                    SH.BlackHoleMassAccretionRate as BHAccretionrate, \
                    SH.Vmax as Vmax, \
                    SH.VmaxRadius as Vmaxradius, \
                    MK.DiscToTotal as DiscToTotal,\
                    MK.DispAnisotropy as DispAnisotropy, \
                    MK.DMEllipticity as DMEllipticity, \
                    MK.Ellipticity as StellarEllipticity, \
                    MK.KappaCoRot as StellarCoRotKE, \
                    MK.MedOrbitCircu as MedOrbitCircu, \
                    MK.RotToDispRatio as RotToDispRatio, \
                    MK.Triaxiality as Triaxiality \
                FROM \
                    %s_Subhalo as SH, \
                    %s_Subhalo as ref, \
                    %s_Aperture as AP, \
                    %s_MorphoKinem as MK \
                WHERE \
                    ref.MassType_Star between 1.0e10 and 8.0e10 and \
                    ref.StarFormationRate between 0.1 and 15 and \
                    ref.MassType_BH between 1.0e6 and 1.0e7 and \
                    ref.SnapNum=28 and \
                    SH.GalaxyID between ref.GalaxyID and ref.LastProgID and \
                    SH.GalaxyID = AP.GalaxyID and \
                    SH.GalaxyID = MK.GalaxyID and \
                    AP.ApertureSize = 30 and\
                    ref.Image_face IS NOT null\
                ORDER BY \
                    ref.GalaxyID, \
                    SH.Redshift"%( sim_name , sim_name, sim_name, sim_name)

        myData = sql.execute_query (con , Query)
        df=pd.DataFrame(myData, columns=['ProjGalaxyID','DescGalaxyID','z','Z','face','mass', 'DiscToTotal', 'HalfMassRadius','VelDisp','BHmass','SFR'])
        df['face']=  df['face'].str.decode("utf-8")
        df['face']=df['face'].str.replace('"<img src=', '').str.replace('>"', '').str.replace("'",'')
        df=df.assign(name1 = lambda x: x.face)
        df['name1']=df['name1'].str.replace('http://virgodb.cosma.dur.ac.uk/eagle-webstorage/'+sim_name+'_Subhalo/', '')
        df=df.assign(filename = lambda x: sim_name +'' + x.name1)
        df['image']=df.apply(lambda x: download_image(x.face, x.filename, sim_name, querytype), axis=1)
        df.to_csv('evolvingEAGLEimages'+querytype+'df'+sim_name+'.csv')
        print(df['image'])
   
    


if __name__ == "__main__":
    mySims = np.array (['RecalL0025N0752'])
    #querytype = allbranches or mainbranch
    querytype= 'allbranches'
    getdata(mySims, querytype)
    

    