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
#con = sql.connect("<username>", password="<password>")
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

        if querytype == 'mainbranch':
            Query = "SELECT \
                    ref.GalaxyID as ProjGalaxyID, \
                    SH.GalaxyID as DescGalaxyID, \
                    SH.DescendantID as DescID, \
                    SH.Redshift as z, \
                    AP1.Mass_Star as Starmass1, \
                    AP1.Mass_Gas as Gasmass1,\
                    AP1.SFR as SFR1,\
                    AP3.Mass_Star as Starmass3, \
                    AP3.Mass_Gas as Gasmass3,\
                    AP3.SFR as SFR3,\
                    AP10.Mass_Star as Starmass10, \
                    AP10.Mass_Gas as Gasmass10,\
                    AP10.SFR as SFR10\
                FROM \
                    %s_Subhalo as SH, \
                    %s_Subhalo as ref, \
                    %s_Aperture as AP1, \
                    %s_Aperture as AP3, \
                    %s_Aperture as AP10 \
                WHERE \
                    ref.MassType_Star between 1.0e10 and 1.0e11 and \
                    ref.StarFormationRate between 0.1 and 15 and \
                    ref.SnapNum=28 and \
                    ((SH.SnapNum > ref.SnapNum and ref.GalaxyID between SH.GalaxyID and SH.TopLeafID) or (SH.SnapNum <= ref.SnapNum and SH.GalaxyID between ref.GalaxyID and ref.TopLeafID)) and \
                    SH.GalaxyID = AP1.GalaxyID and \
                    SH.GalaxyID = AP3.GalaxyID and \
                    SH.GalaxyID = AP10.GalaxyID and \
                    AP1.ApertureSize = 1 and\
                    AP3.ApertureSize = 3 and\
                    AP10.ApertureSize = 10 and\
                    SH.Redshift <3 \
                ORDER BY \
                    ref.GalaxyID, \
                    SH.Redshift"%(sim_name, sim_name, sim_name, sim_name, sim_name)
            myData = sql.execute_query(con , Query)
            print('gotdata')
            df=pd.DataFrame(myData, columns=['ProjGalaxyID','DescGalaxyID','DescID','z','Starmass1', 'Gasmass1','SFR1','Starmass3','Gasmass3','SFR3', 'Starmass10', 'Gasmass10','SFR10'])
            print('madedf')
            df.to_csv('aperture'+querytype+'df'+sim_name+'.csv')
            print('done')

        if querytype == 'reduced':
            Query = "SELECT \
                    ref.GalaxyID as ProjGalaxyID, \
                    SH.GalaxyID as DescGalaxyID, \
                    SH.DescendantID as DescID, \
                    SH.Redshift as z, \
                    AP1.Mass_Star as Starmass1, \
                    AP1.Mass_BH as BHmass1, \
                    AP1.Mass_DM as DMmass1, \
                    AP1.Mass_Gas as Gasmass1,\
                    AP1.SFR as SFR1,\
                    AP3.Mass_Star as Starmass3, \
                    AP3.Mass_BH as BHmass3, \
                    AP3.Mass_DM as DMmass3, \
                    AP3.Mass_Gas as Gasmass3,\
                    AP3.SFR as SFR3\
                FROM \
                    %s_Subhalo as SH, \
                    %s_Subhalo as ref, \
                    %s_Aperture as AP1, \
                    %s_Aperture as AP3 \
                WHERE \
                    ref.MassType_Star between 1.0e10 and 1.0e11 and \
                    ref.StarFormationRate between 0.1 and 15 and \
                    ref.SnapNum=28 and \
                    ((SH.SnapNum > ref.SnapNum and ref.GalaxyID between SH.GalaxyID and SH.TopLeafID) or (SH.SnapNum <= ref.SnapNum and SH.GalaxyID between ref.GalaxyID and ref.TopLeafID)) and \
                    SH.GalaxyID = AP1.GalaxyID and \
                    SH.GalaxyID = AP3.GalaxyID and \
                    AP1.ApertureSize = 1 and\
                    AP3.ApertureSize = 3 and\
                    SH.Redshift <3 \
                ORDER BY \
                    ref.GalaxyID, \
                    SH.Redshift"%(sim_name, sim_name, sim_name, sim_name)
                
            myData = sql.execute_query(con , Query)
            print('gotdata')
            df=pd.DataFrame(myData, columns=['ProjGalaxyID','DescGalaxyID','DescID','z','Starmass1', 'BHmass1','DMmass1','Gasmass1','SFR1','Starmass3', 'BHmass3','DMmass3','Gasmass3','SFR3'])
            print('madedf')
            df.to_csv('aperture'+querytype+'df'+sim_name+'.csv')
            print('done')
    


if __name__ == "__main__":
    mySims = np.array(['RefL0050N0752'])
    #querytype = allbranches or mainbranch
    querytype= 'mainbranch'
    getdata(mySims, querytype)
    

    