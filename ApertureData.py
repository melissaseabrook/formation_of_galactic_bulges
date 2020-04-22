"""Extract Propeties of galaxies at dfferent images"""

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
con = sql.connect("mrz438", password="HDfrcZQ4")
#con = sql.connect("<username>", password="<password>")
def download_image(url, filename, sim_name, querytype):
    #download image 
    print(filename)
    if url == '':
        return('')
    else:
        urllib.request.urlretrieve(url, 'evolvinggalaxyimagebin'+''+querytype+''+sim_name+'/'+filename)
        local_path_image=os.path.join(dir_base, 'evolvinggalaxyimagebin'+''+querytype+''+sim_name+'/'+filename)
        return(local_path_image)

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
                    AP5.Mass_Star as Starmass5, \
                    AP5.Mass_Gas as Gasmass5,\
                    AP5.SFR as SFR5,\
                    AP20.Mass_Star as Starmass20, \
                    AP20.Mass_Gas as Gasmass20,\
                    AP20.SFR as SFR20\
                FROM \
                    %s_Subhalo as SH, \
                    %s_Subhalo as ref, \
                    %s_Aperture as AP5, \
                    %s_Aperture as AP20 \
                WHERE \
                    ref.MassType_Star between 1.0e10 and 1.0e11 and \
                    ref.StarFormationRate between 0.1 and 15 and \
                    ref.SnapNum=28 and \
                    ((SH.SnapNum > ref.SnapNum and ref.GalaxyID between SH.GalaxyID and SH.TopLeafID) or (SH.SnapNum <= ref.SnapNum and SH.GalaxyID between ref.GalaxyID and ref.TopLeafID)) and \
                    SH.GalaxyID = AP5.GalaxyID and \
                    SH.GalaxyID = AP20.GalaxyID and \
                    AP5.ApertureSize = 5 and\
                    AP20.ApertureSize = 20 and\
                    SH.Redshift <3 \
                ORDER BY \
                    ref.GalaxyID, \
                    SH.Redshift"%(sim_name, sim_name, sim_name, sim_name)
            myData = sql.execute_query(con , Query)
            print('gotdata')
            df=pd.DataFrame(myData, columns=['ProjGalaxyID','DescGalaxyID','DescID','z','Starmass5','Gasmass5','SFR5','Starmass20','Gasmass20','SFR20'])
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

        if querytype == 'extra':
            Query = "SELECT \
                    SH.GalaxyID as DescGalaxyID, \
                    SH.HalfMassRad_DM as DMHalfMassRad, \
                    AP.Mass_DM as DMMass100AP, \
                    FOF.Group_M_Crit500 as M500DM, \
                    FOF.Group_R_Crit500 as R500DM, \
                    FOF.Group_M_Mean200 as M200DMmean, \
                    FOF.Group_R_Crit200 as R200DMmean \
                FROM \
                    %s_Subhalo as SH, \
                    %s_Subhalo as ref, \
                    %s_Aperture as AP, \
                    %s_FOF as FOF \
                WHERE \
                    ref.MassType_Star between 1.0e10 and 1.0e11 and \
                    ref.StarFormationRate between 0.1 and 15 and \
                    ref.SnapNum=28 and \
                    ((SH.SnapNum > ref.SnapNum and ref.GalaxyID between SH.GalaxyID and SH.TopLeafID) or (SH.SnapNum <= ref.SnapNum and SH.GalaxyID between ref.GalaxyID and ref.TopLeafID)) and \
                    FOF.GroupID = SH.GroupID and \
                    SH.GalaxyID = AP.GalaxyID and \
                    SH.SubGroupNumber = 0 and \
                    AP.ApertureSize = 100 and\
                    SH.Redshift <3 \
                ORDER BY \
                    ref.GalaxyID, \
                    SH.Redshift"%(sim_name, sim_name, sim_name,sim_name)

            myData = sql.execute_query(con , Query)
            print('gotdata')
            df=pd.DataFrame(myData, columns=['DescGalaxyID','DMHalfMassRad', 'DMMass100AP','M500DM','R500DM','M200DMmean','R200DMmean'])
            print('madedf')
            df.to_csv('data'+querytype+'df'+sim_name+'.csv')
            print('done')

        if querytype == 'extra1':
            Query = "SELECT \
                    SH.GalaxyID as DescGalaxyID, \
                    AP5.Mass_DM as DMMass5, \
                    AP30.Mass_DM as DMMass30 \
                FROM \
                    %s_Subhalo as SH, \
                    %s_Subhalo as ref, \
                    %s_Aperture as AP5, \
                    %s_Aperture as AP30 \
                WHERE \
                    ref.MassType_Star between 1.0e10 and 1.0e11 and \
                    ref.StarFormationRate between 0.1 and 15 and \
                    ref.SnapNum=28 and \
                    ((SH.SnapNum > ref.SnapNum and ref.GalaxyID between SH.GalaxyID and SH.TopLeafID) or (SH.SnapNum <= ref.SnapNum and SH.GalaxyID between ref.GalaxyID and ref.TopLeafID)) and \
                    SH.GalaxyID = AP5.GalaxyID and \
                    SH.GalaxyID = AP30.GalaxyID and \
                    SH.SubGroupNumber = 0 and \
                    AP5.ApertureSize = 5 and\
                    AP30.ApertureSize = 30 and\
                    SH.Redshift <3 \
                ORDER BY \
                    ref.GalaxyID, \
                    SH.Redshift"%(sim_name, sim_name, sim_name,sim_name)

            myData = sql.execute_query(con , Query)
            print('gotdata')
            df=pd.DataFrame(myData, columns=['DescGalaxyID','DMMass5','DMMass30'])
            print('madedf')
            df.to_csv('data'+querytype+'df'+sim_name+'.csv')
            print('done')

        if querytype == 'extra2':
            Query = "SELECT \
                    ref.GalaxyID as ProjGalaxyID, \
                    SH.GalaxyID as DescGalaxyID, \
                    SH.Redshift as z, \
                    SH.SF_ThermalEnergy as SFThermalEnergy, \
                    SH.NSF_ThermalEnergy as NSFThermalEnergy, \
                    SH.SF_Mass as SFMass, \
                    SH.NSF_Mass as NSFMass, \
                    FOF.Group_M_Crit200 as M200DM, \
                    FOF.Group_R_Crit200 as R200DM \
                FROM \
                    %s_Subhalo as SH, \
                    %s_Subhalo as ref, \
                    %s_FOF as FOF \
                WHERE \
                    ref.MassType_Star between 1.0e10 and 1.0e11 and \
                    ref.StarFormationRate between 0.1 and 15 and \
                    ref.SnapNum=28 and \
                    ((SH.SnapNum > ref.SnapNum and ref.GalaxyID between SH.GalaxyID and SH.TopLeafID) or (SH.SnapNum <= ref.SnapNum and SH.GalaxyID between ref.GalaxyID and ref.TopLeafID)) and \
                    FOF.GroupID = SH.GroupID and \
                    SH.SubGroupNumber = 0 and \
                    SH.Redshift <3 \
                ORDER BY \
                    ref.GalaxyID, \
                    SH.Redshift"%(sim_name, sim_name, sim_name)
                
            myData = sql.execute_query(con , Query)
            print('gotdata')
            df=pd.DataFrame(myData, columns=['ProjGalaxyID','DescGalaxyID','z','SFThermalEnergy', 'NSFThermalEnergy','SFMass','NSFMass','M200DM','R200DM'])
            print('madedf')
            df.to_csv('data'+querytype+'df'+sim_name+'.csv')
            print('done')

                
        if querytype == 'extra3':
            Query = "SELECT \
                    SH.GalaxyID as DescGalaxyID, \
                    SH.SF_MassWeightedTemperature as SF_MassWeightedTemperature, \
                    SH.NSF_MassWeightedTemperature NSF_MassWeightedTemperature \
                FROM \
                    %s_Subhalo as SH, \
                    %s_Subhalo as ref \
                WHERE \
                    ref.MassType_Star between 1.0e10 and 1.0e11 and \
                    ref.StarFormationRate between 0.1 and 15 and \
                    ref.SnapNum=28 and \
                    ((SH.SnapNum > ref.SnapNum and ref.GalaxyID between SH.GalaxyID and SH.TopLeafID) or (SH.SnapNum <= ref.SnapNum and SH.GalaxyID between ref.GalaxyID and ref.TopLeafID)) and \
                    SH.SubGroupNumber = 0 and \
                    SH.Redshift <3 \
                ORDER BY \
                    ref.GalaxyID, \
                    SH.Redshift"%(sim_name, sim_name)


        if querytype == 'extra4':
            Query = "SELECT \
                    SH.GalaxyID as DescGalaxyID, \
                    FOF.Group_M_Crit2500 as M2500DM, \
                    FOF.Group_R_Crit2500 as R2500DM \
                FROM \
                    %s_Subhalo as SH, \
                    %s_Subhalo as ref, \
                    %s_FOF as FOF \
                WHERE \
                    ref.MassType_Star between 1.0e10 and 1.0e11 and \
                    ref.StarFormationRate between 0.1 and 15 and \
                    ref.SnapNum=28 and \
                    ((SH.SnapNum > ref.SnapNum and ref.GalaxyID between SH.GalaxyID and SH.TopLeafID) or (SH.SnapNum <= ref.SnapNum and SH.GalaxyID between ref.GalaxyID and ref.TopLeafID)) and \
                    FOF.GroupID = SH.GroupID and \
                    SH.SubGroupNumber = 0 and \
                    SH.Redshift <3 \
                ORDER BY \
                    ref.GalaxyID, \
                    SH.Redshift"%(sim_name, sim_name, sim_name)

            myData = sql.execute_query(con , Query)
            print('gotdata')
            df=pd.DataFrame(myData, columns=['DescGalaxyID','M2500DM','R2500DM'])
            print('madedf')
            print(df)
            df.to_csv('data'+querytype+'df'+sim_name+'.csv')
            print('done')

if __name__ == "__main__":
    mySims = np.array(['RefL0050N0752'])
    #querytype = allbranches or mainbranch
    querytype= 'extra4'
    getdata(mySims, querytype)
    

    