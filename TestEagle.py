"""Initial test of data extraction for the EAGLE database"""

import eagleSqlTools as sql
import numpy as np
import matplotlib .pyplot as plt
import urllib

mySims = np.array ([('RefL0050N0752', 50.)])

con = sql.connect("hwg083", password="KJHjqm91")
for sim_name , sim_size in mySims:
    print(sim_name)
    myQuery = "SELECT TOP 100 \
                    ref.GalaxyID as ProjGalaxyID, \
                    SH.GalaxyID as DescGalaxyID, \
                    SH.DescendantID as DescID, \
                    SH.Redshift as z, \
                    SH.Image_Face as face, \
                    SH.HalfMassRad_Star as HalfMassRadius, \
                    AP.VelDisp as VelDisp, \
                    AP.SFR as SFR, \
                    SH.StellarInitialMass as StellarInitialMass, \
                    SH.BlackHoleMassAccretionRate as BHAccretionrate, \
                    SH.VmaxRadius as Vmaxradius, \
                    MK.DiscToTotal as DiscToTotal\
                FROM \
                    %s_Subhalo as SH, \
                    %s_Subhalo as ref, \
                    %s_Aperture as AP, \
                    %s_MorphoKinem as MK \
                WHERE \
                    ref.MassType_Star between 1.0e10 and 2.0e10 and \
                    ref.StarFormationRate between 0.1 and 5 and \
                    ref.MassType_BH between 1.0e6 and 1.0e7 and \
                    ref.SnapNum=28 and \
                    ((SH.SnapNum > ref.SnapNum and ref.GalaxyID between SH.GalaxyID and SH.TopLeafID) or (SH.SnapNum <= ref.SnapNum and SH.GalaxyID between ref.GalaxyID and ref.TopLeafID)) and \
                    SH.GalaxyID = AP.GalaxyID and \
                    SH.GalaxyID = MK.GalaxyID and \
                    AP.ApertureSize = 30 and\
                    ref.Image_face IS NOT null and\
                    ref.GalaxyID in (SELECT TOP 10\
                                        refA.GalaxyID as ProjGalaxyID \
                                    FROM \
                                        %s_Subhalo as SHA, \
                                        %s_Subhalo as refA\
                                    WHERE \
                                        refA.MassType_Star between 1.0e10 and 2.0e10 and \
                                        refA.StarFormationRate between 0.1 and 1 and \
                                        refA.MassType_BH between 1.0e6 and 1.0e7 and \
                                        refA.SnapNum=28 and \
                                        ((SHA.SnapNum > refA.SnapNum and refA.GalaxyID between SHA.GalaxyID and SHA.TopLeafID) or (SHA.SnapNum <= refA.SnapNum and SHA.GalaxyID between refA.GalaxyID and refA.TopLeafID)) \
                                    GROUP BY \
                                        refA.GalaxyID \
                                    HAVING COUNT(refA.GalaxyID)>25) \
                ORDER BY \
                    SH.Redshift"%(sim_name, sim_name, sim_name, sim_name, sim_name, sim_name)
    # Execu te que ry .
    myData = sql.execute_query (con , myQuery)
    # Norma l ize by volume and b in w id th .
    plt.plot(myData['SFR'], myData['Starmass'], label= sim_name , linewidth =2)
    #plt.xlabel(r'log_${10}$ M$ {∗}$ [ M$ {\odot}$]', fontsize =20)
    #plt.ylabel(r'log_${10}$ dn/dlog$ {10}$( M$ {∗}$) [cMpc$^{−3}$]', fontsize =20)
    plt.tight_layout()
    plt.legend()
    plt.show()
