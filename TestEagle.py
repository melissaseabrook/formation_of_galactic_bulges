# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:24:33 2019

@author: Melissa
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import eagleSqlTools as sql
import numpy as np
import matplotlib .pyplot as plt
# Array of chosen simulations. entries refer to simulation name and comoving box length
mySims = np.array ([('RefL0100N1504',100.),('AGNdT9L0050N0752',50.),('RecalL0025N0752', 25.)])
#Uses eagleSQLTools module to connect to database for username and password
#If username and password not given, module will prompt for it.
con = sql.connect("hwg083", password="KJHjqm91")
for sim_name,sim_size in mySims:
    print(sim_name)
    # construct and execute query for eacj simulation. This query returns the number of galaxies 
    # for a given 30 pkpc aperture stellar mass bin (centred with 0.2 dex width)
    myQuery = "SELECT TOP 100\
                    SH.VmaxRadius as r_max, \
                	SH.Vmax as v_max \
                FROM \
                    %s_SubHalo as SH, \
                    %s_Aperture as AP \
                WHERE \
                    SH.GalaxyID = AP.GalaxyID and \
                    AP.ApertureSize = 30 and \
                    AP.Mass_Star > 1e8 and \
                    AP.Mass_Star < 1e9 and \
                    SH.SnapNum = 27 \
                ORDER BY VmaxRadius"%(sim_name, sim_name)

    # Execute query .
    myData = sql.execute_query (con , myQuery)
    print(myData[0:20])
    # Normalize by volume and bin width .
    plt.scatter(myData['r_max'], np.log10(myData['v_max']), label= sim_name , linewidth =0.1, alpha=0.8)
    
# Label Plot
plt.xlabel(r"log$_{10}$ M$_{∗}$ [M$_{\odot}$]", fontsize =20)
plt.ylabel(r"log$_{10}$ dn/dlog$_{10}$(M$_{∗}$) [cMpc$^{−3}$]", fontsize =20)
plt.tight_layout()
plt.legend()
plt.show()
plt.savefig('test.png')
plt.close()