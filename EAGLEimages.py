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
mySims = np.array ([('RecalL0025N0752', 25.)])
dir_base=os.getcwd()
def download_image(url, filename):
    urllib.request.urlretrieve(url, 'galaxyimagebinRecal/'+filename)
    local_path_image=os.path.join(dir_base, 'galaxyimagebinRecal/'+filename)
    return(local_path_image)

# MW properties: M* = 5 10^10 solar masses, SFR = 3.5 solar masses per year, 
# Mh =- 1.1 10^12 solar masses:
# see https://www.annualreviews.org/doi/10.1146/annurev-astro-081915-023441
#Uses eagleSQLTools module to connect to database for username and password
#If username and password not given, module will prompt for it.
con = sql.connect("hwg083", password="KJHjqm91")
for sim_name,sim_size in mySims:
    print(sim_name)
    
    myQuery = "SELECT \
                    SH.Redshift as z, \
                    SH.Stars_Metallicity as Z, \
                    SH.Image_Face as face, \
                    AP.Mass_Star as mass \
                FROM \
                    %s_Subhalo as SH, \
                    %s_Aperture as AP \
                WHERE \
                    SH.GalaxyID = AP.GalaxyID and \
                    AP.ApertureSize = 30 and \
                    SH.MassType_Star between 2.0e10 and 8.0e10 and \
                    AP.Mass_Star between 2.0e10 and 8.0e10 and \
                    SH.StarFormationRate between 0.1 and 10 and \
                    SH.SnapNum = 28 and \
                    SH.Image_face IS NOT null\
                ORDER BY \
                    AP.Mass_Star"%( sim_name , sim_name)
    # Execute query .
    myData = sql.execute_query (con , myQuery)
    df=pd.DataFrame(myData, columns=['z','Z','face','mass'])
    df['face']=  df['face'].str.decode("utf-8")
    df['face']=df['face'].str.replace('"<img src=', '').str.replace('>"', '').str.replace("'",'')
    df=df.assign(name1 = lambda x: x.face)
    df['name1']=df['name1'].str.replace('http://virgodb.cosma.dur.ac.uk/eagle-webstorage/'+sim_name+'_Subhalo/', '')
    df=df.assign(filename = lambda x: sim_name +'' + x.name1)
    df['image']=df.apply(lambda x: download_image(x.face, x.filename), axis=1)
    print(df['image'])
