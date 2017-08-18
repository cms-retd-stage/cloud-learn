#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 4 août 2017

@author: almeidamanceroi
'''

import h5py
#import sys
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import math
import time

#from gestion_variables_cibles import jour_de_l_annee
#import sys
#import netCDF4
#from pyproj import Proj

# def readVIIRSMAIA(viirs_M_file,code_maia):
def readVIIRSMAIA(viirs_M_file):
    '''
    Cree le fichier d'entrainement MAIA en selectionnant que les donnees fiables
    '''
    
    f = h5py.File(viirs_M_file)  
#     myh5=h5py.File(code_maia)
    
    df = pd.DataFrame()
      
#     filepath1 = "/rd/merzhin/safo/retraitement/clim_OSTIA/D001-UKMO-L4HRfnd-GLOB-v01-fv02-OSTIARANclim_m.nc"
#     with netCDF4.Dataset(filepath1) as nc:
#         liste_lon = nc.variables['lon'][:]
#         liste_lat = nc.variables['lat'][:]
#     
     
    '''
    #creation Obs
    '''
 
    obs = "Obs_04 Obs_05 Obs_06 Obs_08 Obs_13 Obs_16 Obs_22 Obs_37 Obs_40 Obs_87 Obs_108 Obs_120"   
    channels = {"M1" : "Obs_04",
                "M4" : "Obs_05",
                "M5" : "Obs_06",
                "M7" : "Obs_08",
                "M9" : "Obs_13",
                "M10": "Obs_16",
                "M11": "Obs_22",
                "M12": "Obs_37",
                "M13": "Obs_40",
                "M14": "Obs_87",
                "M15": "Obs_108",
                "M16": "Obs_120"}
    #allObs = {}
    for chan,obs in channels.items():
        if chan in ["M1","M4","M5","M7","M9","M10", "M11"]:
            datasetName = "All_Data/VIIRS-" + chan + "-SDR_All/Reflectances"
        else:
            datasetName = "All_Data/VIIRS-" + chan + "-SDR_All/BrightnessTemperature"
        print datasetName
        #allObs[obs] = f[datasetName][:]
        df[obs]=f[datasetName][:].flatten()[:1228800]
        print len(df[obs])
         
    '''
    #creation entete
    '''
     
    print 'creation entete'  
     
    df["pix_lon"] = f["All_Data/VIIRS-MOD-GEO_All/Longitude"][:].flatten()[:1228800]
    df["pix_lat"] = f["All_Data/VIIRS-MOD-GEO_All/Latitude"][:].flatten()[:1228800]
    df["pix_satzen"] = f["All_Data/VIIRS-MOD-GEO_All/SatelliteZenithAngle"][:].flatten()[:1228800]
    df["pix_sunzen"] = f["All_Data/VIIRS-MOD-GEO_All/SolarZenithAngle"][:].flatten()[:1228800]
    df["pix_satazi"] = f["All_Data/VIIRS-MOD-GEO_All/SatelliteAzimuthAngle"][:].flatten()[:1228800]
    df["pix_sunazi"]= f["All_Data/VIIRS-MOD-GEO_All/SolarAzimuthAngle"][:].flatten()[:1228800]
    df["pix_alt"] = f["All_Data/VIIRS-MOD-GEO_All/Height"][:].flatten()[:1228800]
     
     
    '''
    #affinage des Obs reflectance
    '''
     
    print 'affinage des Obs reflectance'
    for obs in ['Obs_04','Obs_05','Obs_06','Obs_08','Obs_13','Obs_16','Obs_22']:
        df[obs]=df[obs]*100*np.cos(df['pix_sunzen']*math.pi/180)
        NaN=np.logical_or(df[obs]<=-999,df[obs]>999)
        df[obs][NaN]=-999.9
     
    '''
    #creation pix_tm
    '''
      
      
    print 'creation pix_tm'    
    TM = h5py.File('/rd/merzhin/safnwp/pascale/ATLAS/landsea.h5')
      
    latatlas=np.round((df['pix_lat']-90)*17999/-179.98,0)
    lonatlas=np.round((df['pix_lon']+180)*35999/359.98,0)
      
    taille=df.shape[0]
    print 'taille:',taille
    Atlas=np.zeros(taille)
      
    for i in xrange(taille):
        if (latatlas[i]>17999) or (lonatlas[i]>35999):
            Atlas[i]=99
        else:
            Atlas[i]=TM['landsea'][latatlas[i],lonatlas[i]].flatten()
      
      
    df['pix_tm']=Atlas
      
    terre=(df['pix_tm'] == 1)
    desert=(df['pix_tm'] == 2)
    snow=(df['pix_tm'] == 3)
    coast=(df['pix_tm'] == 4)
    lake=(df['pix_tm'] == 5)
      
    df['pix_tm'][terre]=2
    df['pix_tm'][desert]=3
    df['pix_tm'][snow]=4
    df['pix_tm'][coast]=5
    df['pix_tm'][lake]=1
      
     
    '''
    #creation daytime
    '''
     
    print 'creation daytime'
         
    nuit=(df['pix_sunzen']>=90)
    aube=np.logical_and((df['pix_sunzen']<90),(df['pix_sunzen']>83))
    jour=(df['pix_sunzen']<=83)
     
    df['pix_daytime']=np.zeros(np.shape(df['pix_sunzen'])[0])
    df['pix_daytime'][nuit]=0
    df['pix_daytime'][aube]=1
    df['pix_daytime'][jour]=2
     
    '''
    #creation sst
    '''
#===============================================================================
#      
#     print 'creation sst'
#     #constantes pour le calcul de la sst:
#     an=1.01612
#     bn=0.01709
#     cn=0.85154
#     dn=0.36969
#     en=1.15754
#     fn=0.86056
# #     aj=1.00055
# #     bj=0.00852
# #     cj=1.29073
# #     dj=0.77930
# #     ej=0.04010
# #     fj=1.29304
# #     gj=0.65369
#     S=(1./np.cos(df["pix_satzen"]*(math.pi/180)))-1 #S=sécante-1  
#     jouretaube=(df['pix_sunzen']<90)
#     df['sst']=np.zeros(np.shape(df['pix_sunzen'])[0])
#     df['sst'][nuit]=((an+bn*S[nuit])*df["Obs_37"][nuit]+(cn+dn*S[nuit])*(df["Obs_108"][nuit]-df["Obs_120"][nuit])+en+fn*S[nuit])-273.15
#     df['sst'][jouretaube]=-999 #on ne peut pas calculer pour jour et aube car il faut une valeur de température
#       
#===============================================================================
     
    '''
    #creation variances (idem que pour SST Clim)
    '''
      
    print 'creation variances'
     
    all_vars =['var_04','var_05','var_06','var_08','var_13','var_16','var_22','var_37','var_40','var_87','var_108','var_120']
    count=0
    for chan,obs in channels.items():
        vari=[]
        if chan in ["M1","M4","M5","M7","M9","M10", "M11"]:
            datasetName = "All_Data/VIIRS-" + chan + "-SDR_All/Reflectances"
        else:
            datasetName = "All_Data/VIIRS-" + chan + "-SDR_All/BrightnessTemperature"
        print 'var',datasetName
         
        vari.append(np.var([f[datasetName][0,0],f[datasetName][0,1],f[datasetName][1,0],f[datasetName][1,1]]))
        for k in range(1,3199):
            vari.append(np.var([f[datasetName][0,k-1],f[datasetName][0,k],f[datasetName][0,k+1],f[datasetName][1,k-1],f[datasetName][1,k],f[datasetName][1,k+1]]))
        vari.append(np.var([f[datasetName][0,3198],f[datasetName][0,3199],f[datasetName][1,3198],f[datasetName][1,3199]]))
         
        for j in xrange(1,383):
            vari.append(np.var([f[datasetName][j-1,0],f[datasetName][j-1,1],f[datasetName][j,0],f[datasetName][j,1],f[datasetName][j+1,0],f[datasetName][j+1,1]]))
            for k in range(1,3199):
                vari.append(np.var([f[datasetName][j-1,k-1],f[datasetName][j,k-1],f[datasetName][j+1,k-1],
                            f[datasetName][j-1,k],f[datasetName][j,k],f[datasetName][j+1,k],
                            f[datasetName][j-1,k+1],f[datasetName][j,k+1],f[datasetName][j+1,k+1]]))
            vari.append(np.var([f[datasetName][j-1,3198],f[datasetName][j-1,3199],f[datasetName][j,3198],f[datasetName][j,3199],f[datasetName][j+1,3198],f[datasetName][j+1,3199]]))
         
        vari.append(np.var([f[datasetName][382,0],f[datasetName][382,1],f[datasetName][383,0],f[datasetName][383,1]]))
        for k in range(1,3199):
            vari.append(np.var([f[datasetName][382,k-1],f[datasetName][382,k],f[datasetName][382,k+1],f[datasetName][383,k-1],f[datasetName][383,k],f[datasetName][383,k+1]]))
        vari.append(np.var([f[datasetName][382,3198],f[datasetName][382,3199],f[datasetName][383,3198],f[datasetName][383,3199]]))
         
        df[all_vars[count]]=vari
        count+=1
          
     
    '''
    #creation SST Clim (maxsst, minsst et ansst) (annulé car itération prenaient trop de temps (6 jours))
    '''
    #===========================================================================
    # print 'creation SST clim'
    # mm=6 #mois
    # dd=27 #jour
    # filepathclim = '/rd/merzhin/safo/retraitement/clim_OSTIA/D'+jour_de_l_annee(mm,dd)+'-UKMO-L4HRfnd-GLOB-v01-fv02-OSTIARANclim_m.nc'
    # lo=df["pix_lon"] #longitude
    # lati=df["pix_lat"] #latitude
    # #maxs=[]
    # #mins=[]
    # ans=[]
    # k=0
    # taille=df.shape[0]
    # while k<taille:
    #     if k%100==0:
    #         print k
    #     if k%5==0:
    #         ind_lon=0
    #         ind_lat=0
    #         while liste_lon[ind_lon]<lo[k]:
    #             ind_lon+=1
    #         while liste_lat[ind_lat]<lati[k]:
    #             ind_lat+=1
    #         with netCDF4.Dataset(filepathclim) as nc:
    #             #maxsst=nc.variables['maximum_sst'][0,ind_lat,ind_lon]
    #             #minsst=nc.variables['minimum_sst'][0,ind_lat,ind_lon]
    #             ansst=nc.variables['analysed_sst'][0,ind_lat,ind_lon]
    #         #if not isinstance(maxsst, np.float32):
    #         #    maxsst=0
    #         #if not isinstance(minsst, np.float32):
    #         #    minsst=0
    #         if not isinstance(ansst, np.float32):
    #             ansst=0
    #         #maxs.append(maxsst)
    #         #mins.append(minsst)
    #         ans.append(ansst)
    #         ans.append(ansst)
    #         ans.append(ansst)
    #         ans.append(ansst)
    #         ans.append(ansst)
    #     k+=1
    # #df["maxsst"] = maxs
    # #df["minsst"] = mins
    # df["ansst"] = ans      
    #===========================================================================
       
    '''
    ----------------------------------------------------------------------------------------------------
    '''
#     
#     print 'fov qual' 
#     mask=2**0+2**1    
#     lst=myh5[u'DATA/fov_qual'][:]&mask
#     lst=lst/2**0
#     df['FOV']=lst.flatten()
#     
#     
#     print 'cloud mask confidence'
#     mask=2**5+2**6
#     lst=myh5[u'DATA/CloudMask'][:]&mask
#     lst=lst/2**5
#     df['CMC']=lst.flatten()
#     
#     
#     print 'cloud type'
#     mask=2**4+2**5+2**6+2**7+2**8
#     lst=myh5[u'DATA/CloudType'][:]&mask
#     lst=lst/2**4
#     df['code']=lst.flatten()


    '''
    #creation du fichier
    '''

    print 'creation du fichier'
    
    print df.head()  
    #print df[150:175]   
    
    pd.DataFrame.to_csv(df,'/rd/merzhin/safnwp/isabel_data/test_read_viirs4_vars_demi.txt', sep=',',index=False)
    return 1
'''
-------------------------------------------------------------------------------------------------------------
'''


if __name__ == '__main__':
    start_time = time.time()
    print 'départ'
    pd.options.mode.chained_assignment = None  # default='warn'
    readVIIRSMAIA('/rd/merzhin/safnwp/pascale/AAPP/SITUATIONS/20160627_t0140079/input/canaux_m.h5')
    
    
    tps_final=round(time.time()-start_time,2)
    print("--- %s seconds---" % tps_final)
    print '----end----'
    
