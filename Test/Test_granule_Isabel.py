#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 10 août 2017

@author: almeidamanceroi
'''

import pandas as pd
import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import copy
import math
import cPickle
import time

from PIL import Image

def liste_peigne(viirs):
    '''
    Ajoute l'effet de peigne a l'image finale
    '''
    A=np.logical_and(viirs['Obs_04']<=-999,viirs['Obs_05']<=-999)
    B=np.logical_and(viirs['Obs_06']<=-999,viirs['Obs_08']<=-999)
    C=np.logical_and(viirs['Obs_13']<=-999,viirs['Obs_16']<=-999)
    D=np.logical_and(viirs['Obs_22']<=-999,viirs['Obs_37']<=-999)
    E=np.logical_and(viirs['Obs_40']<=-999,viirs['Obs_87']<=-999)
    F=np.logical_and(viirs['Obs_108']<=-999,viirs['Obs_120']<=-999)
    
    G=np.logical_and(A,B)
    H=np.logical_and(C,D)
    I=np.logical_and(E,F)
    
    J=np.logical_and(G,H)
    peigne=np.logical_and(I,J)
    
#     viirs.loc[peigne,'code']=20
#     viirs.loc[0,'code']=1
#     viirs.loc[1,'code']=20
#     return viirs    
    return peigne


def SaveCode(code,name='test_granule.ppm'):
    '''
    Enregistre le code de la granule sous forme d'image
    '''
    code=pd.Series(code)
    viirs=pd.DataFrame(code)
    viirs['rouge']=np.zeros(len(code))
    viirs['vert']=np.zeros(len(code))
    viirs['bleu']=np.zeros(len(code))
    
    C01= (code == 1)
    viirs['rouge'][C01]=int(0x00)
    viirs['vert'][C01]=int(0x8c)   
    viirs['bleu'][C01]=int(0x30)   
    #008c30
    
    C02= (code == 2)
    viirs['rouge'][C02]=int(0x00)
    viirs['vert'][C02]=int(0x00)   
    viirs['bleu'][C02]=int(0x00)
    #black
        
    C03= (code == 3)
    viirs['rouge'][C03]=int(0xff)
    viirs['vert'][C03]=int(0xbb)   
    viirs['bleu'][C03]=int(0xff)
    #ffbbff
        
    C04= (code == 4)
    viirs['rouge'][C04]=int(0xdd)
    viirs['vert'][C04]=int(0xa0)   
    viirs['bleu'][C04]=int(0xdd)
    #dda0dd
        
    C05= (code == 5)
    viirs['rouge'][C05]=int(0xff)
    viirs['vert'][C05]=int(0xa5)   
    viirs['bleu'][C05]=int(0x00)
    #ffa500
        
    C06= (code == 6)
    viirs['rouge'][C06]=int(0xff)
    viirs['vert'][C06]=int(0x66)   
    viirs['bleu'][C06]=int(0x00)
    #ff6600
        
    C07= (code == 7)
    viirs['rouge'][C07]=int(0xff)
    viirs['vert'][C07]=int(0xd8)   
    viirs['bleu'][C07]=int(0x00)
    #ffd800
        
    C08= (code == 8)
    viirs['rouge'][C08]=int(0xff)
    viirs['vert'][C08]=int(0xb6)   
    viirs['bleu'][C08]=int(0x00)
    #ffb600
        
    C09= (code == 9)
    viirs['rouge'][C09]=int(0xff)
    viirs['vert'][C09]=int(0xff)   
    viirs['bleu'][C09]=int(0x00)
    #ffff00
        
    C10= (code ==10 )
    viirs['rouge'][C10]=int(0xd8)
    viirs['vert'][C10]=int(0xff)   
    viirs['bleu'][C10]=int(0x00)
    #d8ff00
        
    C11= (code == 11)
    viirs['rouge'][C11]=int(0xcc)
    viirs['vert'][C11]=int(0xcc)   
    viirs['bleu'][C11]=int(0x00)
    #cccc00
        
    C12= (code == 12)
    viirs['rouge'][C12]=int(0xd8)
    viirs['vert'][C12]=int(0xb5)   
    viirs['bleu'][C12]=int(0x75)
    #d8b575
        
    C13= (code == 13)
    viirs['rouge'][C13]=int(0xff)
    viirs['vert'][C13]=int(0xff)   
    viirs['bleu'][C13]=int(0xff)
    #ffffff
        
    C14= (code == 14)
    viirs['rouge'][C14]=int(0xff)
    viirs['vert'][C14]=int(0xe0)   
    viirs['bleu'][C14]=int(0xaa)
    #ffe0aa
        
    C15= (code ==15 )
    viirs['rouge'][C15]=int(0x00)
    viirs['vert'][C15]=int(0x00)   
    viirs['bleu'][C15]=int(0xff)
    #0000ff
        
    C16= (code == 16)
    viirs['rouge'][C16]=int(0x00)
    viirs['vert'][C16]=int(0xb2)   
    viirs['bleu'][C16]=int(0xff)
    #00b2ff
        
    C17= (code == 17)
    viirs['rouge'][C17]=int(0x00)
    viirs['vert'][C17]=int(0xff)   
    viirs['bleu'][C17]=int(0xe5)
    #00ffe5
        
    C18= (code ==18 )
    viirs['rouge'][C18]=int(0x00)
    viirs['vert'][C18]=int(0xff)   
    viirs['bleu'][C18]=int(0xb2)
    #00ffb2
    
    C19= (code ==19 )
    viirs['rouge'][C19]=int(0xd8)
    viirs['vert'][C19]=int(0x00)   
    viirs['bleu'][C19]=int(0xff)
    
    C20= (code ==20 )
    viirs['rouge'][C20]=int(0x66)
    viirs['vert'][C20]=int(0x0f)   
    viirs['bleu'][C20]=int(0x00)
    
#     rgbArray= np.zeros((768,3200,3), 'uint8')
#     rouge=np.reshape(viirs['rouge'], (768,3200))
#     vert=np.reshape(viirs['vert'], (768,3200))
#     bleu=np.reshape(viirs['bleu'], (768,3200))

#     rgbArray= np.zeros((384,3200,3), 'uint8') #correspond a un demi granule
#     rouge=np.reshape(viirs['rouge'], (384,3200))
#     vert=np.reshape(viirs['vert'], (384,3200))
#     bleu=np.reshape(viirs['bleu'], (384,3200))
    
    
#     rgbArray= np.zeros((192,3200,3), 'uint8') #correspond a un quart de granule
#     rouge=np.reshape(viirs['rouge'], (192,3200))
#     vert=np.reshape(viirs['vert'], (192,3200))
#     bleu=np.reshape(viirs['bleu'], (192,3200))
    
    
    rgbArray= np.zeros((288,3200,3), 'uint8') #correspond a 3/8 de granule
    rouge=np.reshape(viirs['rouge'], (288,3200))
    vert=np.reshape(viirs['vert'], (288,3200))
    bleu=np.reshape(viirs['bleu'], (288,3200))
    
    
#     rgbArray= np.zeros((30,5,3), 'uint8') #valeurs pour test_test_granule_Isabel.ppm
#     rouge=np.reshape(viirs['rouge'], (30,5))
#     vert=np.reshape(viirs['vert'], (30,5))
#     bleu=np.reshape(viirs['bleu'], (30,5))
    
    
    rgbArray[..., 0] = rouge
    rgbArray[..., 1] = vert
    rgbArray[..., 2] = bleu
    
    
    img = Image.fromarray(rgbArray)
    
    #img.show()
    img.save('/home/mcms/almeidamanceroi/workspace/workspace/VIIRS/'+name)
    
    return 1

'----------------------------------------------------------------------------------------------------------------------------------------------'

def model_granule(X):
    X['SP14_13']=100*(X.Obs_87-X.Obs_40)/(X.Obs_87+X.Obs_40)
    X['SP13_12']=100*(X.Obs_40-X.Obs_37)/(X.Obs_40+X.Obs_37)
    X['SP15_16']=100*(X.Obs_108-X.Obs_120)/(X.Obs_108+X.Obs_120)
    X['SP15_14']=100*(X.Obs_108-X.Obs_87)/(X.Obs_108+X.Obs_87)
    X['SP15_12']=100*(X.Obs_108-X.Obs_37)/(X.Obs_108+X.Obs_37)
    X['B14_15']=X.Obs_87/X.Obs_108 #NOAA beta
    X['B16_15']=X.Obs_120/X.Obs_108 #NOAA beta
    X['B7_5']=X.Obs_08/X.Obs_06
    X['SP16_12']=(X.Obs_120-X.Obs_37)/(X.Obs_120+X.Obs_37)
    X['SP4_10']=(X.Obs_05-X.Obs_16)/(X.Obs_05+X.Obs_16)
    X['SP14_15']=X.Obs_87-X.Obs_108
    X['test6']=np.cos(X['pix_sunzen']*math.pi/180)
    X['B15_12']=X.Obs_108/X.Obs_37
    X['test3']=np.maximum(X.Obs_120-X.Obs_108 ,np.maximum( X.Obs_120-X.Obs_37,X.Obs_108-X.Obs_37))
    X['test4']=(X.Obs_108-X.Obs_37)*(X.Obs_108-X.Obs_37)
    X['SP5_10']=(X.Obs_06-X.Obs_16)/(X.Obs_06+X.Obs_16) #classif neige
    X['SP5_11']=(X.Obs_06-X.Obs_22)/(X.Obs_06+X.Obs_22) #classif neige mieux
    X['SP10_11']=(X.Obs_16-X.Obs_22)/(X.Obs_16+X.Obs_22)
    X['BW']=(X.Obs_04+X.Obs_05+X.Obs_06)/(3*255)
    X['test2b']=X.Obs_108*X.Obs_108
    X['test3b']=X.Obs_22*X.Obs_22
    X['test4b']=X.Obs_87*X.Obs_87
    X['D_Obs_87_120']=X.Obs_87-X.Obs_120
    X['D_Obs_108_120']=X.Obs_108-X.Obs_120
    X['D_Obs87_108']=X.Obs_87-X.Obs_108
    X['D_Obs_13_37']=X.Obs_13-X.Obs_37
    X['D_Obs_06_16']=X.Obs_06-X.Obs_16
    
    dataset=['Obs_06', 'Obs_05', 'Obs_08', 'Obs_04', 'Obs_13', 'Obs_22', 'Obs_16', 'Obs_40', 'Obs_37',
              'Obs_108', 'Obs_87', 'Obs_120', 'pix_satzen', 'pix_sunzen', 'pix_satazi', 'pix_sunazi',
               'pix_alt', 'pix_tm', 'pix_daytime', 'SP14_13', 'SP13_12', 'SP15_16', 'SP15_14',
                'SP15_12', 'B14_15', 'B16_15', 'B7_5', 'SP16_12', 'SP4_10', 'SP14_15', 'test6', 'B15_12',
                 'test3', 'test4', 'SP5_10', 'SP5_11', 'SP10_11', 'BW', 'test2b', 'test3b', 'test4b',
                  'D_Obs_87_120', 'D_Obs_108_120', 'D_Obs87_108', 'D_Obs_13_37', 'D_Obs_06_16','var_05',
                  'var_06','var_08','var_13','var_16','var_22','var_37','var_40','var_87','var_108','var_120'] 
     
    data=X[dataset]
    return data

'Paramètres du granule:'
"Obs_06,Obs_05,Obs_08,Obs_04,Obs_13,Obs_22,Obs_16,Obs_40,Obs_37,Obs_108,Obs_87,Obs_120,"
"pix_lon,pix_lat,pix_satzen,pix_sunzen,pix_satazi,pix_sunazi,pix_alt,pix_tm,pix_daytime,sst"

'----------------------------------------------------------------------------------------------------------------------------------------------'

if __name__ == '__main__':
    start_time = time.time()
    pd.options.mode.chained_assignment = None  # default='warn'
    
    
    
    viirs=pd.read_table('/home/mcms/almeidamanceroi/workspace/workspace/VIIRS/test_read_viirs4_vars_demi.txt',sep=',')
    vii=viirs[:][:921600] # /!\ changer la taille aux deuxièmes crochets selon la taille du granule ainsi que SaveCode.
    viirs_data= model_granule(vii)
#     lon=viirs_data['pix_lon']
#     lat=viirs_data['pix_lat']
#     Xdata=viirs_data.drop(['pix_lon','pix_lat'],1)
    peigne=liste_peigne(vii) #1:pixel du pigne, 2:autre pixel
      
    # load the classifier
    with open('/home/mcms/almeidamanceroi/workspace/workspace/VIIRS/sauvegarde_classifiers/my_dumped_classifier_pour_granule_vars.pkl', 'rb') as fid:
        classifieur_loaded = cPickle.load(fid)
    #print classifieur_loaded
    print ("--- temps avant prediction: %s seconds---" % round(time.time()-start_time,2))
      
    Codes_pred=classifieur_loaded.predict(viirs_data)
    print 'prediction des codes realisee'
    codes_avc_peigne=[20 if peigne[x] else Codes_pred[x] for x in xrange(len(Codes_pred))]
      
    SaveCode(codes_avc_peigne,name='test_granule_Isabel4_vars2.ppm') #ATTENTION: changer a chaque fois

#     test=[i for i in [1,2,3,4,6,8,10,12,14,15,16,17,18,19,20] for k in range(10)]
#     SaveCode(test,name='test_test_granule_Isabel.ppm')
    
    tps_final=round(time.time()-start_time,2)
    print("--- %s seconds---" % tps_final)
    print '--------------------end--------------------'
    
    
    