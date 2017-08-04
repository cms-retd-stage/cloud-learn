# -*- coding: utf-8 -*-
'''
Created on 13 juin 2016

@author: delatailler; almeidamanceroi
'''

import cPickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time

from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.classification import confusion_matrix
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.lda import LDA

from pandas.tools.plotting import radviz

from skimage.filters.rank.generic import equalize

from pointe_cibles import drawMap
from pointe_cibles import drawhist
from pointe_cibles import printimportance
from pointe_cibles import printimportance2
from pointe_cibles import draw3D

from gestion_fichier import indexprint

"""
---------------------------------------------------------------------------------------------
"""

def modif_code_nuage(viirs):
    viirs_nuage=viirs
    clair=viirs['code']<5
    fract=viirs['code']==19
    nuage=np.logical_and(viirs['code']>5,viirs['code']!=19)
    viirs_nuage['code'][clair]=0
    viirs_nuage['code'][fract]=19
    viirs_nuage['code'][nuage]=2
  
    return viirs_nuage


def compute_score(estimateur,X,Y):
    xval=cross_val_score(estimateur,X,Y,cv=10,n_jobs=-1)
    score=np.mean(xval)
    print 'score cross-validation:', score
    return score


def equalization(X):
    X_equ=X
    cdf=X_equ.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0)
    
    return cdf


def creation_TrainTest(viirs_data,viirs_tm,ratio):
    """
    !!! TM non modifie par randomm !!!
    ne pas utiliser pour Drawmap(1)
    """
    print 'Creation Train/Test'
    n=int(ratio*len(viirs_data))
    
    viirs_data=viirs_data.iloc[np.random.permutation(len(viirs_data))] #mélange les lignes du dataframe
    viirs_data=viirs_data.reset_index(drop=True) #réinitialise les indices sans créer une nouvelle colonne
    
    lon=viirs_data['pix_lon']
    lat=viirs_data['pix_lat']
    viirs_data=viirs_data.drop(['pix_lon','pix_lat'],1)
    
    viirs_target=viirs_data['code'] #on garde le code qui est déjà de 1 a 20
    viirs_data=viirs_data.drop('code',1)
    
    X_train=viirs_data.iloc[:n]
    Y_train=viirs_target.iloc[:n]
    
    X_test=viirs_data.iloc[n:]
    Y_test=viirs_target.iloc[n:]
    
    lon=lon.iloc[n:] #on garde les lat et lon que pour les donnees test
    lat=lat.iloc[n:]
    tm=viirs_tm.iloc[n:] #idem pour tm
    print 'Train/Test cree'
    return X_train,X_test,Y_train,Y_test,lon,lat,tm


def confusion_Mat_Tab(y_true,y_pred):
    
    deb=y_true.first_valid_index()
    fin=len(y_true)
    conf=np.zeros(fin)
    
    for i in xrange(fin):
        if (y_true[i+deb]==True) and (y_pred[i]==True): 
            # vrai nuage
            conf[i]=1
        elif (y_true[i+deb]==True) and (y_pred[i]==False): 
            # faux autre
            conf[i]=2
        elif (y_true[i+deb]==False) and (y_pred[i]==True): 
            # faux nuage
            conf[i]=3
        elif (y_true[i+deb]==False) and (y_pred[i]==False): 
            # vrai autre
            conf[i]=4
    
    
    return conf


def post_model(viirs,TT):
    '''
    TT : Train=0, Test=1
    '''
    if TT==0:
        Y_train=viirs['code']
        X_train=viirs.drop(['pix_lon','pix_lat','code'],1)
    
        return X_train,Y_train
    elif TT==1:
        X_test=viirs.drop(['pix_lon','pix_lat','code'],1)
        return X_test
    
    
"""
---------------------------------------------------------------------------------------------
"""


def model_0(X):
    target=X.code
    X=X[['pix_tm','Obs_08','Obs_16','Obs_04','Obs_05','Obs_06']]
    print 'modele cree'
    return X,target


def model_1_nuit(X):
    target=X.code
    X=X[['pix_tm','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120']]
    print 'modele cree'
    return X,target


def model_1_jour(X):
    tm=X['pix_tm']
    
    X['SP14_13']=100*(X.Obs_87-X.Obs_40)/(X.Obs_87+X.Obs_40)
    X['SP4_7']=(X.Obs_04-X.Obs_08)/(X.Obs_04+X.Obs_08)
    X['SP10_11']=(X.Obs_16-X.Obs_22)
    X['SP13_12']=X.Obs_40-X.Obs_37
    X['SP14_12']=(X.Obs_87-X.Obs_37)/(X.Obs_87+X.Obs_37)
    
    X['SP15_12']=(X.Obs_108-X.Obs_37)
    X['SP16_12']=(X.Obs_120-X.Obs_37)/(X.Obs_120+X.Obs_37)
    X['SP4_10']=(X.Obs_05-X.Obs_16)/(X.Obs_05+X.Obs_16)
    
    X['SP14_15']=X.Obs_87-X.Obs_108
    
    X['B14_15']=X.Obs_87/X.Obs_108 #NOAA beta
    X['B16_15']=X.Obs_120/X.Obs_108 #NOAA beta
    X['B13_15']=X.Obs_40/X.Obs_108 #NOAA beta
    X['B7_5']=X.Obs_08/X.Obs_06
    
    
    dataset=['sst_clim','B14_15','B16_15','SP14_13','pix_alt','SP4_7','pix_tm','SP10_11','Obs_108','Obs_120','Obs_87','Obs_13']
    
    allObs=['Obs_04','Obs_05','Obs_06','Obs_08','Obs_13','Obs_16','Obs_22','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120'] 
    for i in allObs:
        for j in allObs:
            X[i+j+'DAKKA']=X[i]-X[j]
            dataset=np.append(dataset,i+j+'DAKKA')
    
    data=X[dataset]
    data[['pix_lat','pix_lon','code']]=X[['pix_lat','pix_lon','code']]
    return data,tm,dataset


def model_sepa_jour(X):
    tm=X['pix_tm']
    X['SP14_13']=100*(X.Obs_87-X.Obs_40)/(X.Obs_87+X.Obs_40)
    X['SP12_13']=(X.Obs_37-X.Obs_40)/(X.Obs_37+X.Obs_40)
    X['SP4_7']=(X.Obs_04-X.Obs_08)/(X.Obs_04+X.Obs_08)
    X['SP16_13']=(X.Obs_120-X.Obs_40)
    X['SP10_11']=100*(X.Obs_16-X.Obs_22)/(X.Obs_16+X.Obs_22)
    X['pix_tm']=X.pix_tm
    X['BW']=(X.Obs_04+X.Obs_05+X.Obs_06)/(3*255)
    X['test1']=abs(X.SP4_7)
    X['test2']=X.Obs_108*X.Obs_108
    X['test3']=X.Obs_22*X.Obs_22
    X['test4']=X.Obs_87*X.Obs_87
    
    X['SP10_11']=(X.Obs_16-X.Obs_22)/(X.Obs_16+X.Obs_22)
    
    X['SP5_10']=(X.Obs_06-X.Obs_16)/(X.Obs_06+X.Obs_16) #classif neige
    X['SP5_11']=(X.Obs_06-X.Obs_22)/(X.Obs_06+X.Obs_22) #classif neige mieux
    
    X['B14_15']=X.Obs_87/X.Obs_108 #NOAA beta
    X['B16_15']=X.Obs_120/X.Obs_108 #NOAA beta
    X['B13_15']=X.Obs_40/X.Obs_108 #NOAA beta
    X['B15_12']=X.Obs_108/X.Obs_37
    
    X['test']=X['SP10_11']<0
    X['test2']=X['test']*X['SP5_11']
    
    '''
    mer=(X['pix_tm']==0)
    X['Emis_108'][mer]=0.98 #creation emissivite mer
    
    X['temp_atlas']=X['ts_prev']
    X['temp_atlas'][mer]=X['sst_clim'][mer]
    X['test5']=X.Obs_108/((X.Emis_108+1)*X.temp_atlas) #Rapport TempBrillance/TempSurface
    '''
    
    X['ref']=X.Obs_22-X.Obs_16
    pos=X['ref']>0
    X['refcu']=X['ref']*pos
    
    X['rouge']=X['Obs_08']-0.5*X['refcu']-0.25*X['Obs_13']
    X['bleu']=X['Obs_22']+X['Obs_13']
    
    
    #print X[['Emis_108','code','ts_prev','Obs_108','test5','SP14_13','pix_tm']]
    #X.plot(kind='scatter',x='test5',y='SP14_13',c='code')
    #radviz(X[['test5','SP14_13','SP4_7','code']], 'code')
    #draw3D(viirs=X, xs='SP14_13', ys='test5', zs='cwv_prev', code='code')
    plt.show()
    #'pix_satzen','pix_sunzen','pix_satazi','pix_sunazi',
    #'SP14_13','pix_alt','SP4_7','pix_tm','SP10_11'
    
    
    
    dataset=['rouge','bleu','test2','SP5_11','B15_12','B13_15','B16_15','B14_15','SP14_13','SP4_7','pix_alt','pix_tm','SP10_11']
    
    allObs=['Obs_04','Obs_05','Obs_06','Obs_08','Obs_13','Obs_16','Obs_22','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120'] 
    for i in xrange(len(allObs)):
        for j in xrange(len(allObs)):
            if j>i:
                X[allObs[i]+allObs[j]+'DAKKA']=X[allObs[i]]-X[allObs[j]]
                dataset=np.append(dataset,allObs[i]+allObs[j]+'DAKKA')
    
    
    #'SP14_13','test5','SP4_7','pix_tm','pix_alt','cwv_prev','SP10_11' best
    # 'pix_tm','Obs_120','Obs_108','Obs_37' test
    
    data=X[dataset]


    
    data[['pix_lat','pix_lon','code']]=X[['pix_lat','pix_lon','code']]
    
    return data,tm, dataset


def model_sepa_nuit(X):
    tm=X['pix_tm']
    X['SP14_13']=100*(X.Obs_87-X.Obs_40)/(X.Obs_87+X.Obs_40)
    X['SP14_12']=100*(X.Obs_87-X.Obs_37)/(X.Obs_87+X.Obs_37)
    X['SP13_12']=100*(X.Obs_40-X.Obs_37)/(X.Obs_40+X.Obs_37)
    
    X['SP15_16']=100*(X.Obs_108-X.Obs_120)/(X.Obs_108+X.Obs_120)
    X['SP15_14']=100*(X.Obs_108-X.Obs_87)/(X.Obs_108+X.Obs_87)
    
    X['SP15_12']=100*(X.Obs_108-X.Obs_37)/(X.Obs_108+X.Obs_37)
    '''
    mer=(X['pix_tm']==0)
    X['Emis_108'][mer]=0.98 #creation emissivite mer
    X['temp_atlas']=X['ts_prev']
    X['temp_atlas'][mer]=X['sst_clim'][mer]
    X['test5']=X.Obs_108/((X.Emis_108+1)*X.temp_atlas) #Rapport TempBrillance/TempSurface
    
    X['test1']=(X.Emis_108-X.ts_prev)/(X.Emis_108+X.ts_prev) #difference pondere TS,T108
    
    X['test2']=(X.Obs_108-X.ts_prev)  #difference TS,T108
    '''
    X['B14_15']=X.Obs_87/X.Obs_108 #NOAA beta
    X['B16_15']=X.Obs_120/X.Obs_108 #NOAA beta
    X['B13_15']=X.Obs_40/X.Obs_108 #NOAA beta
    X['B15_12']=X.Obs_108/X.Obs_37
    
    
    
    X['test3']=np.maximum(X.Obs_120-X.Obs_108 ,np.maximum( X.Obs_120-X.Obs_37,X.Obs_108-X.Obs_37))
    X['test4']=(X.Obs_108-X.Obs_37)*(X.Obs_108-X.Obs_37)
    #X.plot(kind='scatter',x='SP15_16',y='SP15_14',c='code')
    #radviz(X[['test5','SP15_16','SP15_12','code']], 'code')
    #draw3D(viirs=X, xs='test4', ys='Obs_108', zs='pix_tm', code='code')
    plt.show()
    
    dataset=['test4','B15_12','B16_15','SP15_12','SP14_12','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120']
    
    
    allObs=['Obs_37','Obs_40','Obs_87','Obs_108','Obs_120']
    
    for i in xrange(len(allObs)):
        for j in xrange(len(allObs)):
            if j>i:
                X[allObs[i]+allObs[j]+'DAKKA']=X[allObs[i]]-X[allObs[j]]
                dataset=np.append(dataset,allObs[i]+allObs[j]+'DAKKA')
    
    
    
    
    data=X[dataset]
    #'SP14_12','SP15_12','SP15_14','test5','SP13_12' best
    #'T700','test5','T850','cwv_prev','ts_prev','ps_prev','B13_15','B16_15','SP15_12','SP14_12','Obs_37','Obs_87','Obs_108','Obs_120'
    #'test4','B15_12','test5','cwv_prev','ts_prev','B16_15','SP15_12','SP14_12','Obs_37','Obs_87','Obs_108','Obs_120'
    data[['pix_lat','pix_lon','code']]=X[['pix_lat','pix_lon','code']]
    return data,tm,dataset


def model_tri_nuage_simple_jour(X):
    tm=X['pix_tm']
    X['SP14_13']=100*(X.Obs_87-X.Obs_40)/(X.Obs_87+X.Obs_40)
    X['SP4_7']=(X.Obs_04-X.Obs_08)/(X.Obs_04+X.Obs_08)
    X['SP10_11']=(X.Obs_16-X.Obs_22)
    X['SP13_12']=X.Obs_40-X.Obs_37
    X['SP14_12']=(X.Obs_87-X.Obs_37)/(X.Obs_87+X.Obs_37)
    
    X['SP15_12']=(X.Obs_108-X.Obs_37)
    X['SP16_12']=(X.Obs_120-X.Obs_37)/(X.Obs_120+X.Obs_37)
    X['SP4_10']=(X.Obs_05-X.Obs_16)/(X.Obs_05+X.Obs_16)
    
    X['SP14_15']=X.Obs_87-X.Obs_108
    """
    mer=(X['pix_tm']==0)
    X['Emis_108'][mer]=0.98 #creation emissivite mer
    X['test5']=X.Obs_108/((X.Emis_108+1)*X.ts_prev)
    """
    X['B14_15']=X.Obs_87/X.Obs_108 #NOAA beta
    X['B16_15']=X.Obs_120/X.Obs_108 #NOAA beta
    X['B13_15']=X.Obs_40/X.Obs_108 #NOAA beta
    X['B7_5']=X.Obs_08/X.Obs_06
    
    X['TRI14_15_16']=(X.Obs_87-X.Obs_108)*(X.Obs_108-X.Obs_120)
    
    X['test6']=np.cos(X['pix_sunzen']*math.pi/180)
    X['test7']=np.cos(X['pix_sunazi']*math.pi/180)
    X['test8']=np.cos(X['pix_satzen']*math.pi/180)
    X['test9']=np.cos(X['pix_satazi']*math.pi/180)
    
    #X.plot(kind='scatter',x='SP14_15',y='B14_15',c='code')
    #radviz(X[['B14_15','SP10_11','B16_15','Obs_108','code']], 'code')
    #draw3D(viirs=X, xs='pix_tm', ys='SP10_11', zs='B14_15', code='code')
    plt.show()
    
    database=['B14_15','B16_15','SP10_11','Obs_108','Obs_120','Obs_87','Obs_13']
    dataset=database
    """
    for i in database:
        X[i+'sunzen']=X[i]*X['test6']
        dataset=np.append(dataset,i+'sunzen')
    for i in database: 
        X[i+'sunazi']=X[i]*X['test7']
        dataset=np.append(dataset,i+'sunazi')
    for i in database: 
        X[i+'satzen']=X[i]*X['test8']
        dataset=np.append(dataset,i+'satzen')
    for i in database: 
        X[i+'satazi']=X[i]*X['test9']
        dataset=np.append(dataset,i+'satazi')
    
    dataset=np.append(dataset,['test6','test7','test8','test9','pix_sunzen','pix_sunazi','pix_satzen','pix_satazi'])    
    """
    
    allObs=['Obs_04','Obs_05','Obs_06','Obs_08','Obs_13','Obs_16','Obs_22','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120'] 
    for i in xrange(len(allObs)):
        for j in xrange(len(allObs)):
            if j>i:
                X[allObs[i]+allObs[j]+'DAKKA']=X[allObs[i]]-X[allObs[j]]
                dataset=np.append(dataset,allObs[i]+allObs[j]+'DAKKA')
    
    #'B14_15','B16_15','SP10_11','Obs_108','Obs_120','Obs_87','Obs_13' best
    data=X[dataset]
    data[['pix_lat','pix_lon','code']]=X[['pix_lat','pix_lon','code']]
    return data,tm,dataset


def model_tri_clair_simple_jour(X):
    tm=X['pix_tm']
    X['SP5_10']=(X.Obs_06-X.Obs_16)/(X.Obs_06+X.Obs_16) #classif neige
    X['SP4_7']=(X.Obs_04-X.Obs_08)/(X.Obs_04+X.Obs_08)
    X['SP5_11']=(X.Obs_06-X.Obs_22)/(X.Obs_06+X.Obs_22) #classif neige
    X['SP10_11']=(X.Obs_16-X.Obs_22)/(X.Obs_16+X.Obs_22)
    #X['SP14_15']=X.Obs_87-X.Obs_108
    X['test']=X['SP10_11']<0
    X['test2']=X['test']*X['SP5_11']
    
    #X.plot(kind='scatter',x='Obs_108',y='SP4_7',c='code')
    #radviz(X[['pix_tm','SP4_7','Obs_108','code']], 'code')
    #draw3D(viirs=X, xs='SP4_7', ys='Obs_108', zs='pix_tm', code='code')
    plt.show()
    
    dataset=['SP5_11','SP4_7','pix_tm','Obs_108']
    #'SP4_7','pix_tm','Obs_108 met ombre et sol froid en neige
    #'SP5_11','SP4_7','pix_tm','Obs_108' met ombre en neige
    '''
    #met les ombres en mer
    allObs=['Obs_04','Obs_05','Obs_06','Obs_08','Obs_13','Obs_16','Obs_22','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120'] 
    for i in xrange(len(allObs)):
        for j in xrange(len(allObs)):
            if j>i:
                X[allObs[i]+allObs[j]+'DAKKA']=X[allObs[i]]-X[allObs[j]]
                dataset=np.append(dataset,allObs[i]+allObs[j]+'DAKKA')
   '''
    data=X[dataset]
    data[['pix_lat','pix_lon','code']]=X[['pix_lat','pix_lon','code']]
    return data,tm,dataset


def model_tri_nuage_simple_nuit(X):
    tm=X['pix_tm']
    X['SP14_13']=100*(X.Obs_87-X.Obs_40)/(X.Obs_87+X.Obs_40)
    X['SP14_12']=100*(X.Obs_87-X.Obs_37)/(X.Obs_87+X.Obs_37)
    X['SP13_12']=100*(X.Obs_40-X.Obs_37)/(X.Obs_40+X.Obs_37)
    
    X['SP15_16']=100*(X.Obs_108-X.Obs_120)/(X.Obs_108+X.Obs_120)
    X['SP15_14']=100*(X.Obs_108-X.Obs_87)/(X.Obs_108+X.Obs_87)
    
    X['SP15_12']=100*(X.Obs_108-X.Obs_37)/(X.Obs_108+X.Obs_37)
    
    X['B14_15']=X.Obs_87/X.Obs_108 #NOAA beta
    X['B16_15']=X.Obs_120/X.Obs_108 #NOAA beta
    X['B13_15']=X.Obs_40/X.Obs_108 #NOAA beta
    
    #X.plot(kind='scatter',x='Obs_108',y='SP4_7',c='code')
    #radviz(X[['pix_tm','SP4_7','Obs_108','code']], 'code')
    #draw3D(viirs=X, xs='Obs_120', ys='Obs_108', zs='Obs_87', code='code')
    plt.show()
    
    
    dataset=['B13_15','B16_15','SP15_16','SP15_12','SP14_13','SP14_12','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120']
    data=X[dataset]
    data[['pix_lat','pix_lon','code']]=X[['pix_lat','pix_lon','code']]
    return data,tm,dataset


def model_tri_clair_simple_nuit(X):
    tm=X['pix_tm']
    X['SP14_13']=100*(X.Obs_87-X.Obs_40)/(X.Obs_87+X.Obs_40)
    X['SP14_12']=100*(X.Obs_87-X.Obs_37)/(X.Obs_87+X.Obs_37)
    X['SP13_12']=100*(X.Obs_40-X.Obs_37)/(X.Obs_40+X.Obs_37)
    
    X['SP15_16']=100*(X.Obs_108-X.Obs_120)/(X.Obs_108+X.Obs_120)
    X['SP15_14']=100*(X.Obs_108-X.Obs_87)/(X.Obs_108+X.Obs_87)
    
    X['SP15_12']=100*(X.Obs_108-X.Obs_37)/(X.Obs_108+X.Obs_37)
    
    #X.plot(kind='scatter',x='Obs_108',y='SP4_7',c='code')
    #radviz(X[['pix_tm','SP4_7','Obs_108','code']], 'code')
    #draw3D(viirs=X, xs='pix_tm', ys='sst_clim', zs='Obs_87', code='code')
    plt.show()
    
    dataset=['pix_tm','Obs_108','Obs_37','SP14_12','SP13_12','SP15_14','SP15_12','Obs_40','Obs_87','Obs_120']
    data=X[dataset]
    data[['pix_lat','pix_lon','code']]=X[['pix_lat','pix_lon','code']]
    return data,tm,dataset

def model_nouvelles_cibles_clair_simple_nuit(X):
    tm=X['pix_tm']
    X['SP14_13']=100*(X.Obs_87-X.Obs_40)/(X.Obs_87+X.Obs_40)
    X['SP14_12']=100*(X.Obs_87-X.Obs_37)/(X.Obs_87+X.Obs_37)
    X['SP13_12']=100*(X.Obs_40-X.Obs_37)/(X.Obs_40+X.Obs_37)
    X['SP15_16']=100*(X.Obs_108-X.Obs_120)/(X.Obs_108+X.Obs_120)
    X['SP15_14']=100*(X.Obs_108-X.Obs_87)/(X.Obs_108+X.Obs_87)
    X['SP15_12']=100*(X.Obs_108-X.Obs_37)/(X.Obs_108+X.Obs_37)
    dataset=['pix_tm','SP14_12','SP13_12','SP15_14','SP15_12','SP15_16','SP14_13','alt_NWP','ps_prev','ts_prev','T500','albedo_clim','sst', 'var_37', 'var_40', 'var_87', 'var_108', 'var_120','maxsst','minsst','ansst','glace']
    #dataset=['pix_tm','Obs_108','Obs_37','SP14_12','SP13_12','SP15_14','SP15_12','Obs_40','Obs_87','Obs_120']
    #Obs_04,Obs_05,Obs_06,Obs_08,Obs_13,Obs_16,Obs_22,Obs_37,Obs_40,Obs_87,Obs_108,Obs_120,code,texte,alt_NWP,ps_prev,ts_prev,t2m_prev,cwv_prev,T500,T700,T850,sst_clim,sstmoy_clim,albedo_clim,sst,var_04,var_05,var_06,var_08,var_13,var_16,var_22,var_37,var_40,var_87,var_108,var_120
    
    data=X[dataset]
    data[['pix_lat','pix_lon','code']]=X[['pix_lat','pix_lon','code']]
    return data,tm,dataset



def model_simple_cible_nouveau(X):
    tm=X['pix_tm']
    X['SP14_13']=100*(X.Obs_87-X.Obs_40)/(X.Obs_87+X.Obs_40)
    #X['SP14_12']=100*(X.Obs_87-X.Obs_37)/(X.Obs_87+X.Obs_37)
    X['SP13_12']=100*(X.Obs_40-X.Obs_37)/(X.Obs_40+X.Obs_37)
    X['SP15_16']=100*(X.Obs_108-X.Obs_120)/(X.Obs_108+X.Obs_120)
    X['SP15_14']=100*(X.Obs_108-X.Obs_87)/(X.Obs_108+X.Obs_87)
    X['SP15_12']=100*(X.Obs_108-X.Obs_37)/(X.Obs_108+X.Obs_37)
    
    X['B14_15']=X.Obs_87/X.Obs_108 #NOAA beta
    X['B16_15']=X.Obs_120/X.Obs_108 #NOAA beta
    #X['B13_15']=X.Obs_40/X.Obs_108 #NOAA beta
    X['B7_5']=X.Obs_08/X.Obs_06
    #X['SP4_7']=(X.Obs_04-X.Obs_08)/(X.Obs_04+X.Obs_08)
    X['SP16_12']=(X.Obs_120-X.Obs_37)/(X.Obs_120+X.Obs_37)
    X['SP4_10']=(X.Obs_05-X.Obs_16)/(X.Obs_05+X.Obs_16)
    
    X['SP14_15']=X.Obs_87-X.Obs_108
    #X['TRI14_15_16']=(X.Obs_87-X.Obs_108)*(X.Obs_108-X.Obs_120)
    X['test6']=np.cos(X['pix_sunzen']*math.pi/180)
    X['B15_12']=X.Obs_108/X.Obs_37
    X['test3']=np.maximum(X.Obs_120-X.Obs_108 ,np.maximum( X.Obs_120-X.Obs_37,X.Obs_108-X.Obs_37))
    X['test4']=(X.Obs_108-X.Obs_37)*(X.Obs_108-X.Obs_37)
    #X['bleu']=X['Obs_22']+X['Obs_13']
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
    
    dataset=['SP13_12','BW','SP14_13','D_Obs_87_120','D_Obs_108_120','D_Obs87_108','D_Obs_13_37','D_Obs_06_16','SP5_11','SP5_10','test4b','test3b','test2b','test4','test3','B15_12','test6','SP14_15','SP4_10','SP16_12','SP10_11','B7_5','B16_15','B14_15','SP15_12','SP15_14','SP15_16','pix_tm','pix_sunzen','Obs_05','Obs_06','Obs_13','Obs_16','Obs_22','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120','alt_NWP','ts_prev','cwv_prev','sst','var_04','var_05','var_06','var_08','var_16','var_22','var_37','var_87','var_108','var_120','maxsst','minsst','ansst','glace']
    #dataset=['pix_sunzen','pix_alt','pix_tm','Obs_04','Obs_05','Obs_06','Obs_08','Obs_13','Obs_16','Obs_22','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120','alt_NWP','ts_prev','t2m_prev','cwv_prev','T700','T850','sst_clim','sstmoy_clim','sst','var_04','var_05','var_06','var_08','var_13','var_16','var_22','var_37','var_87','var_108','var_120']
    
    data=X[dataset]
    data[['pix_lat','pix_lon','code']]=X[['pix_lat','pix_lon','code']]
    return data,tm,dataset


"""
-----------------------------------------------------------------------------------------------------------------------------------------
"""

if __name__ == '__main__':
    start_time = time.time()
    #Recuperation des donnees
    
    #viirs=pd.read_table('Nouvelles_cibles_a_utiliser/simple_cible_clear_night_pascale_full.txt',sep=',')
    viirs=pd.read_table('Nouvelles_cibles_a_utiliser/simple_cible_nouveau.txt',sep=',')
    
    print list(viirs.columns.values)
    print '\n'
    #print viirs
    
    #viirs_data,tm,dataset=model_nouvelles_cibles_clair_simple_nuit(viirs)
    viirs_data,tm,dataset= model_simple_cible_nouveau(viirs)
    print dataset
    
    tps_estimateur_av=time.time()
    X_train,X_test,Y_train,Y_test,X_lon,X_lat,X_tm=creation_TrainTest(viirs_data,tm,0.8)
      
    print 'creation de l estimateur'
    estimateur=RandomForestClassifier(n_estimators=2500,min_samples_leaf=5,n_jobs=-1)
    
     
    print 'calcul fit'
    estimateur.fit(X_train, Y_train)
    
    tps_estimateur=time.time()
    tps_est=tps_estimateur-tps_estimateur_av
    print("---Durée de créatioin et entrainement de l'estimateur: %s---" % tps_est)
    
    y_true, y_pred = Y_test, estimateur.predict(X_test)
    
    
    printimportance2(dataset, estimateur) 
     
    print 'codes des cibles et nombre d apparitions dans le test\n'
    print Y_test.value_counts().sort_index()
    
    
    print '\nmatrice de confusion: \n'
    print confusion_matrix(y_true, y_pred)
    
    print '\n'
    compute_score(estimateur, X_train, Y_train)
    print '\n'
    print estimateur
     
    print '\nRapport \n'
    print classification_report(y_true, y_pred)
      
    
    tps_final=round(time.time()-start_time,2)
    print("--- %s seconds en total---" % tps_final)
     
    
#     # save the classifier
#     with open('my_dumped_classifier4.pkl', 'wb') as fid:
#         cPickle.dump(estimateur, fid)    

#     # load it again
#     with open('my_dumped_classifier.pkl', 'rb') as fid:
#     gnb_loaded = cPickle.load(fid) 
     
     
    print '--------------------end--------------------'
    
    
    """
    #drawmap2
    code=confusion_Mat_Tab(y_true, y_pred)
    drawMap(X_lat,X_lon,X_tm,code,2)
    """
    
    
    """
    #MinMax
    mms=MinMaxScaler()
    X_train=mms.fit_transform(X_train)
    X_test=mms.transform(X_test)
    """
    
    
    """
    #test arbres 
    print viirs.count()
    
    viirs_data,viirs_target= model_1_jour(viirs)
    X_train,X_test,Y_train,Y_test=creation_TrainTest(viirs_data, viirs_target, 0.8)
       
    estimateur=RandomForestClassifier(n_estimators=500,min_samples_leaf=20,n_jobs=-1)
    compute_score(estimateur, X_train, Y_train)
    
    score_esti=estimateur.fit(X_train, Y_train).score(X_test,Y_test)
    y_true=Y_test
    y_pred=estimateur.predict(X_test)
    print '\nRapport \n'
    print classification_report(y_true, y_pred)
    print 'score estimateur',score_esti
    """
    
    """
    'Obs_04','Obs_05','Obs_06','Obs_08','Obs_13','Obs_16','Obs_22','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120'
    """
    
    """
    #test SVC basiques
    print viirs.count(),'\n'
    
   
    estimateur=svm.SVC(kernel='rbf',C=1,gamma=0.001)
    compute_score(estimateur, X_train, Y_train)
    
    
    score_esti=estimateur.fit(X_train, Y_train).score(X_test,Y_test)
    y_true=Y_test
    y_pred=estimateur.predict(X_test)
    print '\nRapport \n'
    print classification_report(y_true, y_pred)
    print 'score estimateur',score_esti
    
    print 'wololo'
    """
    
    
    
    """
    #test Gridsearch
    print np.logspace(-3, 1, 5)
    
    print 'creation des parametres'
    tuned_parameter={'kernel':['rbf'],'gamma':gamma,'C':C}
    
    print 'creation de l estimateur'
    estimateur=GridSearchCV(SVC(C=1), tuned_parameter,scoring='precision')
    
    
    print 'calcul fit'
    estimateur.fit(X_train, Y_train)
    y_true, y_pred = Y_test, estimateur.predict(X_test)
    
    print '\nmeilleur estimateur: \n'
    print confusion_matrix(y_true, y_pred)
    print estimateur.best_estimator_
    
    print '\nRapport \n'
    print classification_report(y_true, y_pred)
    
    
    """
    
    
       
    
    
    
    
    
    
    