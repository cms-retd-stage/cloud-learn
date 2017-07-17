'''
Created on 20 juin 2016

@author: delatailler
'''

import numpy as np
import pandas as pd
import pylab as plt
from pointe_cibles import drawhist

def indexprint():
    print 'yyyy\nmm \ndd \nhh\nmn\nss\nip\nil\npix_lon\npix_lat\npix_satzen\npix_sunzen\npix_satazi\npix_sunazi\npix_alt\npix_tm\npix_daytime\nObs_04\nObs_05\nObs_06\nObs_08\nObs_13\nObs_16\nObs_22\nObs_37\nObs_40\nObs_87\nObs_108\nObs_120\ncode\ntexte\nalt_NWP\nps_prev\nts_prev\nt2m_prev\ncwv_prev\nT500\nT700\nT850\nsst_clim\nsstmoy_clim\nalbedo_clim\nEmis_04\nEmis_05\nEmis_06\nEmis_08\nEmis_13\nEmis_16\nEmis_22\nEmis_37\nEmis_40\nEmis_87\nEmis_108\nEmis_120\nRefl_04\nRefl_05\nRefl_06\nRefl_08\nRefl_13\nRefl_16\nRefl_22\nRefl_37\nRefl_40\nRefl_87\nRefl_108\nRefl_120'
    print'\n'
    return 1


def separation_jour_nuit(viirs):
    viirs_jour=viirs
    viirs_nuit=viirs
    viirs_truc=viirs
    viirs_aube=viirs
    for i in xrange(len(viirs['pix_daytime'])):
        if (viirs_nuit['pix_daytime'][i]!=0): viirs_nuit=viirs_nuit.drop([i])
        if (viirs_aube['pix_daytime'][i]!=1): viirs_aube=viirs_aube.drop([i])
        if (viirs_jour['pix_daytime'][i]!=2): viirs_jour=viirs_jour.drop([i])
        if (viirs_truc['pix_daytime'][i]!=3): viirs_truc=viirs_truc.drop([i])
    pd.DataFrame.to_csv(viirs_nuit, 'dataVIIRS/atlas/cible_nuit.txt', sep=',',index=False)
    pd.DataFrame.to_csv(viirs_aube, 'dataVIIRS/atlas/cible_aube.txt', sep=',',index=False)
    pd.DataFrame.to_csv(viirs_jour, 'dataVIIRS/atlas/cible_jour.txt', sep=',',index=False)
    pd.DataFrame.to_csv(viirs_truc, 'dataVIIRS/atlas/cible_truc.txt', sep=',',index=False)
    print 'separation effectuee'
    return viirs_nuit, viirs_aube, viirs_jour, viirs_truc


def separation_clair_nuage(viirs,daytime):
    viirs_clair=viirs
    viirs_nuage=viirs
    for i in xrange(len(viirs['code'])):
        if (viirs['code'][i]>=5):
            viirs_clair=viirs_clair.drop([i])
        else:
            viirs_nuage=viirs_nuage.drop([i])
    
    if daytime==0:
        pd.DataFrame.to_csv(viirs_clair, 'dataVIIRS/atlas/clair_nuage/cible_nuit_clair_frac.txt',sep=',',index=False)
        pd.DataFrame.to_csv(viirs_nuage, 'dataVIIRS/atlas/clair_nuage/cible_nuit_nuage_frac.txt',sep=',',index=False)
    elif daytime==1:
        pd.DataFrame.to_csv(viirs_clair, 'dataVIIRS/atlas/clair_nuage/cible_aube_clair_frac.txt',sep=',',index=False)
        pd.DataFrame.to_csv(viirs_nuage, 'dataVIIRS/atlas/clair_nuage/cible_aube_nuage_frac.txt',sep=',',index=False)
    elif daytime==2:
        pd.DataFrame.to_csv(viirs_clair, 'dataVIIRS/atlas/clair_nuage/cible_jour_clair_frac.txt',sep=',',index=False)
        pd.DataFrame.to_csv(viirs_nuage, 'dataVIIRS/atlas/clair_nuage/cible_jour_nuage_frac.txt',sep=',',index=False)
    elif daytime==3:
        pd.DataFrame.to_csv(viirs_clair, 'dataVIIRS/atlas/clair_nuage/cible_truc_clair_frac.txt',sep=',',index=False)
        pd.DataFrame.to_csv(viirs_nuage, 'dataVIIRS/atlas/clair_nuage/cible_truc_nuage_frac.txt',sep=',',index=False)
    
    return 1


def groupe_classification(viirs,JN):  #modification des valeurs de code, N=0,A=1, J=2, T=3
    
    for i in xrange(len(viirs['code'])):
        if (viirs['code'][i]<=157 and viirs['code'][i]>=150): 
            viirs['code'][i]=1 # terre
        elif (viirs['code'][i]==103): 
            viirs['code'][i]=6  # sable dans la mer mis en VL
        elif (viirs['code'][i]<=107 and viirs['code'][i]>=101): 
            viirs['code'][i]=2 # mer
        elif (viirs['code'][i]==181 or viirs['code'][i]==182): 
            viirs['code'][i]=4 #glace
        elif (viirs['code'][i]==191 or viirs['code'][i]==192): 
            viirs['code'][i]=3 #neige
        elif (viirs['code'][i]==501 or viirs['code'][i]==502 or viirs['code'][i]==601 or viirs['code'][i]==602 or viirs['code'][i]==5): 
            viirs['code'][i]=6 #VL
        elif (viirs['code'][i]==503 or viirs['code'][i]==504 or viirs['code'][i]==7): 
            viirs['code'][i]=8 #low
        elif (viirs['code'][i]==801 or viirs['code'][i]==802 or viirs['code'][i]==606 or viirs['code'][i]==607 or viirs['code'][i]==9):
            viirs['code'][i]=10 #medium
        elif (viirs['code'][i]==812 or viirs['code'][i]==707 or viirs['code'][i]==11): 
            viirs['code'][i]=12 #high opaq
        elif (viirs['code'][i]==608 or viirs['code'][i]==609 or viirs['code'][i]==13):
            viirs['code'][i]=14 #very high opaq
        elif (viirs['code'][i]==701 or viirs['code'][i]==702 or viirs['code'][i]==703 or viirs['code'][i]==704): 
            viirs['code'][i]=15 #thin cirrus
        elif (viirs['code'][i]==811):
            viirs['code'][i]=16 # mean+thick cirrus
        elif (viirs['code'][i]==705 or viirs['code'][i]==706): 
            viirs['code'][i]=18 #cirrus above
        elif (viirs['code'][i]==200 or viirs['code'][i]==19):
            viirs['code'][i]=19 #fractionne
        elif (viirs['code'][i]==900 or viirs['code'][i]==20 ): 
            viirs=viirs.drop([i]) #suppression des indetermines et des non-voulus (5-7-9-11-13 cumuli
        
    
    if (JN==2):
        pd.DataFrame.to_csv(viirs, 'dataVIIRS/atlas/cible_jour_modifiee_frac.txt', sep=',',index=False)
    elif (JN==0):
        pd.DataFrame.to_csv(viirs, 'dataVIIRS/atlas/cible_nuit_modifiee_frac.txt', sep=',',index=False)
    elif (JN==3):
        pd.DataFrame.to_csv(viirs, 'dataVIIRS/atlas/cible_truc_modifiee_frac.txt', sep=',',index=False)
    elif (JN==1):
        pd.DataFrame.to_csv(viirs, 'dataVIIRS/atlas/cible_aube_modifiee_frac.txt', sep=',',index=False)
    else:
        pd.DataFrame.to_csv(viirs, 'dataVIIRS/atlas/test.txt', sep=',',index=False)
        
    print 'fin de la modification'
    return viirs


def gestionGranule_train(viirs,nom=''):
    
    if nom != '':
        nom=nom+'/'+nom
    
    clairOK=np.logical_and(viirs['code']<5,viirs['CMC']==3)
    nuageOK=np.logical_and(viirs['code']>5,viirs['CMC']==0)
    viirsOK=np.logical_or(viirs['CMC']==3,viirs['CMC']==0)
    
    Trainclair=viirs[clairOK]
    Trainnuage=viirs[nuageOK]
    Trainviirs=viirs[viirsOK]
    
    pd.DataFrame.to_csv(Trainclair, 'dataVIIRS/granule_train/'+nom+'_clair.txt', sep=',',index=False)
    pd.DataFrame.to_csv(Trainnuage, 'dataVIIRS/granule_train/'+nom+'_nuage.txt', sep=',',index=False)
    pd.DataFrame.to_csv(Trainviirs, 'dataVIIRS/granule_train/'+nom+'_sepa.txt', sep=',',index=False)
    
    return 1



if __name__ == '__main__':
    
    viirs=pd.read_table('dataVIIRS/test/testalakon2.txt',sep=',')
    
    indexprint()
    
    
    
    print 'wololo'
    
    
    
    
    
    
    
    
    
    
    
    
    