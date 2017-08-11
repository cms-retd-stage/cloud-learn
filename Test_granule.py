'''
Created on Jul 5, 2016

@author: delatailler
'''
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from PIL import Image

'''
----------------------------------------------------------------------------------------------------
'''


def modif_code_nuage(viirs):
    '''
    Modifie les codes pour reduire les donnees d entrainement a 3 classes
    '''
    print 'modif code nuage'
    viirs_nuage=copy.deepcopy(viirs)
    
    clair=viirs['code']<5
    fract=viirs['code']==19
    
    nuage=np.logical_and(viirs['code']>5,viirs['code']!=19)
    
    viirs_nuage.loc[clair,'code']=0
    viirs_nuage.loc[fract,'code']=19
    viirs_nuage.loc[nuage,'code']=2
    
    return viirs_nuage


def post_model(viirs,TT):
    '''
    TT : Train=0, Test=1
    Cree la distinction entre donnees de test et donnees d'entrainement
    '''
    
    if TT==0:
        Y_train=viirs['code']
        X_train=viirs.drop(['code'],1)
        return X_train,Y_train
    
    elif TT==1:
        X_test=viirs.drop(['code'],1)
        return X_test


def ajout_peigne(viirs):
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
    
    viirs.loc[peigne,'code']=20
    viirs.loc[0,'code']=1
    viirs.loc[1,'code']=20
    return viirs    


def ShowMap(viirs,canal,name='',cluster=False):
    '''
    Visualisation de la granule pour le canal donne
    Creation de la palette de couleur MAIA si canal='code'
    '''
    data=viirs[canal]
    data=np.reshape(data, (768,3200))
    if (canal=='code') and (cluster==False):
        fig, ax = plt.subplots()
        
        cmap = mpl.colors.ListedColormap([ '#008c30','black','#ffbbff','#dda0dd','#ffa500','#ff6600','#ffd800','#ffb600','#ffff00','#d8ff00','#cccc00','#d8b575','#ffffff','#ffe0aa','#0000ff','#00b2ff','#00ffe5','#00ffb2','#d800ff','#660f00'])
        cmap.set_over('0.25')
        cmap.set_under('0.75')
        
        #bounds = range(21)
        #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        #cb2 = mpl.colorbar.ColorbarBase(data, cmap=cmap,norm=norm)
        #plt.imshow(data)
        
        cax = ax.imshow(data, cmap=cmap)
        cbar = fig.colorbar(cax, orientation='horizontal')
        
        
    else:
        ok=(data>-700) & (data<999)
        data_ok=np.ma.masked_array(data,mask=~ok)
        
        plt.imshow(data_ok)
        plt.colorbar(orientation="horizontal")
    
    plt.title(name+'_'+canal)
    
    plt.show()
    return 1


def ShowRGB(viirs,R,G,B,name=''):
    '''
    Permet de visualiser la granule en composision coloree
    Enregistre l'image si name!=''
    '''
    
    rgbArray = np.zeros((768,3200,3), 'uint8')
    
    rouge=np.reshape(viirs[R], (768,3200))
    vert=np.reshape(viirs[G], (768,3200))
    bleu=np.reshape(viirs[B], (768,3200))
    
    
    rgbArray[..., 0] = rouge
    rgbArray[..., 1] = vert
    rgbArray[..., 2] = bleu
    img = Image.fromarray(rgbArray)
    
    #img.show()
    if name!='':
        img.save('/rd/merzhin/safnwp/stage/romuald/'+name)
    return 1


def estimateurSplit(test,estimateur,nbSplit,conf=False):
    '''
    Calcul de la prevision en nbSplit pour reduire l'utilisation d'arzhur
    Calcul de la confiance si demande !!! augmentation du temps de calcul
    '''
    long=test.shape[0]
    test['code']=np.zeros(long)+20
    test['conf']=np.zeros(long)
    n=int(long/nbSplit)
    
    for i in range(nbSplit):
        viirs_cut=test.iloc[int(n*i):int(n*(i+1))]
        viirs_cut=viirs_cut.drop('code',1)
        viirs_cut=viirs_cut.drop('conf',1)
        
        viirs_cut['code']=estimateur.predict(viirs_cut)
        test['code'][int(n*i):int(n*(i+1))]=viirs_cut['code']
        
        
        if conf==True:
            viirs_cut=viirs_cut.drop('code',1)
            confiance=estimateur.predict_proba(viirs_cut)
        
            viirs_cut['conf']=np.amax(confiance, axis=1)
            test['conf'][int(n*i):int(n*(i+1))]=viirs_cut['conf']
            
        
    return test['code'],test['conf']


def SaveCode(viirs,name='test.ppm'):
    '''
    Enregistre le code de la granule sous forme d'image
    '''
    
    code=viirs['code']
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
    
    rgbArray= np.zeros((768,3200,3), 'uint8')
    rouge=np.reshape(viirs['rouge'], (768,3200))
    vert=np.reshape(viirs['vert'], (768,3200))
    bleu=np.reshape(viirs['bleu'], (768,3200))
    rgbArray[..., 0] = rouge
    rgbArray[..., 1] = vert
    rgbArray[..., 2] = bleu
    
    
    img = Image.fromarray(rgbArray)
    
    img.show()
    img.save('/rd/merzhin/safnwp/stage/romuald/'+name)
    
    return 1


'''
----------------------------------------------------------------------------------------------------
'''


def model_sepa_jour(X):
    
    X['SP5_11']=(X.Obs_06-X.Obs_22)/(X.Obs_06+X.Obs_22) #classif neige mieux
    X['SP10_11']=(X.Obs_16-X.Obs_22)/(X.Obs_16+X.Obs_22)
    X['SP14_13']=100*(X.Obs_87-X.Obs_40)/(X.Obs_87+X.Obs_40)
    X['SP4_7']=(X.Obs_04-X.Obs_08)/(X.Obs_04+X.Obs_08)
    
    X['B14_15']=X.Obs_87/X.Obs_108 
    X['B16_15']=X.Obs_120/X.Obs_108 
    X['B13_15']=X.Obs_40/X.Obs_108 
    X['B15_12']=X.Obs_108/X.Obs_37
    
    X['test']=X['SP10_11']<0
    X['test2']=X['test']*X['SP5_11']
    
    
    X['ref']=X.Obs_22-X.Obs_16
    pos=X['ref']>0
    X['refcu']=X['ref']*pos
    X['rouge']=X['Obs_08']-0.5*X['refcu']-0.25*X['Obs_13']
    X['bleu']=X['Obs_22']+X['Obs_13']
    

    dataset=['rouge','bleu','test2','B15_12','B13_15','B16_15','B14_15','SP5_11','SP10_11','SP14_13','SP4_7','pix_alt','pix_tm']
    
    allObs=['Obs_04','Obs_05','Obs_06','Obs_08','Obs_13','Obs_16','Obs_22','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120'] 
    for i in xrange(len(allObs)):
        for j in xrange(len(allObs)):
            if j>i:
                X[allObs[i]+allObs[j]+'DAKKA']=X[allObs[i]]-X[allObs[j]]
                dataset=np.append(dataset,allObs[i]+allObs[j]+'DAKKA')
    
    
    data=X[dataset]
    data['code']=X['code']
    return data


def model_sepa_nuit(X):
    
    X['SP15_12']=100*(X.Obs_108-X.Obs_37)/(X.Obs_108+X.Obs_37)
    X['SP14_12']=100*(X.Obs_87-X.Obs_37)/(X.Obs_87+X.Obs_37)
    
    X['B15_12']=X.Obs_108/X.Obs_37
    X['B16_15']=X.Obs_120/X.Obs_108 
    
    X['test4']=(X.Obs_108-X.Obs_37)*(X.Obs_108-X.Obs_37)
    
    dataset=['test4','B15_12','B16_15','SP15_12','SP14_12','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120']
    allObs=['Obs_37','Obs_40','Obs_87','Obs_108','Obs_120']
    
    for i in xrange(len(allObs)):
        for j in xrange(len(allObs)):
            if j>i:
                X[allObs[i]+allObs[j]+'DAKKA']=X[allObs[i]]-X[allObs[j]]
                dataset=np.append(dataset,allObs[i]+allObs[j]+'DAKKA')
    
     
    data=X[dataset]
    data['code']=X['code']
    return data


def model_sepa_aube(X):
    '''
    !!! NON VALABLE
    '''
    dataset=['pix_tm']
    allObs=['Obs_04','Obs_05','Obs_06','Obs_08','Obs_13','Obs_16','Obs_22','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120'] 
    
    for i in xrange(len(allObs)):
        for j in xrange(len(allObs)):
            if j>i:
                X[allObs[i]+allObs[j]+'DAKKA']=X[allObs[i]]-X[allObs[j]]
                dataset=np.append(dataset,allObs[i]+allObs[j]+'DAKKA')
    
    data=X[dataset]
    data['code']=X['code']
    return data


def model_tri_nuage_simple_jour(X):
    
    X['SP10_11']=(X.Obs_16-X.Obs_22)
    
    X['B14_15']=X.Obs_87/X.Obs_108
    X['B16_15']=X.Obs_120/X.Obs_108 
    
    dataset=['B14_15','B16_15','SP10_11','Obs_108','Obs_120','Obs_87','Obs_13']
    
    allObs=['Obs_04','Obs_05','Obs_06','Obs_08','Obs_13','Obs_16','Obs_22','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120'] 
    for i in xrange(len(allObs)):
        for j in xrange(len(allObs)):
            if j>i:
                X[allObs[i]+allObs[j]+'DAKKA']=X[allObs[i]]-X[allObs[j]]
                dataset=np.append(dataset,allObs[i]+allObs[j]+'DAKKA')
    
   
    data=X[dataset]
    data['code']=X['code']
    return data
    
 
def model_tri_clair_simple_jour(X):
    
    X['SP4_7']=(X.Obs_05-X.Obs_08)/(X.Obs_05+X.Obs_08)
    X['SP5_11']=(X.Obs_06-X.Obs_22)/(X.Obs_06+X.Obs_22) #classif neige
    
    dataset=['SP5_11','SP4_7','pix_tm','Obs_108']
    
    data=X[dataset]
    data['code']=X['code']
    return data


def model_tri_nuage_simple_aube(X):
    '''
    !!! NON VALABLE
    '''
    dataset=['pix_tm']
    allObs=['Obs_04','Obs_05','Obs_06','Obs_08','Obs_13','Obs_16','Obs_22','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120'] 
    
    for i in xrange(len(allObs)):
        for j in xrange(len(allObs)):
            if j>i:
                X[allObs[i]+allObs[j]+'DAKKA']=X[allObs[i]]-X[allObs[j]]
                dataset=np.append(dataset,allObs[i]+allObs[j]+'DAKKA')
    
    data=X[dataset]
    data['code']=X['code']
    return data


def model_tri_clair_simple_aube(X):
    '''
    !!! NON VALABLE
    '''
    dataset=['pix_tm']
    allObs=['Obs_04','Obs_05','Obs_06','Obs_08','Obs_13','Obs_16','Obs_22','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120'] 
    
    for i in xrange(len(allObs)):
        for j in xrange(len(allObs)):
            if j>i:
                X[allObs[i]+allObs[j]+'DAKKA']=X[allObs[i]]-X[allObs[j]]
                dataset=np.append(dataset,allObs[i]+allObs[j]+'DAKKA')
    
    data=X[dataset]
    data['code']=X['code']
    return data


def model_tri_nuage_simple_nuit(X):
    
    X['SP15_16']=100*(X.Obs_108-X.Obs_120)/(X.Obs_108+X.Obs_120)
    X['SP15_12']=100*(X.Obs_108-X.Obs_37)/(X.Obs_108+X.Obs_37)
    X['SP14_13']=100*(X.Obs_87-X.Obs_40)/(X.Obs_87+X.Obs_40)
    X['SP14_12']=100*(X.Obs_87-X.Obs_37)/(X.Obs_87+X.Obs_37)
    
    X['B16_15']=X.Obs_120/X.Obs_108 
    X['B13_15']=X.Obs_40/X.Obs_108 
   
    dataset=['B13_15','B16_15','SP15_16','SP15_12','SP14_13','SP14_12','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120']
    data=X[dataset]
    data['code']=X['code']
    return data


def model_tri_clair_simple_nuit(X):
    
    X['SP14_12']=100*(X.Obs_87-X.Obs_37)/(X.Obs_87+X.Obs_37)
    X['SP13_12']=100*(X.Obs_40-X.Obs_37)/(X.Obs_40+X.Obs_37)
    X['SP15_14']=100*(X.Obs_108-X.Obs_87)/(X.Obs_108+X.Obs_87)
    X['SP15_12']=100*(X.Obs_108-X.Obs_37)/(X.Obs_108+X.Obs_37)
    
    dataset=['pix_tm','SP14_12','SP13_12','SP15_14','SP15_12','Obs_108','Obs_37','Obs_40','Obs_87','Obs_120']
    data=X[dataset]
    data['code']=X['code']
    return data


def model_cluster_jour(X):
 
    dataset=['Obs_04','Obs_05','Obs_06','Obs_08','Obs_13','Obs_16','Obs_22','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120']
    
    sds=StandardScaler()                    #
    X[dataset]=sds.fit_transform(X[dataset])# ceci sert a mettre les variables a la meme echelle
    
    data=X[dataset]
    return data
 

def model_cluster_nuit(X):
    
    X['B13_15']=X.Obs_40/X.Obs_108
    X['B16_15']=X.Obs_120/X.Obs_108
    X['B15_12']=X.Obs_108/X.Obs_37
    
    X['SP15_12']=100*(X.Obs_108-X.Obs_37)/(X.Obs_108+X.Obs_37)
    X['SP14_12']=100*(X.Obs_87-X.Obs_37)/(X.Obs_87+X.Obs_37)
    X['SP14_13']=100*(X.Obs_87-X.Obs_40)/(X.Obs_87+X.Obs_40)
    
    X['test4']=(X.Obs_108-X.Obs_37)*(X.Obs_108-X.Obs_37) 
    
    dataset=['pix_tm','B13_15','B16_15','B15_12','SP15_12','SP14_12','SP14_13','test4','Obs_37','Obs_40','Obs_87','Obs_108','Obs_120']
    
    sds=StandardScaler()
    X[dataset]=sds.fit_transform(X[dataset])
    
    data=X[dataset]
    return data


'''
----------------------------------------------------------------------------------------------------
'''


def test_granule(viirs,taille_foret=2500,confiance=False,fractionne=False,train_VIIRS=False):
    '''
    confiance: calcul de la certitude de l'algorithme
    fractionne: prise en compte des fractionne (code=19)
    train_VIIRS: entrainement sur granules
    
    '''
    if fractionne==False:
        frac=''
    else:
        frac='_frac'
    
    ratio=0.01 # % des donnees VIIRS utilisee si train_VIIRS=True
    
    VIIRS_target=viirs
    VIIRS_len=len(VIIRS_target['Obs_08'])
    VIIRS_target['code']=np.zeros(VIIRS_len)+20
    VIIRS_target['confsep']=np.zeros(VIIRS_len)
    VIIRS_target['conftri']=np.zeros(VIIRS_len)
    VIIRS_target['conf']=np.zeros(VIIRS_len)
    
    nuit=(VIIRS_target['pix_daytime']==0)
    aube=(VIIRS_target['pix_daytime']==1)
    jour=(VIIRS_target['pix_daytime']==2)
    
    
    
    '''
    NUIT
    '''
    print 'Classification Nuit \n'
    
    if (len(VIIRS_target[nuit])==0):
        print 'Pas de nuit\n'
    
    else:
    
        print 'Separation clair/nuage \n'
        
        #creation des donnees d'entrainement
        if train_VIIRS==False:
            viirs_train=pd.read_table('dataVIIRS/atlas/sepa/cible_nuit_modifiee'+frac+'.txt',sep=',')
        
        else:
            viirs_train=pd.read_table('/rd/merzhin/safnwp/stage/romuald/dataset/NuitSepa.txt',sep=',')
                
            
                
            if fractionne==False:
                fract=viirs_train['code']==19
                viirs_train=viirs_train.drop(viirs_train.index[fract])
                
            taille=viirs_train.shape[0]
            n=int(ratio*taille)
            viirs_data=viirs_train.iloc[np.random.permutation(taille)]
            viirs_data=viirs_data.reset_index(drop=True)
            viirs_train=viirs_data.iloc[:n]
                
            
            
        
        
        viirs_train=modif_code_nuage(viirs_train)
        
        viirs_train=model_sepa_nuit(viirs_train)
        X_train,Y_train=post_model(viirs_train,0)
        
        #creation des donnees de test
        test_sepa=model_sepa_nuit(VIIRS_target[nuit])
        test_sepa=post_model(test_sepa,1)
        
        
        print 'creation estimateur'
        estimateurNS=RandomForestClassifier(n_estimators=taille_foret,min_samples_leaf=25,n_jobs=-1)
        print 'calcul fit'
        estimateurNS.fit(X_train,Y_train)
        print 'prediction'
        
        
        VIIRS_target['code'][nuit],VIIRS_target['confsep'][nuit]=estimateurSplit(test_sepa, estimateurNS, 10,confiance)
        
        nuitclair=np.logical_and(VIIRS_target['pix_daytime']==0,VIIRS_target['code']==0)
        nuitnuage=np.logical_and(VIIRS_target['pix_daytime']==0,VIIRS_target['code']==2)
        
        print VIIRS_target[nuitclair].shape[0]
        print VIIRS_target[nuitnuage].shape[0]
        
               
        
       
        print '\nTri Clair \n'
        if (len(VIIRS_target[nuitclair])==0):
            print 'Pas de clair'
        else:
            #creation des donnees d'entrainement
            
            if train_VIIRS==False:
                viirs_train=pd.read_table('dataVIIRS/atlas/clair_nuage/cible_nuit_clair'+frac+'.txt',sep=',')
           
            else:
                viirs_train=pd.read_table('/rd/merzhin/safnwp/stage/romuald/dataset/NuitClair.txt',sep=',')
                
                if fractionne==False:
                    fract=viirs_train['code']==19
                    viirs_train=viirs_train.drop(viirs_train.index[fract])
                
                taille=viirs_train.shape[0]
                n=int(ratio*taille)
                viirs_data=viirs_train.iloc[np.random.permutation(taille)]
                viirs_data=viirs_data.reset_index(drop=True)
                viirs_train=viirs_data.iloc[:n]
                    
                           
            
            
            viirs_train=model_tri_clair_simple_nuit(viirs_train)
            X_train,Y_train=post_model(viirs_train,0)
            
            #creation des donnees de test
            test_clair=model_tri_clair_simple_nuit(VIIRS_target[nuitclair])
            test_clair=post_model(test_clair,1)
            
            print 'creation estimateur'
            estimateurNTC=RandomForestClassifier(n_estimators=taille_foret,min_samples_leaf=25,n_jobs=-1)
            print 'calcul fit'
            estimateurNTC.fit(X_train,Y_train)
            print 'prediction'
            
            
            VIIRS_target['code'][nuitclair],VIIRS_target['conftri'][nuitclair]=estimateurSplit(test_clair,estimateurNTC , 10,confiance)
        
        
        print '\nTri Nuage'
        if (len(VIIRS_target[nuitnuage])==0):
            print 'Pas de nuage'
        else:
            #creation des donnees d'entrainement
            
            if train_VIIRS==False:
                viirs_train=pd.read_table('dataVIIRS/atlas/clair_nuage/cible_nuit_nuage'+frac+'.txt',sep=',')
            
            else:
                viirs_train=pd.read_table('/rd/merzhin/safnwp/stage/romuald/dataset/NuitNuage.txt',sep=',')
                
                
                if fractionne==False:
                    fract=viirs_train['code']==19
                    viirs_train=viirs_train.drop(viirs_train.index[fract])
                
                taille=viirs_train.shape[0]
                n=int(ratio*taille)
                viirs_data=viirs_train.iloc[np.random.permutation(taille)]
                viirs_data=viirs_data.reset_index(drop=True)
                viirs_train=viirs_data.iloc[:n]
                    
                
            
            
            viirs_train=model_tri_nuage_simple_nuit(viirs_train)
            X_train,Y_train=post_model(viirs_train,0)
            
            #creation des donnees de test
            test_nuage=model_tri_nuage_simple_nuit(VIIRS_target[nuitnuage])
            test_nuage=post_model(test_nuage,1)
            
            print 'creation estimateur'
            estimateurNTN=RandomForestClassifier(n_estimators=taille_foret,min_samples_leaf=25,n_jobs=-1)
            print 'calcul fit'
            estimateurNTN.fit(X_train,Y_train)
            print 'prediction'
            
            
            VIIRS_target['code'][nuitnuage],VIIRS_target['conftri'][nuitnuage]=estimateurSplit(test_nuage,estimateurNTN , 10,confiance)
        
    
    
    
    
    
    '''
    AUBE
    '''
    print 'Classification Aube \n'
    if (len(VIIRS_target[aube])==0):
        print 'Pas d aube\n'
    else:
        VIIRS_target['code'][aube]=20
        print '|*********************|'
        print 'Pas de modele'
        print '|*********************|'
    '''
    Aucun modele fiable, code donne a titre indicatif
    '''
    '''
    
        print 'Separation clair/nuage \n'
        
        #creation des donnees d'entrainement
        if train_VIIRS==False:
            
            VIIRS_target['code'][aube]=20
            print '|*********************|'
            print 'Pas de modele'
            print '|*********************|'
                
        else:
            #creation des donnees d'entrainement    
            viirs_train=pd.read_table('/rd/merzhin/safnwp/stage/romuald/dataset/AubeSepa.txt',sep=',')
            
                
            if fractionne==False:
                fract=viirs_train['code']==19
                viirs_train=viirs_train.drop(viirs_train.index[fract])
                
            taille=viirs_train.shape[0]
            n=int(ratio*taille)
            viirs_data=viirs_train.iloc[np.random.permutation(taille)]
            viirs_data=viirs_data.reset_index(drop=True)
            viirs_train=viirs_data.iloc[:n]
            
            viirs_train=modif_code_nuage(viirs_train)
            
            viirs_train=model_sepa_aube(viirs_train)
            X_train,Y_train=post_model(viirs_train,0)
            
            #creation des donnees de test
            test_sepa=model_sepa_aube(VIIRS_target[aube])
            test_sepa=post_model(test_sepa,1)
            
            
            print 'creation estimateur'
            estimateurAS=RandomForestClassifier(n_estimators=taille_foret,min_samples_leaf=25,n_jobs=-1)
            print 'calcul fit'
            estimateurAS.fit(X_train,Y_train)
            print 'prediction'
            
            VIIRS_target['code'][aube],VIIRS_target['confsep'][aube]=estimateurSplit(test_sepa, estimateurAS, 10,confiance)
            
            aubeclair=np.logical_and(VIIRS_target['pix_daytime']==0,VIIRS_target['code']==0)
            aubenuage=np.logical_and(VIIRS_target['pix_daytime']==0,VIIRS_target['code']==2)
            
            print VIIRS_target[aubeclair].shape[0]
            print VIIRS_target[aubenuage].shape[0]
            
           
            print '\nTri Clair \n'
            if (len(VIIRS_target[aubeclair])==0):
                print 'Pas de clair'
            else:
                #creation des donnees d'entrainement
                
                
                viirs_train=pd.read_table('/rd/merzhin/safnwp/stage/romuald/dataset/AubeClair.txt',sep=',')
                
                
                
                if fractionne==False:
                    fract=viirs_train['code']==19
                    viirs_train=viirs_train.drop(viirs_train.index[fract])
                
                taille=viirs_train.shape[0]
                n=int(ratio*taille)
                viirs_data=viirs_train.iloc[np.random.permutation(taille)]
                viirs_data=viirs_data.reset_index(drop=True)
                viirs_train=viirs_data.iloc[:n]
                
                
                viirs_train=model_tri_clair_simple_aube(viirs_train)
                X_train,Y_train=post_model(viirs_train,0)
                
                #creation des donnees de test
                test_clair=model_tri_clair_simple_aube(VIIRS_target[aubeclair])
                test_clair=post_model(test_clair,1)
                
                print 'creation estimateur'
                estimateurATC=RandomForestClassifier(n_estimators=taille_foret,min_samples_leaf=25,n_jobs=-1)
                print 'calcul fit'
                estimateurATC.fit(X_train,Y_train)
                print 'prediction'
                
                VIIRS_target['code'][aubeclair],VIIRS_target['conftri'][aubeclair]=estimateurSplit(test_clair,estimateurATC , 10,confiance)
            
            
            
            print '\nTri Nuage'
            if (len(VIIRS_target[aubenuage])==0):
                print 'Pas de nuage'
            
            else:
                #creation des donnees d'entrainement
                
                viirs_train=pd.read_table('/rd/merzhin/safnwp/stage/romuald/dataset/NuitNuage.txt',sep=',')
                
                
                if fractionne==False:
                    fract=viirs_train['code']==19
                    viirs_train=viirs_train.drop(viirs_train.index[fract])
                
                taille=viirs_train.shape[0]
                n=int(ratio*taille)
                viirs_data=viirs_train.iloc[np.random.permutation(taille)]
                viirs_data=viirs_data.reset_index(drop=True)
                viirs_train=viirs_data.iloc[:n]
                 
                
                
                viirs_train=model_tri_nuage_simple_aube(viirs_train)
                X_train,Y_train=post_model(viirs_train,0)
                
                #creation des donnees de test
                test_nuage=model_tri_nuage_simple_aube(VIIRS_target[aubenuage])
                test_nuage=post_model(test_nuage,1)
                
                print 'creation estimateur'
                estimateurATN=RandomForestClassifier(n_estimators=taille_foret,min_samples_leaf=25,n_jobs=-1)
                print 'calcul fit'
                estimateurATN.fit(X_train,Y_train)
                print 'prediction'
                
                VIIRS_target['code'][aubenuage],VIIRS_target['conftri'][aubenuage]=estimateurSplit(test_nuage,estimateurATN , 10,confiance)
            
       
    '''
       
     

    
    '''
    JOUR
    '''
    print 'Classification Jour \n'
    
    if (len(VIIRS_target[jour])==0):
        print 'Pas de jour\n'
    else:
        print 'Separation clair/nuage \n'
        
        #creation des donnees d'entrainement
        if train_VIIRS==False:
                viirs_train=pd.read_table('dataVIIRS/atlas/sepa/cible_jour_modifiee'+frac+'.txt',sep=',')
        else:
            viirs_train=pd.read_table('/rd/merzhin/safnwp/stage/romuald/dataset/JourSepa.txt',sep=',')
            
            if fractionne==False:
                fract=viirs_train['code']==19
                viirs_train=viirs_train.drop(viirs_train.index[fract])
            
            taille=viirs_train.shape[0]
            print taille
            n=int(ratio*taille)
            viirs_data=viirs_train.iloc[np.random.permutation(taille)]
            viirs_data=viirs_data.reset_index(drop=True)
            viirs_train=viirs_data.iloc[:n]
        
        
        viirs_train=modif_code_nuage(viirs_train)
        
        
        viirs_train=model_sepa_jour(viirs_train)
        X_train,Y_train=post_model(viirs_train,0)
        
        #creation des donnees de test
        test_sepa=model_sepa_jour(VIIRS_target[jour])
        test_sepa=post_model(test_sepa,1)
        
        
        print 'creation estimateur'
        estimateurJS=RandomForestClassifier(n_estimators=taille_foret,min_samples_leaf=25,n_jobs=-1)
        print 'calcul fit'
        estimateurJS.fit(X_train,Y_train)
        print 'prediction'
        
        VIIRS_target['code'][jour],VIIRS_target['confsep'][jour]=estimateurSplit(test_sepa,estimateurJS , 10,confiance)
        
        
        jourclair=np.logical_and(VIIRS_target['pix_daytime']==2,VIIRS_target['code']==0)
        journuage=np.logical_and(VIIRS_target['pix_daytime']==2,VIIRS_target['code']==2)
        
        
        
        '''
        CLAIR
        '''
        print '\nTri Clair \n'
        if (len(VIIRS_target[jourclair])==0):
            print 'Pas de clair'
        else:
            #creation des donnees d'entrainement
            if train_VIIRS==False:
                viirs_train=pd.read_table('dataVIIRS/atlas/clair_nuage/cible_jour_clair'+frac+'.txt',sep=',')
                '''
                neigechaude=np.logical_and(viirs_train['Obs_108']>272,viirs_train['code']==3)
                viirs_train=viirs_train.drop(viirs_train.index[neigechaude])
                '''
            else:
                viirs_train=pd.read_table('/rd/merzhin/safnwp/stage/romuald/dataset/JourClair.txt',sep=',')
            
                taille=viirs_train.shape[0]
                n=int(ratio*taille)
                viirs_data=viirs_train.iloc[np.random.permutation(taille)]
                viirs_data=viirs_data.reset_index(drop=True)
                viirs_train=viirs_data.iloc[:n]
                
                
            viirs_train=model_tri_clair_simple_jour(viirs_train)
            X_train,Y_train=post_model(viirs_train,0)
            
            #creation des donnees de test
            test_clair=model_tri_clair_simple_jour(VIIRS_target[jourclair])
            test_clair=post_model(test_clair,1)
            
            print 'creation estimateur'
            estimateurJTC=RandomForestClassifier(n_estimators=taille_foret,min_samples_leaf=25,n_jobs=-1)
            print 'calcul fit'
            estimateurJTC.fit(X_train,Y_train)
            print 'prediction'
            
            VIIRS_target['code'][jourclair],VIIRS_target['conftri'][jourclair]=estimateurSplit(test_clair,estimateurJTC , 10,confiance)
            
        '''
        NUAGE
        '''
        print '\nTri Nuage'
        if (len(VIIRS_target[journuage])==0):
            print 'Pas de nuage'
        else:
            #creation des donnees d'entrainement
            if train_VIIRS==False:
                viirs_train=pd.read_table('dataVIIRS/atlas/clair_nuage/cible_jour_nuage'+frac+'.txt',sep=',')
                
            else:
                viirs_train=pd.read_table('/rd/merzhin/safnwp/stage/romuald/dataset/JourNuage.txt',sep=',')
            
                
                taille=viirs_train.shape[0]
                n=int(ratio*taille)
                viirs_data=viirs_train.iloc[np.random.permutation(taille)]
                viirs_data=viirs_data.reset_index(drop=True)
                viirs_train=viirs_data.iloc[:n]
                
            
            
            viirs_train=model_tri_nuage_simple_jour(viirs_train)
            
            test=viirs_train['code']<5
            print len(viirs_train['code'][test])
            
            X_train,Y_train=post_model(viirs_train,0)
            
            #creation des donnees de test
            test_nuage=model_tri_nuage_simple_jour(VIIRS_target[journuage])
            test_nuage=post_model(test_nuage,1)
            
            print 'creation estimateur'
            estimateurJTN=RandomForestClassifier(n_estimators=taille_foret,min_samples_leaf=25,n_jobs=-1)
            print 'calcul fit'
            estimateurJTN.fit(X_train,Y_train)
            print 'prediction'
            
            VIIRS_target['code'][journuage],VIIRS_target['conftri'][journuage]=estimateurSplit(test_nuage,estimateurJTN , 10,confiance)
            
        
        
    VIIRS_target=ajout_peigne(VIIRS_target)
    VIIRS_target['conf']=VIIRS_target['confsep']*VIIRS_target['conftri']
    return VIIRS_target


def test_cluster(viirs,nb_cluster=13):
    '''
    Tente de trouver nb_cluster dans la granule 
    '''
    
    VIIRS_target = viirs
    VIIRS_len = len(VIIRS_target['pix_daytime'])
    VIIRS_target['code'] = np.zeros(VIIRS_len) + 20
    
    nuit=(VIIRS_target['pix_daytime']==0)
    aube=(VIIRS_target['pix_daytime']==1)
    jour=(VIIRS_target['pix_daytime']==2)
    
    
    '''
    NUIT
    '''
    print 'Classification Nuit \n'
    
    if (len(VIIRS_target[nuit])==0):
        print 'Pas de nuit\n'
    
    else:
        test_sepa=model_cluster_nuit(VIIRS_target[nuit])
        
        print 'creation de l estimateur'
        estimateurN=KMeans(n_clusters=nb_cluster,precompute_distances='auto',n_jobs=-1)
        
        print 'prediction'
        VIIRS_target['code'][nuit]=estimateurN.fit_predict(test_sepa)
        
    
    '''
    AUBE
    '''
    print 'Classification Aube \n'
    
    if (len(VIIRS_target[aube])==0):
        print 'Pas d aube\n'
    
    else:
        VIIRS_target['code'][aube]=20
        print '|*********************|'
        print 'Pas de modele'
        print '|*********************|'
    
    
    
    '''
    JOUR
    '''
    print 'Classification Jour \n'
    
    if (len(VIIRS_target[jour])==0):
        print 'Pas de jour\n'
    
    else:
        
        test_sepa=model_cluster_jour(VIIRS_target[jour])
        
        print 'creation de l estimateur'
        estimateurJ=KMeans(n_clusters=nb_cluster,precompute_distances='auto',n_jobs=-1)
        
        
        
        print 'prediction'
        VIIRS_target['code'][jour]=estimateurJ.fit_predict(test_sepa)
    
    
    VIIRS_target=ajout_peigne(VIIRS_target)
    return VIIRS_target
    

'''
----------------------------------------------------------------------------------------------------
'''



if __name__ == '__main__':
    
    granule='VIIRS2'
    print granule
    viirs=pd.read_table('dataVIIRS/test/'+granule+'.txt',sep=',')
    
    viirs=test_cluster(viirs, 20)
    
    
    #viirs=test_granule(viirs,confiance=False,fractionne=False,train_VIIRS=False)

    
    #SaveCode(viirs,granule+'/traincrash.ppm')
    
    ShowMap(viirs, 'code', granule,cluster=True)
    
    print 'wololo'
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    