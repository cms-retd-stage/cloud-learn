'''
Created on 17 juin 2016

@author: roquetp; delatailler
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from mpl_toolkits.basemap import Basemap, cm
from modif_code_nuage import modif_code_nuage
from pandas.tools.plotting import andrews_curves

from mpl_toolkits.mplot3d import Axes3D


def drawMap(plat,plon,tm,code,test):
    """
    Montre la carte des cibles d'entrainement
    test=1: affichage type de terrain + nuage/clair
    test=2: affichage confusion clair/nuage
    test=3: affichage code simple
    """
    
    lat_min = 25
    lat_max = 80
    lon_min = -40
    lon_max = 40
    m = Basemap(projection='cyl',llcrnrlat=lat_min,urcrnrlat=lat_max,\
            llcrnrlon=lon_min ,urcrnrlon=lon_max,resolution='c')
    m.drawcoastlines()
    parallels = np.arange(-90.,90,10.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    # draw meridians
    meridians = np.arange(-180.,180.,10.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    x, y = m(plon,plat ) # compute map proj coordinates.
    #pix_tm :   : 0=sea, 1=land, 2=desert, 3=long-lasting snow,  4=coast, 5=inland water   
    if (test==1):
        sea = (tm == 0)
        m.scatter(x[sea],y[sea],s=15, marker ="o", color = "DarkBlue")
    
        lacs = (tm == 1)
        m.scatter(x[lacs],y[lacs],s=15, marker ="o", color = "purple")  
     
        terre = (tm == 2) 
        m.scatter(x[terre],y[terre],s=15, marker ="o", color = "DarkGreen")
    
        desert = (tm == 3)
        m.scatter(x[desert],y[desert],s=15, marker ="o", color = "Goldenrod")
    
        ice = (tm == 4)
        m.scatter(x[ice],y[ice],s=15, marker ="o", color = "cyan")
    
        coast = (tm == 5)
        m.scatter(x[coast],y[coast],s=15, marker ="o", color = "Orange")
    
    
        nuage = (code == True)
        m.scatter(x[nuage],y[nuage],s=5, marker ="o", color = "silver")
        
        plt.show()
    
    elif (test==2):
        vrainuage = (code == 1)
        m.scatter(x[vrainuage],y[vrainuage],s=15, marker ="o", color = "grey")
        
        fauxnuage = (code == 3)
        m.scatter(x[fauxnuage],y[fauxnuage],s=15, marker ="o", color = "yellow")
        
        fauxautre = (code == 2)
        m.scatter(x[fauxautre],y[fauxautre],s=15, marker ="o", color = "red")
        
        vraiautre = (code == 4)
        m.scatter(x[vraiautre],y[vraiautre],s=15, marker ="o", color = "darkgreen")
        
        plt.show()
    
    elif (test==3):
        C01= (code == 1)
        m.scatter(x[C01],y[C01],s=5, marker ="o", color='#008c30')
     
        C02= (code == 2)
        m.scatter(x[C02],y[C02],s=5, marker ="o", color='black')
        
        C03= (code == 3)
        m.scatter(x[C03],y[C03],s=5, marker ="o", color='#ffbbff')
        
        C04= (code == 4)
        m.scatter(x[C04],y[C04],s=5, marker ="o", color='#dda0dd')
        
        C05= (code == 5)
        m.scatter(x[C05],y[C05],s=5, marker ="o", color='#ffa500')
        
        C06= (code == 6)
        m.scatter(x[C06],y[C06],s=5, marker ="o", color='#ff6600')
        
        C07= (code == 7)
        m.scatter(x[C07],y[C07],s=5, marker ="o", color='#ffd800')
        
        C08= (code == 8)
        m.scatter(x[C08],y[C08],s=5, marker ="o", color='#ffb600')
        
        C09= (code == 9)
        m.scatter(x[C09],y[C09],s=5, marker ="o", color='#ffff00')
        
        C10= (code ==10 )
        m.scatter(x[C10],y[C10],s=5, marker ="o", color='#d8ff00')
        
        C11= (code == 11)
        m.scatter(x[C11],y[C11],s=5, marker ="o", color='#cccc00')
        
        C12= (code == 12)
        m.scatter(x[C12],y[C12],s=5, marker ="o", color='#d8b575')
        
        C13= (code == 13)
        m.scatter(x[C13],y[C13],s=5, marker ="o", color='#ffffff')
        
        C14= (code == 14)
        m.scatter(x[C14],y[C14],s=5, marker ="o", color='#ffe0aa')
        
        C15= (code ==15 )
        m.scatter(x[C15],y[C15],s=5, marker ="o", color='#0000ff') 
        
        C16= (code == 16)
        m.scatter(x[C16],y[C16],s=5, marker ="o", color='#00b2ff')
        
        C17= (code == 17)
        m.scatter(x[C17],y[C17],s=5, marker ="o", color='#00ffe5')
        
        C18= (code ==18 )
        m.scatter(x[C18],y[C18],s=5, marker ="o", color='#00ffb2')
        
        C19= (code ==19 )
        m.scatter(x[C19],y[C19],s=5, marker ="o", color='#d800ff')
        plt.show()
    return 1
        
    """
    # en fait on a
    # sea = 0
    # terre = 2
    # desert = 3
    # ice = 4
    # lac = 1
    """
    
 
def drawhist(viirs,donnees,name=''):
    '''
    Affiche l'histogramme d'une variable
    '''
    
    plt.hist(viirs[donnees],30)
    plt.title(name+'_'+donnees)
    plt.show()
    
    return 1


def printimportance(data,estimateur):
    '''
    Affiche l'importance des entrees pour un random forest
    '''
    imp=estimateur.feature_importances_
    print 'Importances:'
    for i in xrange(len(imp)):
        print data[i],'\t',imp[i]
    print '\n'
    return 1

def printimportance2(data,estimateur):
    '''
    Affiche l'importance des entrees pour un random forest (en ordre)
    '''
    imp=estimateur.feature_importances_
    ind_imp_sort=list(reversed(np.argsort(imp)))
    print 'Importances:'
    for i in ind_imp_sort:
        print data[i],'  ',imp[i]
    print '\n'
    return 1


def draw3D(viirs,xs,ys,zs,code):
    '''
    Affiche le scatterplot 3D de 3 variables
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotX=viirs[xs]
    plotY=viirs[ys]
    plotZ=viirs[zs]
    
    ax.scatter(plotX, plotY, plotZ,c=viirs[code])
    
    
    ax.set_xlabel(xs)
    ax.set_ylabel(ys)
    ax.set_zlabel(zs)
    ax.legend()
    plt.show()
    
    return 1
 

def ViirsShow(viirs,canal):
    data=viirs[canal]
    data=np.reshape(data, (768,3200))
    ok=(data>-999)
    data_ok=np.ma.masked_array(data,mask=~ok)
    viirs[canal]=data_ok
    plt.imshow(data_ok)
    plt.colorbar(orientation="horizontal")
    plt.title(canal)
    plt.show()
    return 1


if __name__ == '__main__':
    #fic1 = "/rd/merzhin/safnwp/stage/romuald/dataset/JourSepa.txt"
#     fic1 = "dataVIIRS/atlas/sepa/cible_jour_modifiee_frac.txt"
#     viirs = pd.read_csv(fic1, sep=',')
#     
#     neige=viirs['code']==1
#     
#     drawhist(viirs[neige], 'Obs_108', 'terre')


    
    fic1 = "/home/mcms/almeidamanceroi/workspace/workspace/VIIRS/Nouvelles_cibles_a_utiliser/simple_cible_nouveau_modif_sst.txt"
    viirs = pd.read_csv(fic1, sep=',')
    
    
    drawhist(viirs, 'code', 'codes parmi les cibles')
    
    
    
    '''
    lats = viirs["pix_lat"]
    lons = viirs["pix_lon"]
    tm = viirs["pix_tm"]
    code=viirs["code"]
    drawMap(lats,lons,tm,code,3)
    '''
    print 'wololo'
    
    
    
    
    
    
    
    
 