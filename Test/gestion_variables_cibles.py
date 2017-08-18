#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 1 août 2017

@author: almeidamanceroi
'''

# import numpy as np
# import pandas as pd
# import pylab as plt


def jour_de_l_annee(month, day): 
    '''Prend en entree trois entiers (annee, mois, jour) et retourne le string du jour de l'annee (de 001 à 365)'''
    if month == 2 and day == 29: #cas particulier des années bissextiles
        return '059l'
    else:
        j = 0
        if month == 2:
            j = 31
        elif month == 3:
            j = 59
        elif month == 4:
            j = 90
        elif month == 5:
            j = 120
        elif month == 6:
            j = 151
        elif month == 7:
            j = 181
        elif month == 8:
            j = 212
        elif month == 9:
            j = 243
        elif month == 10:
            j = 273
        elif month == 11:
            j = 304
        elif month == 12:
            j = 334
        j = j + day
        strj = str(j)
        if len(strj) == 1:
            strj = '00' + strj
        elif len(strj) == 2:
            strj = '0' + strj
        return strj




if __name__ == '__main__':
    
    
    print jour_de_l_annee(1,4)
       
    print '...end...'
    
    
    
    
    
    
    
    
    
    
    
    
    
