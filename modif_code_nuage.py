'''
Created on 20 juin 2016

@author: delatailler
'''

def modif_code_nuage(viirs): #separation nuage/reste
    viirs_nuage=viirs
    viirs_nuage['code']=viirs['code']>=5
    return viirs_nuage




if __name__ == '__main__':
    print 'wololo'