# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 23:16:02 2021

@author: DimiP
"""
import numpy as np





class model():
    
    def __init__(self,y0=np.array([1e6-50,200,50,50,1,0,0])):
        self.y0=y0
        self.y=y0
        
    
    def transitions(self):
        pass
        
    
    
        
    
    





def transitions(y,params,**option):
    result=np.zeros(10)  
    
    if option["choix"] == 'D':
        # transisitons array ['S_E', 'E_I', 'E_C','I_H','H_ICU','ICU_D','C_R','I_R','H_R','ICU_R']
        result[0] =  (y[2]+y[3])*params['coeff_SE']*y[0]/(1e6-y[6])
        result[1] =  y[1]*params['coeff_Esortant']*(1-params['ratio_EC_Esortant'])
        result[2] = y[1]*params['coeff_Esortant']*params['ratio_EC_Esortant']
        result[3] = y[3]*params['coeff_Isortant']*params['ratio_IH_Isortant']
        result[4] = y[4]*params['coeff_Hsortant']*params['ratio_H_ICU_Hsortant']
        result[5] = y[5]*params['coeff_ICUsortant']*params['ratio_ICUD_ICUsortant']
        result[6] = y[2]*params['coeff_CR']
        result[7] = y[3]*params['coeff_Isortant']*(1-params['ratio_IH_Isortant'])
        result[8] = y[4]*params['coeff_Hsortant']*(1-params['ratio_ICUD_ICUsortant'])
        result[9] = y[5]*params['coeff_ICUsortant']*(1-params['ratio_ICUD_ICUsortant'])
    
    if option["choix"] =='P':
        result[0] =  (y[2]+y[3])*params['coeff_SE']*y[0]/(1e6-y[6])
        
    
    return result

#negative log likelihood
def LMAIN(data_obs,data_sim,p):
    
    comp1=binomloglike(data_obs[:,0],data_sim[0][:,2],np.divide(data_sim[2][:,1]*p[17],nozero(data_sim[0][:,2])))
    comp2=binomloglike(data_obs[:,1],data_sim[0][:,2],np.divide(data_sim[2][:,3]*p[10]*p[11],nozero(data_sim[0][:,2])))
    comp3=binomloglike(data_obs[:,2],data_sim[0][:,4],np.divide(data_sim[2][:,4]*p[12]*p[13],nozero(data_sim[0][:,4])))
    comp4=binomloglike(data_obs[:,3],data_sim[0][:,5],np.divide(data_sim[2][:,5]*p[15]*p[16],nozero(data_sim[0][:,5])))
    return comp1,comp2,comp3,comp4
