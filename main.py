#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:59:16 2019

@author: yuanneu
"""

from functionpackage import *

import numpy as np
import math
import pickle
import sys,argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'The Parameter Estimation and Variationakl Inference Algorithm',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--N',help='The expert index')
    parser.add_argument('--fold',help="The fold number")
    parser.add_argument('--loopT',help="The inner loop iteration number")
    parser.add_argument('--output',help="The output name")
    args = parser.parse_args()







    cst=str('Net')+str(args.N)+'L'+str(args.fold)+'.p'
    file=open(cst,'rb')
    Ct=pickle.load(file)
    Xarray=Ct['Xarray']
    RankPlack=Ct['Plack']
    RankMul=Ct['Mul']
    
    
          
    Clist=[3**i for i in range(-20,10)]
    
    BetaMAPlist=[]
    BetaVIlist=[]
    Covlist=[]
    
    LowBoundList=[]
    for C_value in Clist:
        ####the C_value is the prior distribution's 
        length=np.shape(Xarray)[1]
        Mn,Sn,lowbound=EMPlackett(Xarray,RankPlack,C_value,args.loopT)
        # the function EMPlackett is the variational inference code to return mean covariance and lower bound
        LowBoundList.append(lowbound)
        Beta_MAP=MapEstimation(Xarray,RankMul,C_value)
        # the MapEstimation can return the parameter estimation given by MAP method
        BetaVIlist.append(Mn)
        BetaMAPlist.append(Beta_MAP)
        Covlist.append(Sn)
    
    Cp={}
    Cp['betamap']=BetaMAPlist
    Cp['betavi']=BetaVIlist  
    Cp['covvi']=Covlist
    Cp['lowbound']=LowBoundList
    csts='args.output'+str(args.N)+'Fold'+str(args.fold)+str('.p')
    pickle.dump(Cp,open(csts,"wb"))