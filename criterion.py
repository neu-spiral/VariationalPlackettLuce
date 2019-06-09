#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:15:50 2019

@author: yuanneu
"""

import sys
#sys.path.insert(0,'shared/centos7/anaconda3/3.6/lib/python3.6/site-packages')
import time
import numpy as np
import math
from scipy import special


import argparse
import scipy as sc
import random
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pickle


def Gauss(Mn,Sn,Msample):
    BetaSam=np.random.multivariate_normal(Mn,Sn,Msample)
    return BetaSam

def count_k(Pdict, Kt, real_rank):
    prob_rank_key=[]
    for key, value in sorted(Pdict.items(), key=lambda item:item[1]):
        prob_rank_key.append(key)
    prob_rank_key=np.array(prob_rank_key)
    #rank=np.argsort(np.array(list(Pdict.values()))) 
    up_k_item=list(prob_rank_key[-Kt:])
    low_k_item=list(prob_rank_key[:Kt])
    g_up=sum([real_rank[key] for key in up_k_item])
    min_g_low=sum([real_rank[key] for key in low_k_item])
    return g_up, Kt-min_g_low   

             
                
class Count(): 
    def __init__(self, Label):
        """Initialize by providing submodular function as well as universe Omega over which elements are selected. 
        """
        self.L=Label
    def initialize(self):
        
        self.count={}
        self.num={}
        self.up={}
        self.low={}
        for key in self.L:             
            pi,pj=key 
            try: 
                self.num[pi]+=1
                self.count[pi].append(pj)
            except:
                self.num[pi]=1
                self.count[pi]=[pj]
                
            if self.L[(pi,pj)]==1:            
                try:
                    self.up[pi]+=1
                except:
                    self.up[pi]=1
            else:            
                try:
                    self.low[pi]+=1
                except:
                    self.low[pi]=1
    def rank(self):
        self.rank={}
        #self.rank_low={}
        for key in self.num:
            try:
                self.rank[key]=float(self.up[key])/float(self.num[key])
            except:
                self.rank[key]=0.0
            #self.rank_low[key]=float(self.low[key])/float(self.num[key])
    
    def transfer(self, P1, P2):
        self.Pmean1={}
        self.Pmean2={}
        for key in self.num:
            item_neigh=self.count[key]
            self.Pmean1[key]=np.mean(P1[key,item_neigh])
            self.Pmean2[key]=np.mean(P2[key,item_neigh])
    
    def top_low_k(self, K):
        
        real_score_up_to_low=np.sort(np.array(list(self.rank.values())))
        #sort_low=np.sort(np.array(list(self.rank_low)))
        
        top_k_optimal=np.sum(real_score_up_to_low[-K:])
        low_k_optimal=K-np.sum(real_score_up_to_low[:K])
        
        """
        rank_value1=[]
        for key, value in sorted(self.Pmean1.items(), key=lambda item:item[1]):
            rank_value1.append(value)
        
        rank_value2=[]
        for key, value in sorted(self.Pmean2.items(), key=lambda item:item[1]):
            rank_value2.append(value)
        """
        g_up1,g_low1=count_k(self.Pmean1, K, self.rank)
        g_up2,g_low2=count_k(self.Pmean2, K, self.rank)
        #g_up1,g_low1=count_k(self.Pmean1, K)
        #g_up2,g_low2=count_k(self.Pmean2, K)
        

       
        return g_up1/top_k_optimal,g_low1/low_k_optimal,g_up2/top_k_optimal,g_low2/low_k_optimal
        

def Dcg(test,Pm):
    Nsize=len(test)
    rank_num=len(test[0])
    best_value=0
    for ki in range(rank_num):
        best_value+=(rank_num-1-ki)/np.log(ki+2)
    Dvalue=0
    for bt in test:
        score=Pm[bt]
        sort=np.argsort(-score)
        dvalue=0
        for pi in range(rank_num):
            dvalue+=(rank_num-1-pi)/np.log(sort[pi]+2)
        Dvalue+=dvalue
    return Dvalue/Nsize/best_value

