#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:12:12 2019

@author: yuanneu
"""

import numpy as np
import math
from scipy import special


def lamfunction(x):
    if -20<=x<=20:
        z=(math.exp(x)-1)/(4*x*(math.exp(x)+1))
    else:
        if x<=-20:
            z=-1/(4*x)
        else:
            z=1/(4*x)
    return z

def Norm_Fun(ArrayL):
    z=np.max(ArrayL)
    return ArrayL-z

def Log_Fun(ArrayL):
    ex=np.sum(np.exp(ArrayL-ArrayL[0]))
    return np.log(ex)

def MapEstimation(Xarray,Com_dict,Lam):
    ###### Xarray is the feature matrix
    ###### Com_dict is dictionary save the ranking index
    ###### Lam is the penalty variable
    ###### Rho is the ADMM parameter 
    dim=np.shape(Xarray)[1]
    Beta_old=np.zeros((dim))
    Run=True
    ddt1=0
    Loss_old=1e15
    while(Run):  
        First_D=0
        Second_D=0
        for key in Com_dict:
            comlist=Com_dict[key]
            X_need=Xarray[comlist,:]
            Exp_array=np.exp(Norm_Fun(np.dot(X_need,Beta_old)))
            Norm_Exp=Exp_array/(np.sum(Exp_array))
            Amatrix=np.diag(Norm_Exp)
            Amatrix=Amatrix-np.outer(Norm_Exp,Norm_Exp)
            f_beta_fir=np.dot(X_need.T,Norm_Exp)-X_need[0,:]
            f_beta_sec=np.dot(X_need.T,np.dot(Amatrix,X_need))
            First_D+=f_beta_fir
            Second_D+=f_beta_sec
        First_D+=Lam*Beta_old
        Second_D+=Lam*np.identity(dim)
        Second_inv=np.linalg.inv(Second_D)
        Loss_old=0        
        for key in Com_dict:
            comlist=Com_dict[key]
            Xcom=Xarray[comlist,:]
            Slist=np.dot(Xcom,Beta_old)
            Loss_old+=Log_Fun(Slist)
        Loss_old+=0.5*Lam*np.dot(Beta_old,Beta_old)
        Line=True
        armi=0
        cm_t=0.5
        alpha_t=0.4
        tao=0.5
        p_vector=-np.dot(Second_inv,First_D)
        t_value=-cm_t*np.dot(p_vector,First_D)
        while (Line):
            Loss_new=0
            Beta_new=Beta_old+alpha_t*tao**armi*p_vector
            armi+=1
            for key in Com_dict:
                comlist=Com_dict[key]
                Xcom=Xarray[comlist,:]
                Slist=np.dot(Xcom,Beta_new)
                Loss_new+=Log_Fun(Slist)
            Loss_new+=0.5*Lam*np.dot(Beta_new,Beta_new)
            if ((Loss_old-Loss_new)>=(alpha_t*tao**armi*t_value)):
                Line=False
            else:
                pass
        Loss_new=0
        for key in Com_dict:
            comlist=Com_dict[key]
            Xcom=Xarray[comlist,:]
            Slist=np.dot(Xcom,Beta_new)
            Loss_new+=Log_Fun(Slist)
        Loss_new+=0.5*Lam*np.dot(Beta_new,Beta_new)
        Beta_old=Beta_new.copy()
        if Loss_new-Loss_old>=-1e-9:
            Run=False
        else:
            pass
        ddt1+=1
        if ddt1>=200:
            Run=False
        else:
            pass
    return Beta_old


def XGenerate(Xarray,Rank):
    Xdict={}
    Mvalue=len(Rank)
    for m in range(1,Mvalue+1):
        rank=Rank[m]
        Xa=Xarray[rank,:]
        Xdict[m]=Xa
    return Xdict,Mvalue

def MSInitial(Xdict,Mvalue):
    Xi={}
    Alpha={}
    for m in range(1,Mvalue+1):
        Xa=Xdict[m]
        length=np.shape(Xa)[0]
        Xi[(m,0)]=1
        for t in range(1,length-1):
            for r in range(t,length+1):
                Xi[(m,t,r)]=1
            Alpha[(m,t)]=0  
    return Xi,Alpha


def logT(x):
    y=np.log(1.0/(1+np.exp(-x)))
    return y

def AlterXA(Sn,Mn,Num,Xi,Alpha,Xdict,Mvalue,Lamfun):
    Sv={}
    MT_X={}
    for m in range(1,Mvalue+1):        
        Xa=Xdict[m]
        length=np.shape(Xa)[0]
        mT_x=np.dot(Xa,Mn)
        MT_X[m]=mT_x
        delta_X=Xa[-2,:]-Xa[-1,:]
        Sv[(m,0)]=np.dot(delta_X,np.dot(Sn,delta_X)) 
        Xi[(m,0)]=math.sqrt(Sv[(m,0)]+(mT_x[-2]-mT_x[-1])**2)
        for mj in range(length):
            Sv[(m,mj+1)]=np.dot(Xa[mj,:],np.dot(Sn,Xa[mj,:]))             
    for m in range(1,Mvalue+1):
        mT_x=MT_X[m]
        length=len(mT_x)
        for t in range(1,length-1):
            for num in range(Num):
                for r in range(t,length+1):
                    xi=Sv[(m,r)]+(mT_x[r-1]-Alpha[(m,t)])**2
                    Xi[(m,t,r)]=math.sqrt(xi)
                alpha=(length-t-1)/4.0
                denom=0
                for r in range(t,length+1):
                    lam=Lamfun(Xi[(m,t,r)])
                    alpha+=lam*mT_x[r-1]
                    denom+=lam
                Alpha[(m,t)]=alpha/denom                               
    return Xi,Alpha

def RenewSM(Sn0,Mn0,Xi,Alpha,Xdict,Mvalue,Lamfun):
    invS0=np.linalg.pinv(Sn0)
    invS=invS0.copy()
    invSu=np.dot(invS0,Mn0)
    for m in range(1,Mvalue+1):
        Xa=Xdict[m]
        length=np.shape(Xa)[0]
        deltaX=Xa[-2,:]-Xa[-1,:]
        invS+=2*Lamfun(Xi[(m,0)])*np.outer(deltaX,deltaX)
        invSu+=0.5*(deltaX)
        for t in range(1,length-1):
            invSu+=Xa[t-1,:]
            for r in range(t,length+1):
                weight=Lamfun(Xi[(m,t,r)])
                invS+=2*weight*np.outer(Xa[r-1,:],Xa[r-1,:])
                at=2*weight*Alpha[(m,t)]-0.5
                invSu+=at*Xa[r-1,:]
    Sn=np.linalg.pinv(invS) 
    Mn=np.dot(Sn,invSu)    
    return Mn,Sn

def EMPlackett(Xarray,Rank,sigma0,IterNum):
    dim=np.shape(Xarray)[1]
    Mn0=np.zeros(dim)
    Sn0=1/sigma0*np.identity(dim)
    Xdict,Mvalue=XGenerate(Xarray,Rank)
    Xi,Alpha=MSInitial(Xdict,Mvalue)
    Value=True
    MnC=Mn0.copy()
    ddt=0
    L_list=[]
    while (Value):        
        Mn,Sn=RenewSM(Sn0,Mn0,Xi,Alpha,Xdict,Mvalue,lamfunction)
        if ddt>1:
            L_value=0
            for m in range(1,Mvalue+1):
                length=np.shape(Xdict[m])[0]
                xt0=Xi[(m,0)]
                L_value+=(logT(xt0)-0.5*xt0+lamfunction(xt0)*xt0**2)                   
                for t in range(1,length-1):
                    L_value+=((length-t-1)*Alpha[(m,t)]/2.0)
                    for r in range(t,length+1):
                        xt1=Xi[(m,t,r)]
                        L_value+=(logT(xt1)-0.5*xt1+lamfunction(xt1)*(xt1**2-(Alpha[(m,t)])**2))
            L_value+=(0.5*(np.linalg.slogdet(Sn)[1])-0.5*(np.linalg.slogdet(Sn0)[1]))  
            L_value+=(0.5*np.dot(Mn,np.dot(np.linalg.inv(Sn),Mn))-0.5*np.dot(Mn0,np.dot(np.linalg.inv(Sn0),Mn0)))
            L_list.append(L_value)        
        Xi,Alpha=AlterXA(Sn,Mn,IterNum,Xi,Alpha,Xdict,Mvalue,lamfunction)
        if (np.linalg.norm(Mn-MnC)<=1e-12):
            Value=False
        else:
            MnC=Mn.copy()          
            ddt+=1
            if ddt>=100:
                Value=False
            else:
                pass 
    return Mn,Sn,L_list


def EMupdateVariational(mu0,Sigma0,Xab,Yab):## the prior distribution N(mu0,sigma0),the initial value xi0 (Variational Inference)
# the given absolute feature Xab is NXd, Yab is the given label{0,+1}.
    length=len(Yab)
    xi=np.ones(length)/2
    Value=1
    invS0=np.linalg.inv(Sigma0)
    Featurebiase=0
    for unit in range(length):
        Featurebiase+=0.5*Yab[unit]*Xab[unit,:]
    Sigma=1*Sigma0
    Value=True
    ddp=0
    while(Value):
        Mapmu0=np.dot(invS0,mu0)
        mu=np.dot(Sigma,Mapmu0+Featurebiase)
        invS=invS0.copy()
        for unit in range(length):
            invS+=2*lamfunction(xi[unit])*np.outer(Xab[unit,:],Xab[unit,:])
        Sigma=np.linalg.pinv(invS)
        SigmaPxdot=Sigma+np.outer(mu,mu)
        xi_old=xi.copy()
        for unit in range(length):
            xi[unit]=np.sqrt(np.dot(Xab[unit,:],np.dot(SigmaPxdot,Xab[unit,:])))
        if np.linalg.norm((xi-xi_old))<=1e-8:
            Value=False
        else:
            if ddp>=200:
                Value=False
            else:
                ddp+=1
    return mu,Sigma


def logistic(x):
    y=1.0/(1+np.exp(-x))
    return y


def LogScore(Xarray,Mn,Sn,Msample):
    BetaSam=np.random.multivariate_normal(Mn,Sn,Msample)
    ScoreM=np.dot(Xarray,BetaSam.T)
    ScoreL=logistic(ScoreM) 
    meanS=np.mean(ScoreL,1)
    return meanS

def LogPS(xarray,Beta):
    score=logistic(np.dot(Beta,xarray))
    return score

def GaussianScore(x1,x2, y1, y2, Mu,Sigma):
    x12=x1-x2
    mu=np.dot(x12,Mu)
    sigma=np.sqrt(np.dot(np.dot(x12,Sigma),x12))
    z=mu/sigma
    value=0.5*special.erfc(z/math.sqrt(2))
    if y1>y2:
        score=1-value
    else:
        if y1<y2:
            score=value
        else:
            score=1-abs(value-1)
    return score
