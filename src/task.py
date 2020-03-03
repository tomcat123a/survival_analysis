# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:06:12 2020

@author: Administrator
"""
import subprocess
import pandas as pd
import numpy as np
import os
import torch
from torch.nn import Conv1d, ModuleList ,BatchNorm1d,Sequential,\
AdaptiveAvgPool1d,Linear,MSELoss,LSTM,GRU,MaxPool1d,AdaptiveMaxPool1d,AvgPool1d
 
#import torch.nn.SyncBatchNorm as BatchNorm1d 
from torch.nn import  LeakyReLU,ReLU,Dropout




subprocess.call('pwd',shell=True)
subprocess.check_output('dir',shell=True)
def generate_x(n,p,distribution_type,random_seed ):
    if distribution_type=='gaussian':
        return np.random.normal(0,1,n*p).reshape((n,p))
    if distribution_type=='uniform':
        return np.random.uniform(0,1,n).reshape((n,p))
def simulatedata(n=10,p=10,distribution_type='gaussian',time_distribution='weibull',\
                 la=2,al=1,miu=2,true_f=lambda x: 1.2*x[0]+2*x[1]*x[2]**2,cencorrate=0.5,random_seed=0):
    assert time_distribution  in ['weibull','gompez']
    assert la>0 and miu>0
    
    '''
    n:samples
    p:dimensions
    true_beta:true effective features' coefficients
    noise,gaussian noise added
    al<0
    Gompez hazard  decrease
    al>0 Gompez increase
    0<la<1 weilbull hazard decreases
    0<la<1 weilbull hazard increase
    return x,t,delta(x.shape=(n,p),y.shape=(p,1),delta.shape=(p,1))
    '''
    distribution_type='gaussian'
    n=10
    p=10
    la=1
    miu=0.5
    true_f=lambda x:x[1]*x[2]**2-4*x[4]
    random_seed=0
    np.random.seed(random_seed)
    X=generate_x(n,p,distribution_type,random_seed=random_seed)
    U=np.random.uniform(0.001,0.999, n)
    timelist=[]
    for i in range(n):
        u=U[i]
        sample_contrib_term=np.exp(true_f(X[i]))
        if time_distribution=='weibull':
            #hazard(t|contrib_term)=la*contrib_term*t**(miu-1)
            generated_time=( -np.log(u)/( la*sample_contrib_term) )**(1/miu) 
        if time_distribution=='gompez':
            #hazard(t|contrib_term)=la*contrib_term* exp(al*t)
            generated_time=1/al*np.log( -al/la*np.log(u)*sample_contrib_term+1)
        if time_distribution=='abssin':
            #hazard(t|contrib_term)=la*abs(sin(t))
            generated_time=1/al*np.log( -al/la*np.log(u)*sample_contrib_term+1)
        timelist.append( generated_time )
    time_raw=np.array(timelist)
    cencortime=np.random.exponential(1/cencorrate,n)
    stack_raw_censor_time=np.stack((time_raw,cencortime))
    time_result=np.min(stack_raw_censor_time ,axis=0) 
    censor_result=np.argmax(stack_raw_censor_time,axis=0) 
    censor_rate=1-censor_result.mean( )
    #censor_result[0] ==1 death ==0  censorde
    return X,np.array(time_result),np.array(censor_result),censor_rate

def splitdata(X,t,censor,training_percent,val_percent):
    N=X.shape[0]
    train_n=int(N*training_percent)
    val_n=int(N*val_percent)
     
    return X[:train_n],t[:train_n],censor[:train_n],X[train_n:train_n+val_n],\
    t[train_n:train_n+val_n],censor[train_n:train_n+val_n],\
    X[train_n+val_n:],t[train_n+val_n:],censor[train_n+val_n:]
    
    '''
    split data into training,val,test
    '''
     
 


class LinearBlock(torch.nn.Module):
        def __init__(self, in_channel,out_channel,bn,dr_p ):
            super(LinearBlock, self).__init__()
             
             
            mylist=ModuleList()
            mylist.append(Linear( in_channel,out_channel))
            if bn==1:
                mylist.append(BatchNorm1d(out_channel) )
            mylist.append(ReLU())
            if dr_p>0:
                mylist.append(Dropout(dr_p) )
             
            self.block= Sequential(mylist) 
             
             
        def forward(self, x):
            
            return  self.block(x)
    

class Autosurv(torch.nn.Module):
    def __init__(self, encoder_hidden_size, bn,dr_p,num_out_layers,cat_type,lambda1,dropout_p,noise ):
        super(Autosurv, self).__init__()
        self.encoder=ModuleList()
        self.decoder=ModuleList()
        self.n_encoders=len(encoder_hidden_size)
        for i in range(len(encoder_hidden_size)):
            self.encoder.append( LinearBlock(encoder_hidden_size[i],encoder_hidden_size[i+1]) )
             
        self.cat_type=cat_type   
        decoder_hidden_size=encoder_hidden_size[::-1]
        if self.cat_type=='add':
            for i in range(len(decoder_hidden_size)):
                self.decoder.append( LinearBlock(decoder_hidden_size[i],decoder_hidden_size[i+1]) )
        if self.cat_type in ['cat','null']:
            for i in range(len(decoder_hidden_size)):
                self.decoder.append( LinearBlock(2*decoder_hidden_size[i],decoder_hidden_size[i+1]) )
        def forward(self, x):
            stored=[]
            x0=x
            for i in range(self.n_encoders):
                x=self.encoder(x)
                stored.append(x)
             
            if self.cat_type=='add':
                for i in range(self.n_encoders):
                    x=self.decoder(x+stored[-i])
            if self.cat_type=='cat':
                for i in range(self.n_encoders):
                    x=self.decoder(torch.cat(x,stored[-i],dim=1))
            if self.cat_type=='null':
                for i in range(self.n_encoders):
                    x=self.decoder(x)
            return  x0,stored



def order_make_Rlist(X,t,censor):
    n=X.shape[0]
    neworder=np.argsort(t)
    X=X[neworder]
    t=t[neworder]
    censor=censor[neworder]
    death_idx=np.where(censor==1)
    Rlist=[]
    for j in death_idx:
        Rlist.append(list(range(j,n)))
    return X,t,censor, Rlist
    
    


def build_autosurv():
    
    '''
    return a model,fully connected
    encoder, hidden units = encoder_hidden_size
    ex,encoder_hidden_size=[200,100,50]
    '''
    



class Autosurv(torch.nn.Module):
        def __init__(self, in_channel,out_channel,bn,dr_p ):
            super(Autosurv, self).__init__()
             
             
            mylist=ModuleList()
            mylist.append(Linear( in_channel,out_channel))
            if bn==1:
                mylist.append(BatchNorm1d(out_channel) )
            mylist.append(ReLU())
            if dr_p>0:
                mylist.append(Dropout(dr_p) )
             
            self.block= Sequential(mylist) 
             
             
        def forward(self, x):
            
            return  self.block(x)    


def training(x, t,delta,model):
    '''
    given x,t,delta,train a model
    '''
    
def adjusthyper(x, t,delta,model):
    '''
    given x,t,delta,train a model
    '''
    
    
def evaluate(x, t,delta,model):
    '''
    given x,t,delta,train a model
    '''
