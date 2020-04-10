# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 23:02:52 2020

@author: Administrator
"""

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
import time
from torch.nn import Conv1d, ModuleList ,BatchNorm1d,Sequential,\
AdaptiveAvgPool1d,Linear,MSELoss,LSTM,GRU,MaxPool1d,AdaptiveMaxPool1d,AvgPool1d
import xgboost as xgb
#import torch.nn.SyncBatchNorm as BatchNorm1d 
from torch.nn import  LeakyReLU,ReLU,Dropout

import argparse
Parser=argparse.ArgumentParser()

Parser.add_argument("--id", type=int,default=1,help='saveid')
Parser.add_argument("--d", type=int,default=1000,help='dimension')
parser=Parser.parse_args()


a=surv_experiment()

class surv_experiment():
    def __init__(self,simu=True,cancer='BRCA',n=100,p=10,realp=30,distribution_type='gaussian',\
                 time_distribution='weibull',\
                 la=0.8,al=1,miu=2,cencorrate=0.5,sigma=0.2,random_seed=0,\
                 training_percent=0.6,val_percent=0.2,save=False,saveid=1):
        if simu==True:
            self.full_x,self.full_t,self.full_c,self.c_rate=self.simulatedata(n=n,p=p,realp=realp,distribution_type=distribution_type,\
    time_distribution=time_distribution,\
    la=la,al=al,miu=miu,cencorrate=cencorrate,sigma=sigma,random_seed=random_seed)
            print('censorsamples_rate={}'.format(self.c_rate))
            self.train_x,self.train_t,self.train_c,\
            self.val_x,self.val_t,self.val_c,\
            self.test_x,self.test_t,self.test_c=self.splitdata(self.full_x,\
            self.full_t,self.full_c,\
            training_percent=training_percent,val_percent=val_percent,save=save,saveid=saveid)
            self.features=p
        else:
            self.splitcv(x,t,c,training_percent,nfolds)
        self.seed=random_seed
        self.maxepochs=1000
        self.lr=0.002
        self.verbose=2
    def generate_x(self,n,p,realp,distribution_type,random_seed ,sigma,case=0):
        '''case:0 original add extra dimension noise
           case:1 original linear transformed add extra dimension noise
           case:2 original nonlinear transformed add nonlinear dimension
           X core signal 
           d after adding extra features or transformation
        '''
        X=np.random.uniform(0,1,n*realp).reshape((n,realp)).astype(np.float32,casting='same_kind')
        if case==0:
            extra_x=np.random.uniform(0,1,n*(p-realp)).reshape((n,(p-realp))).astype(np.float32,casting='same_kind')
            d=np.concatenate((X,extra_x),axis=0)
        if case==1:
            assert p/2==int(p/2)
            generate_model=LinearBlock( realp,p/2, bn=0,dr_p=0,no_tail=True,bias=False)
            extra_x=np.random.uniform(0,1,n*(p/2)).reshape((n,(p/2))).astype(np.float32,casting='same_kind')
            X=generate_model(torch.from_numpy(X)).data.numpy()
            d=np.concatenate((X,extra_x),axis=0)
        
        if case==2:
            
            generate_model=Fc( encoder_hidden_size=[realp,int(np.sqrt(p/2)),p/2], bn=0,dr_p=0)
            extra_x=np.random.uniform(0,1,n*(p/2)).reshape((n,(p/2))).astype(np.float32,casting='same_kind')
            X=generate_model(torch.from_numpy(X)).data.numpy()
            d=np.concatenate((X,extra_x),axis=0)
        if distribution_type=='gaussian':
            return d+sigma*np.random.normal(0,1,n*p).reshape((n,p)),X
        
        if distribution_type=='uniform':
            return d+sigma*np.random.uniform(0,1,n*p).reshape((n,p)),X
         
              
    def simulatedata(self,n=10,p=10,realp=30,distribution_type='gaussian',time_distribution='weibull',\
                     la=0.8,al=1,miu=2,cencorrate=0.5,sigma=0.2,random_seed=0,effect_case=0,x_case=0):
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
        '''
        n=10
        p=10
        la=1
        miu=0.5
        true_f=lambda x:x[1]*x[2]**2-4*x[4]
        random_seed=0
        '''
        np.random.seed(random_seed)
        X,essential_X=self.generate_x(n,p,realp,distribution_type,random_seed=random_seed,sigma=sigma,case=x_case)
        U=np.random.uniform(0.01,0.99, n)
        if effect_case==0:
            true_f=[realp,1]
            generate_model=LinearBlock(true_f[0],true_f[1],bn=0,dr_p=0,no_tail=True,bias=False)
        if effect_case==1:
            t_func=lambda x:np.sum( np.array([x[0]*x[1],\
    x[2]*x[3]**2,0.3*np.exp(x[4]),x[5]/(1+abs(x[6]-2*x[7])),\
    np.exp(-0.25*(x[8]+x[9])**2),-x[0]-x[3],x[2]-2*x[4],x[5]*x[6],x[7]**3,np.cos(x[8]) ]) )
        if effect_case==2:
            true_f=[realp,realp,realp,1]
            generate_model=Fc(encoder_hidden_size=true_f,bn=0,dr_p=0)
         
        timelist=[]
        for i in range(n):
            u=U[i]
            if case in [0,2]:
                sample_contrib_term=np.exp(generate_model(torch.from_numpy( essential_X[i]).unsqueeze(0)).cpu().data.numpy()[0] )
            if case in [1]:
                sample_contrib_term=np.exp(t_func( essential_X[i]))
            
            #print(sample_contrib_term)
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
    
    def to32_tensor(x):
        return torch.from_numpy(x.astype(np.float32)) 
    
    
         
    def splitdata(self,X,t,censor,training_percent,val_percent,save,saveid):
        N=X.shape[0]
        train_n=int(N*training_percent)
        val_n=int(N*val_percent)
        folder='../simulationdata/'
        if save==True:
            pd.DataFrame(X[:train_n]).to_csv(folder+'train_x_{}.csv'.format(saveid),index=False)
            pd.DataFrame(t[:train_n]).to_csv(folder+'train_t_{}.csv'.format(saveid),index=False)
            pd.DataFrame(censor[:train_n]).to_csv(folder+'train_c_{}.csv'.format(saveid),index=False)
            pd.DataFrame(X[train_n:train_n+val_n]).to_csv(folder+'val_x_{}.csv'.format(saveid),index=False)
            pd.DataFrame(t[train_n:train_n+val_n]).to_csv(folder+'val_t_{}.csv'.format(saveid),index=False)
            pd.DataFrame(censor[train_n:train_n+val_n]).to_csv(folder+'val_c_{}.csv'.format(saveid),index=False)
            pd.DataFrame(X[train_n:train_n+val_n]).to_csv(folder+'test_x_{}.csv'.format(saveid),index=False)
            pd.DataFrame(t[train_n+val_n:]).to_csv(folder+'test_t_{}.csv'.format(saveid),index=False)
            pd.DataFrame(censor[train_n+val_n:]).to_csv(folder+'test_c_{}.csv'.format(saveid),index=False)
        X,t,censor=to32_tensor(X),to32_tensor(t),to32_tensor(censor)     
        return X[:train_n],t[:train_n],censor[:train_n],X[train_n:train_n+val_n],\
        t[train_n:train_n+val_n],censor[train_n:train_n+val_n],\
        X[train_n+val_n:],t[train_n+val_n:],censor[train_n+val_n:]
        
        '''
        split data into training,val,test
        '''            
                
    def test_model(self,name='as',model,train_x,train_t,train_c,val_x,val_t,val_c,\
    test_x,test_t,test_c,epochs=1000):
         
        tr_x,tr_t,tr_delta,tr_Rlist=order_make_Rlist(train_x,train_t,train_c)
        va_x,va_t,va_delta,va_Rlist=order_make_Rlist(val_x,val_t,val_c)
        test_x,test_t,test_delta,test_Rlist=order_make_Rlist(test_x,test_t,test_c)
        if name in ['as','fc','xgb']:
            tpe_algo = tpe.suggest
            tpe_trials = Trials()
            if name=='as':
                space = [hp.quniform('encoder_depth', 1, 3,1),hp.quniform('bottleneck', 10, 30,1),\
                hp.quniform('fc_depth',1, 3,1),\
                         hp.loguniform('lambda1', 0.001, 10),hp.uniform('drop_out', 0.1, 0.5)]
                self.tr_x=tr_x
                self.tr_Rlist=tr_Rlist
                self.va_x=va_x
                self.va_t=va_t
                self.va_c=va_c
            if name=='fc':
                self.tr_x=tr_x
                self.tr_Rlist=tr_Rlist
                self.va_x=va_x
                self.va_t=va_t
                self.va_c=va_c
                space = [hp.quniform('encoder_depth', 1, 3,1),\
            hp.quniform('bottleneck', 10, 30,1),hp.uniform('drop_out', 0.1, 0.5)]
            if name=='xgb':
                self.tr_x=tr_x.data.numpy()
                self.tr_t=tr_t.data.numpy()
                self.tr_c=tr_c.data.numpy()
                self.va_x=va_x.data.numpy()
                self.va_t=va_t.data.numpy()
                self.va_c=va_c.data.numpy()
                space = [hp.quniform('max_depth',1,5,1),\
            hp.quniform('n_estimator', 10,1000,1)]
            self.name=name
            obj=self.training_mod
            
            tpe_best = fmin(fn=obj, space=space, 
                            algo=tpe_algo, trials=tpe_trials, 
                            max_evals=15,rstate=np.random.RandomState(4))
            if name=='xgb':
                test_x=test_x.data.numpy()
                test_t=test_t.data.numpy()
                test_c=test_c.data.numpy()
            if name in ['glmnet']:
                pass
        if name in ['xgboost']:
            pass
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        _,model=self.training_mod(**tpe_best)
        loss,cidx,marker=self.predict(x=test_x, t=test_t,delta=test_c,\
    model=model,name=name)
        self.model_out_info[name]={'loss':loss,'cindex':cidx,'marker':marker,\
                           'model':model}
        
        return self.model_out_info
           
    def init_model_list(self):
        pass
                    
    def predict(self,x, t,delta,model,verbose,name):
        if name in ['as','fc']:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
            '''
            given x,t,delta,train a model
            '''
            x,t,delta,Rlist=order_make_Rlist(x,t,delta)
              
            model=model.eval()
            model=model.to(device)
             
            t0=time.time()
             
            x=x.to(device)
            if name =='as':
                _,cox_out=model(x)
            if name =='fc':
                cox_out=model(x)
            c_idx=cindex(cox_out.cpu().data.numpy(),Rlist) 
            neg_log_partial_likelihood=coxloss(Rlist).to(device)
            loss_val=neg_log_partial_likelihood(cox_out).cpu().data.numpy()[0]
            
            t1=time.time()-t0
            if verbose==2:
                print('eval {} takes {}s cindex={}'.format(t1,name,c_idx))
            return loss_val,c_idx,cox_out.cpu().data.numpy()    
        if name in ['xgb']:
            x,t,delta,Rlist=order_make_Rlist(x,t,delta)
            marker=gbm.predict(x).reshape(-1)
            c_idx=cindex(marker,Rlist) 
            return 0,c_idx,marker                  
         
    def bo_obj_fc(self,depth,shrink,lambda1,drop_out):
        pass
    
    '''
            param[0] encoder depth
            param[1] encoder bottleneck
            param[2] fc length
    '''
    def training_mod(self,param ):
        #background variabls,outside this function
        #self.tr_x,self.tr_Rlist,self.val_x,self.val_Rlist
        lr=self.lr
        epochs=self.max_epochs
        verbose=self.verbose
        device=self.device
        name=self.name
        
       
        optimizer=torch.optim.Adam(model.parameters(), lr=lr,amsgrad=True) 
        
        if name in ['as','fc']:
            x=self.tr_x.to(device)
            hidden_size=self.make_hidden_list(self.features,param[0],param[1])
            if name=='as':
                param_dict={'encoder_hidden_size':hidden_size, \
       'cox_hidden_size':[param[1]]*param[2]+[1],'bn':1,'dr_p':param[4],
       'num_out_layers':1,\
       'cat_type':'null','lambda1':param[3],'bn_cox':1,'dr_p_cox':param[4],'noise':None}
                model=Autosurv(**param_dict) 
            if name=='fc':
                param_dict={'encoder_hidden_size':hidden_size,'bn':1,'dr_p':param[2]}
                model=Fc(**param_dict) 
            model=model.train()
            model=model.to(device)
            for iteration in range(epoch):
                t0=time.time()
                optimizer.zero_grad()
                #x=x.to(device)
                neg_log_partial_likelihood=coxloss(self.tr_Rlist).to(device) 
                if name=='as':
                    x_tilt,cox_out=model(x)
                    z=neg_log_partial_likelihood(cox_out).to(device)
                    loss = param_dict['lambda1']*MSELoss(reduction='mean')(x_tilt,x) +z
                if name=='fc':
                    cox_out=model(x)
                    loss =neg_log_partial_likelihood(cox_out).to(device)
                #loss_list.append(loss.cpu().data.numpy()[0])
                loss.backward()
                optimizer.step() 
                #evaluation on validation dataset
                if self.earlystop==True:
                    loss,cidx,marker=self.predict(x=self.va_x, t=self.va_t,delta=self.va_c,\
                    model=model,verbose=1,name=name)
                else:
                    loss,cidx,marker=self.predict(x=self.tr_x, t=self.tr_t,delta=self.tr_c,\
                    model=model,verbose=1,name=name)
                loss_hist.append(loss.cpu().data.numpy())
                if len(loss_hist)>4 and loss_hist[-1]>loss_high[-2] and loss_hist[-2]>loss_hist[-3]:
                    break
            if self.earlystop==False:
                loss,cidx,marker=self.predict(x=self.va_x, t=self.va_t,delta=self.va_c,\
                    model=model,verbose=1,name=name)
            return cidx,model
        if name in ['xgb']:
            xgb_tr_x=self.tr_x.data.numpy() 
             
            xgb_tr_t=np.array( [self.tr_t.data.numpy()[idx] if self.tr_c.data.numpy()[idx]==1 \
            else -self.tr_t.data.numpy()[idx] \
    for idx in range(len(self.tr_t.data.numpy()))] )
            xgb_va_x=self.tr_x.data.numpy()
            gbm=xgb.XGBRegressor(objective='survival:cox',booster='gbtree',max_depth=params[0],\
        n_estimators=params[1]).fit(xgb_tr_x,xgb_tr_t)
            loss,cidx,marker=self.predict(x=xgb_va_x.data.numpy(), t=self.va_t.data.numpy(),delta=self.va_c.data.numpy(),\
                model=gbm,verbose=1,name=name)
        return cidx,model   
    def save_model_out_info(self,folder):
        pass
    def make_hidden_list(self,in_size,depth,bottleneck_size):
        step= ( np.log(bottleneck_size)-np.log(in_size) )/depth
        return list( np.exp( np.arange(np.log(in_size),np.log(bottleneck_size)+0.1*step,step) ).astype(int))
                 
def generate_x(n,p,realp,distribution_type,random_seed ,sigma):
    generate_model=Fc( encoder_hidden_size=[realp,int(np.sqrt(p)),p], bn=0,dr_p=0)
    X=np.random.uniform(0,1,n*realp).reshape((n,realp)).astype(np.float32,casting='same_kind')
    d=generate_model(torch.from_numpy(X)).data.numpy()
    if distribution_type=='gaussian':
        return d+sigma*np.random.normal(0,1,n*p).reshape((n,p)),X
    
    if distribution_type=='uniform':
        return d+sigma*np.random.uniform(0,1,n*p).reshape((n,p)),X
     
        
def simulatedata(n=10,p=10,realp=30,distribution_type='gaussian',time_distribution='weibull',\
                 la=0.8,al=1,miu=2,cencorrate=0.5,sigma=0.2,random_seed=0):
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
    '''
    n=10
    p=10
    la=1
    miu=0.5
    true_f=lambda x:x[1]*x[2]**2-4*x[4]
    random_seed=0
    '''
    np.random.seed(random_seed)
    X,essential_X=generate_x(n,p,realp,distribution_type,random_seed=random_seed,sigma=sigma)
    U=np.random.uniform(0.01,0.99, n)
    true_f=[realp,realp,1]
    generate_model=Fc(encoder_hidden_size=true_f,bn=0,dr_p=0)
     
    timelist=[]
    for i in range(n):
        u=U[i]
        sample_contrib_term=np.exp(generate_model(torch.from_numpy( essential_X[i]).unsqueeze(0)).cpu().data.numpy()[0] )
        #print(sample_contrib_term)
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

def to32_tensor(x):
    return torch.from_numpy(x.astype(np.float32)) 


     
def splitdata(X,t,censor,training_percent,val_percent,save,saveid):
    N=X.shape[0]
    train_n=int(N*training_percent)
    val_n=int(N*val_percent)
    folder='../simulationdata/'
    if save==True:
        pd.DataFrame(X[:train_n]).to_csv(folder+'train_x_{}.csv'.format(saveid),index=False)
        pd.DataFrame(t[:train_n]).to_csv(folder+'train_t_{}.csv'.format(saveid),index=False)
        pd.DataFrame(censor[:train_n]).to_csv(folder+'train_c_{}.csv'.format(saveid),index=False)
        pd.DataFrame(X[train_n:train_n+val_n]).to_csv(folder+'val_x_{}.csv'.format(saveid),index=False)
        pd.DataFrame(t[train_n:train_n+val_n]).to_csv(folder+'val_t_{}.csv'.format(saveid),index=False)
        pd.DataFrame(censor[train_n:train_n+val_n]).to_csv(folder+'val_c_{}.csv'.format(saveid),index=False)
        pd.DataFrame(X[train_n:train_n+val_n]).to_csv(folder+'test_x_{}.csv'.format(saveid),index=False)
        pd.DataFrame(t[train_n+val_n:]).to_csv(folder+'test_t_{}.csv'.format(saveid),index=False)
        pd.DataFrame(censor[train_n+val_n:]).to_csv(folder+'test_c_{}.csv'.format(saveid),index=False)
    X,t,censor=to32_tensor(X),to32_tensor(t),to32_tensor(censor)     
    return X[:train_n],t[:train_n],censor[:train_n],X[train_n:train_n+val_n],\
    t[train_n:train_n+val_n],censor[train_n:train_n+val_n],\
    X[train_n+val_n:],t[train_n+val_n:],censor[train_n+val_n:]
    
    '''
    split data into training,val,test
    '''
      


class LinearBlock(torch.nn.Module):
    def __init__(self, in_channel,out_channel,bn,dr_p ,no_tail=False,bias=True):
        super(LinearBlock, self).__init__()
         
         
        mylist=ModuleList()
        mylist.append(Linear( in_channel,out_channel,bias=bias))
        if no_tail==False:
            if bn==1:
                mylist.append(BatchNorm1d(out_channel) )
            
            mylist.append(ReLU())
            if dr_p>0:
                mylist.append(Dropout(dr_p) )
         
        self.block= Sequential(*mylist) 
         
         
    def forward(self, x):
        
        return  self.block(x)


class AutoEncoder(torch.nn.Module):
    def __init__(self, encoder_hidden_size, bn,dr_p,cat_type,lambda1):
        super(AutoEncoder, self).__init__()
        self.encoder=ModuleList()
        self.decoder=ModuleList()
        self.n_encoders=len(encoder_hidden_size)-1
        for i in range(self.n_encoders):
            self.encoder.append( LinearBlock(encoder_hidden_size[i],encoder_hidden_size[i+1], bn,dr_p) )
             
        self.cat_type=cat_type   
        decoder_hidden_size=encoder_hidden_size[::-1]
        if self.cat_type in ['add','null']:
            for i in range(self.n_encoders):
                self.decoder.append( LinearBlock(decoder_hidden_size[i],decoder_hidden_size[i+1], bn,dr_p,\
        no_tail=False if i<self.n_encoders-1 else True ) )
        
        if self.cat_type in ['cat']:
            for i in range(self.n_encoders):
                self.decoder.append( LinearBlock(2*decoder_hidden_size[i],decoder_hidden_size[i+1], bn,dr_p,\
               no_tail=False if i<self.n_encoders-1 else True ) )
        #self.encoder=Sequential(self.encoder)
        #self.decoder=Sequential(self.decoder)
    def forward(self, x):
         
        stored=[]
         
         
        for i in range(self.n_encoders):
             
            x=self.encoder[i](x)
            stored.append(x)
         
        if self.cat_type=='add':
            for i in range(self.n_encoders):
                x=self.decoder[i](x+stored[-i])
        if self.cat_type=='cat':
            for i in range(self.n_encoders):
                x=self.decoder[i](torch.cat(x,stored[-i],dim=1))
        if self.cat_type=='null':
            for i in range(self.n_encoders):
                x=self.decoder[i](x)
         
        return  x,stored

class Fc(torch.nn.Module):
    def __init__(self, encoder_hidden_size, bn,dr_p):
        super(Fc, self).__init__()
        self.encoder=ModuleList()
        
        self.n_encoders=len(encoder_hidden_size)-1 
        for i in range(self.n_encoders):
            self.encoder.append( LinearBlock(encoder_hidden_size[i],\
                                             encoder_hidden_size[i+1], bn,dr_p,\
    no_tail=False if encoder_hidden_size[i+1]!=1 else True) )
             
        self.full_model=Sequential(*self.encoder)
    def forward(self, x):
        
        return  self.full_model(x).squeeze(1)#batch


def order_make_Rlist(X,t,censor):
    n=X.shape[0]
    neworder=np.argsort(t)
    X=X[neworder]
    t=t[neworder]
    censor=censor[neworder]
    death_idx=np.where(censor==1)[0]
    Rlist=[]
    for j in death_idx:
        Rlist.append(list(range(j,n)))
    return X,t,censor, Rlist
    
    


def build_autosurv():
    pass
    '''
    return a model,fully connected
    encoder, hidden units = encoder_hidden_size
    ex,encoder_hidden_size=[200,100,50]
    '''
    



class Autosurv(torch.nn.Module):
    def __init__(self, encoder_hidden_size, cox_hidden_size,bn,dr_p,num_out_layers,\
   cat_type,lambda1,bn_cox,dr_p_cox,noise):
        super(Autosurv, self).__init__()
        self.auto_encoder=AutoEncoder(encoder_hidden_size, bn,dr_p,cat_type,lambda1)
        self.cox=ModuleList()
        self.num_out_layers=num_out_layers
        self.lambda1=lambda1
        if cox_hidden_size[-1]!=1:
            cox_hidden_size.append(1)
        for i in range(len(cox_hidden_size)):
            if i==0:
                self.cox.append( LinearBlock(sum(encoder_hidden_size[-num_out_layers:]),cox_hidden_size[i],bn_cox,dr_p_cox) )
            else:
                self.cox.append( LinearBlock(cox_hidden_size[i-1],cox_hidden_size[i],bn_cox,dr_p_cox) )
        #self.auto_encoder=Sequential(self.auto_encoder)    
        self.cox=Sequential(*self.cox)
    
    def forward(self, x):
        x_tilt,x_mid_list=self.auto_encoder(x)
        y = torch.cat( x_mid_list[-self.num_out_layers:] ,dim=1)
        cox_out = self.cox(y)
        cox_out=cox_out.squeeze(1)
        return  x_tilt,cox_out


def training_Autosurv(x, t,delta,model,lr,lambda1,epoch,verbose,name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
    '''
    given x,t,delta,train a model
    '''
    x,t,delta,Rlist=order_make_Rlist(x,t,delta)
    
    model=model.train()
    model=model.to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr,amsgrad=True) 
    loss_list=[]
    for iteration in range(epoch):
        t0=time.time()
        optimizer.zero_grad()
        x=x.to(device)
        neg_log_partial_likelihood=coxloss(Rlist).to(device) 
        if name=='auto_surv':
            x_tilt,cox_out=model(x)
            
            z=neg_log_partial_likelihood(cox_out).to(device)
             
            loss = MSELoss(reduction='mean')(x_tilt,x) +lambda1*z
        if name=='fc':
            cox_out=model(x)
             
            loss =neg_log_partial_likelihood(cox_out).to(device)
        loss_list.append(loss.cpu().data.numpy()[0])
        loss.backward()
        optimizer.step() 
        t1=time.time()-t0
        if verbose==2:
            print('traing takes {}s,loss={}'.format(t1,loss.cpu().data.numpy()[0]))
        if len(loss_list)>3:
            if loss_list[-1]>loss_list[-2] and loss_list[-2]>loss_list[-3]:
                print('{} converged'.format(name))
                
                return 1
    return  0
            

def training_Fc(x, t,delta,model,lr,epoch,verbose,name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
    '''
    given x,t,delta,train a model
    '''
    x,t,delta,Rlist=order_make_Rlist(x,t,delta)
    
    model=model.train()
    model=model.to(device)
    loss_list=[]
    optimizer=torch.optim.Adam(model.parameters(), lr=lr,amsgrad=True) 
    for iteration in range(epoch):
        t0=time.time()
        optimizer.zero_grad()
        x=x.to(device)
        cox_out=model(x)
        neg_log_partial_likelihood=coxloss(Rlist)
        loss =neg_log_partial_likelihood(cox_out).to(device)
        loss.backward()
        optimizer.step() 
        t1=time.time()-t0
        if verbose==2:
            print('traing takes {}s,loss={}'.format(t1,loss.cpu().data.numpy()[0]))
        if len(loss_list)>3:
            if loss_list[-1]>loss_list[-2] and loss_list[-2]>loss_list[-3]:
                print('{} converged'.format(name))
                
                return 1
    return  0
            
def predict(x, t,delta,model,verbose,name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
    '''
    given x,t,delta,train a model
    '''
    x,t,delta,Rlist=order_make_Rlist(x,t,delta)
      
    model=model.eval()
    model=model.to(device)
     
    t0=time.time()
     
    x=x.to(device)
    if name =='as':
        _,cox_out=model(x)
    if name =='fc':
        cox_out=model(x)
    c_idx=cindex(cox_out.cpu().data.numpy(),Rlist) 
    neg_log_partial_likelihood=coxloss(Rlist).to(device)
    loss_val=neg_log_partial_likelihood(cox_out).cpu().data.numpy()[0]
    
    t1=time.time()-t0
    if verbose==2:
        print('eval {} takes {}s cindex={}'.format(t1,name,c_idx))
    return loss_val,c_idx,cox_out.cpu().data.numpy()
class coxloss(torch.nn.Module):  #resnet
    def __init__(self,Rlist):
        #c=1 uncensored,dead
        #c=0 censored,alive
        super(coxloss, self).__init__()
        self.Rlist=Rlist
    def forward(self,pred): 
#pred.size()==batch,if pred.size()==batch,1,then it should be converted by pred=pred.squueze(1)
        z=torch.zeros(1).to(pred.device)
        for subset in self.Rlist:
            z=z+pred[subset[0]]-torch.log(\
   torch.sum(torch.exp(pred[subset]),dim=0))
        return -z 
 
 
def cindex(y_pred,Rlist):
    cor=0
    total=0
    for subset in Rlist:
        cor+=sum( y_pred[subset[0]]>y_pred[subset[1:]] )
        total+=len(subset)-1
     
    return cor/total    
    
def adjusthyper(x, t,delta,model_list):
    
    '''
    given x,t,delta,select model on eval dataset
    '''
def cv(x, t,delta,model,nfold):
    '''
    given x,t,delta,train a model,cross validation to select hyper parameter
    '''    
    
def comparemodel(case=1,features=20,realp=10,samples_n=100,saveid=1):
    '''
    given x,t,delta,evalute to get loss value or cindex
    
    '''
    if case==1:
        features=features
        torch.manual_seed(saveid)
        x,y,t,rate=simulatedata(samples_n,features,realp=realp,cencorrate=0.5,random_seed=saveid,sigma=1)
        print('{} censored samples'.format(rate))
        train_x,train_y,train_c,val_x,val_y,val_c,\
        test_x,test_y,test_c=splitdata(x,y,t,0.6,0.2,save=False,saveid=saveid)
        auto_surv_list=[]
        fc_list=[]
        as_record=[]
        fc_record=[]
        for bn in [1]:
            for dr_p in [0,0.2,0.5]:
                for hidden_size in [[features,realp],[features,int(0.4*features),realp],[features,int(0.6*features),int(0.3*features),realp]]:
                    param={'encoder_hidden_size':hidden_size+[1],'bn':bn,'dr_p':dr_p}
                    fc_record.append(param)
                    fc_list.append( Fc(**param))
                    for lambda1 in [ 0.001, 0.01,0.1,1, 10]:
                        
                        param={'encoder_hidden_size':hidden_size, 
                                'cox_hidden_size':hidden_size+[1],'bn':bn,'dr_p':dr_p,
                               'num_out_layers':1,
       'cat_type':'null','lambda1':lambda1,'bn_cox':bn,'dr_p_cox':dr_p,'noise':None}
                        auto_surv_list.append( Autosurv(**param) )
                        as_record.append(param)
                     
        as_cindex=[]
         
        for ele in auto_surv_list:
            converged=training_Autosurv(x=train_x, t=train_y,delta=train_c,model=ele,\
            lr=0.01,lambda1=ele.lambda1,epoch=1000,verbose=2,name='auto_surv')
            loss,cidx,marker=predict(x=val_x, t=val_y,delta=val_c,model=ele,verbose=1,name='auto_surv')
            if converged==1:
                as_cindex.append(cidx)
                continue
            if converged!=1:
                print('as not converge')
                as_cindex.append(cidx)
             
        fc_cindex=[]
         
        for ele in fc_list:
            converged=training_Autosurv(x=train_x, t=train_y,delta=train_c,model=ele,\
            lr=0.01,lambda1=1,epoch=1000,verbose=2,name='fc')
            loss,cidx,marker=predict(x=val_x, t=val_y,delta=val_c,model=ele,verbose=1,name='fc')
            if converged==1:
                fc_cindex.append(cidx)
                continue
            if  converged!=1:
                print('fc not converge')
                fc_cindex.append(cidx)
             
        cidx_result=pd.DataFrame()   
        p=pd.DataFrame()
         
         
        loss,cidx,marker=predict(x=test_x, t=test_y,delta=test_c,model=auto_surv_list[np.argmax(as_cindex)],verbose=1,name='auto_surv')
        
        cidx_result['as']=[cidx]
        p['as']=marker
        
         
        loss,cidx,marker=predict(x=test_x, t=test_y,delta=test_c,model=fc_list[np.argmax(fc_cindex)],verbose=1,name='fc')
        cidx_result['fc']=[cidx]
        p['fc']=marker
        
        print(cidx_result)
        cidx_result.to_csv('id_{}_d_{}_cidx.csv'.format(saveid,features),index=False)
        
        p.to_csv('../simulationdata/id_{}_d_{}_marker.csv'.format(saveid,features),index=False)
comparemodel(case=1,features=parser.d,saveid=parser.id)        
