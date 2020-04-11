# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:39:39 2020

@author: Administrator
"""
 
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
from hyperopt import hp,tpe,Trials,fmin,STATUS_OK
import math

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
from hyperopt import hp,tpe,Trials,fmin 
import argparse
class surv_experiment():
    def __init__(self,simu=True,cancer='BRCA',n=100,p=1000,realp=20,\
                distribution_type='gaussian',\
                 time_distribution='weibull',epochs=1000,\
                 la=0.8,al=1,miu=2,cencorrate=0.4,sigma=0.2,random_seed=0,\
                 training_percent=0.6,val_percent=0.2,lr=0.001,es=False,save=True,verbose=1,saveid=1):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')      
        if simu==True:
            self.full_x,self.full_t,self.full_c,self.c_rate=self.simulatedata(n=n,p=p,realp=realp,distribution_type=distribution_type,\
    time_distribution=time_distribution,\
    la=la,al=al,miu=miu,cencorrate=cencorrate,sigma=sigma,random_seed=random_seed,\
    effect_case=0,x_case=0)
            print('censorsamples_rate={}'.format(self.c_rate))
            self.train_x,self.train_t,self.train_c,\
            self.val_x,self.val_t,self.val_c,\
            self.test_x,self.test_t,self.test_c=self.splitdata(self.full_x,\
            self.full_t,self.full_c,\
            training_percent=training_percent,val_percent=val_percent,save=save,saveid=saveid)
            self.features=p
            self.realp=realp
        else:
            x=pd.read_csv('{}_x.csv'.format(cancer))
            t=pd.read_csv('{}_t.csv'.format(cancer))
            c=pd.read_csv('{}_c.csv'.format(cancer))
            self.splitdata(x,\
            t,c,\
            training_percent=training_percent,val_percent=val_percent,save=save,saveid=saveid)
        self.seed=random_seed
        self.maxepochs=epochs
        self.lr=lr
        self.verbose=verbose
        self.noise=sigma
        self.saveid=saveid
        self.model_out_info={}
        self.earlystop=es
        self.n=n
        '''
        a1=self.test_model( 'as',self.train_x,self.train_t,self.train_c,self.val_x,\
                        self.val_t,self.val_c,\
    self.test_x,self.test_t,self.test_c)
        print('a1 done')
         
        a2=self.test_model('fc',self.train_x,self.train_t,self.train_c,self.val_x,\
                        self.val_t,self.val_c,\
    self.test_x,self.test_t,self.test_c)
        print('a2 done')
        self.test_model('xgb',self.train_x,self.train_t,self.train_c,self.val_x,\
                        self.val_t,self.val_c,\
    self.test_x,self.test_t,self.test_c)
        print([a1['name'],a1['cindex'],a1['hyp_best']])
        print([a2['name'],a2['cindex'],a2['hyp_best']])
        ''' 
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
            d=np.concatenate((X,extra_x),axis=1)
        if case==1:
            assert p/2==int(p/2)
            generate_model=LinearBlock( realp,p/2, bn=0,dr_p=0,no_tail=True,bias=False)
            extra_x=np.random.uniform(0,1,n*(p/2)).reshape((n,(p/2))).astype(np.float32,casting='same_kind')
            X=generate_model(torch.from_numpy(X)).data.numpy()
             
            d=np.concatenate((X,extra_x),axis=1)
        
        if case==2:
            
            generate_model=Fc( encoder_hidden_size=[realp,int(np.sqrt(p/2)),p/2], bn=0,dr_p=0)
            extra_x=np.random.uniform(0,1,n*(p/2)).reshape((n,(p/2))).astype(np.float32,casting='same_kind')
            X=generate_model(torch.from_numpy(X)).data.numpy()
            d=np.concatenate((X,extra_x),axis=1)
        if distribution_type=='gaussian':
            return  d+sigma*np.random.normal(0,1,n*p).reshape((n,p)), X
        
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
            if effect_case in [0,2]:
                sample_contrib_term=np.exp(generate_model(torch.from_numpy( essential_X[i]).unsqueeze(0)).cpu().data.numpy()[0][0] )
            if effect_case in [1]:
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
    def bayesopt(self,name):
        tpe_algo = tpe.suggest
        tpe_trials = Trials()
        if name=='as':
            space = [hp.quniform('bottleneck', 10, 30,1),\
         hp.uniform('drop_out', 0.1, 0.5),hp.quniform('encoder_depth', 2, 4,1),\
        hp.quniform('fc_depth',1, 3,1),hp.loguniform('lambda1', -5, 1)]
            def obj(args):
                param={'encoder_hidden_size': self.make_hidden_list(\
      in_size=self.features,bottleneck_size=args[0],depth=args[2]),\
 'cox_hidden_size': self.make_hidden_list(\
      in_size=args[0],bottleneck_size=1,depth=args[3]), 'bn': 1, 'dr_p': args[1], \
 'num_out_layers': 1, 'cat_type': 'null', 'lambda1': args[4],\
 'bn_cox': 1, 'dr_p_cox': args[1], 'noise': None}
                loss,model=self.test_hyp(name,param)
                return {'loss':-loss, 'status': STATUS_OK, 'Trained_Model': model}
                  
            tpe_best = fmin(fn=obj, space=space, 
                            algo=tpe_algo, trials=tpe_trials, 
                            max_evals=15,rstate=np.random.RandomState(self.seed))
            print(tpe_best)
        if name=='fc':
            space = [hp.uniform('drop_out', 0.1, 0.5),\
            hp.quniform('encoder_depth', 2, 4,1)]
            def obj(args):
                 param={'encoder_hidden_size':self.make_hidden_list(\
                      in_size=self.features,bottleneck_size=1,depth=args[1]),\
    'bn':1,'dr_p':args[0]}
                 print(param)
                 loss,model=self.test_hyp(name,\
            param)
                 return {'loss':-loss, 'status': STATUS_OK, 'Trained_Model': model}
                  
            tpe_best = fmin(fn=obj, space=space, 
                            algo=tpe_algo, trials=tpe_trials, 
                            max_evals=15,rstate=np.random.RandomState(self.seed))
        selectedmodel=[x['result']['model'] for x in list(tpe_trials)][\
         np.argmin( x['result']['loss'] for x in list(tpe_trials) )]
        loss,cidx,marker=predict(x=self.test_x, t=self.test_t,\
                    delta=self.test_c,model=selectedmodel,verbose=1,name=name)
    def test_hyp(self,name,param={'encoder_hidden_size':[100,40,10], 
                                'cox_hidden_size':[10,10,1],'bn':1,'dr_p':0.2,
                               'num_out_layers':1,
       'cat_type':'null','lambda1':0.1,'bn_cox':1,'dr_p_cox':0.2,'noise':None}):
        '''
        {'encoder_hidden_size':hidden_size, 
                                'cox_hidden_size':cox_hidden_size+[1],'bn':bn,'dr_p':dr_p,
                               'num_out_layers':1,
       'cat_type':'null','lambda1':lambda1,'bn_cox':bn,'dr_p_cox':dr_p,'noise':None}
        '''
        if name in ['as','fc']:
            if name=='as':
                ele=Autosurv(**param)
            if name=='fc':
                ele=Fc(**param)
            if self.earlystop==False:
                converged=training_Autosurv(x=self.train_x, t=self.train_t,\
        delta=self.train_c,model=ele,\
                lr=self.lr,lambda1=ele.lambda1 if name=='as' else None,epoch=1000,verbose=2,name=name)
                loss,cidx,marker=predict(x=self.val_x, t=self.val_t,\
                    delta=self.val_c,model=ele,verbose=1,name=name)
                if converged==1:
                    print('converged')
                    print(cidx)
                if converged!=1:
                    print('as not converge')
                    print(cidx)
            else:
                loss_list=[]
                for i in range(self.maxepochs):
                    training_Autosurv(x=self.train_x, t=self.train_t,\
        delta=self.train_c,model=ele,\
                lr=self.lr,lambda1=ele.lambda1  if name=='as' else None,epoch=1,verbose=2,name=name)
                    loss,cidx,marker=predict(x=self.val_x, t=self.val_t,\
                    delta=self.val_c,model=ele,verbose=1,name=name)
                    loss_list.append(loss)
                    if len(loss_list)>4 and loss_list[-1]>loss_list[-2] and \
                    loss_list[-2]>loss_list[-3]:
                        print('converged')
                        print(cidx)
                        break
                    if i==self.maxepochs-1:
                        print('fc not converge')
                        print(cidx)
            return cidx,ele
        if name in ['xgb']:
            xgb_tr_x=self.train_x.data.numpy()
             
            xgb_tr_t=np.array( [self.train_t.data.numpy()[idx] if self.train_c.data.numpy()[idx]==1 \
            else -self.train_t.data.numpy()[idx] \
    for idx in range(len(self.train_t.data.numpy()))] )
            xgb_va_x=self.val_x.data.numpy()
             
            #gamma 0-10 max_depth 1-10 eta 0.1-0.5
            gbm=xgb.XGBRegressor(objective='survival:cox',booster='gbtree',\
  max_depth=int(param['max_depth']),\
        eta=param['eta'],gamma=param['gamma']).fit(xgb_tr_x,xgb_tr_t)
            
            loss,cidx,marker=predict(x=xgb_va_x, t=self.val_t.data.numpy(),\
    delta=self.val_c.data.numpy(),\
                model=gbm,verbose=1,name=name)
            return cidx,gbm
        
    def test_model(self,name,train_x,train_t,train_c,val_x,val_t,val_c,\
    test_x,test_t,test_c):
         
        tr_x,tr_t,tr_delta,tr_Rlist=order_make_Rlist(train_x,train_t,train_c)
        va_x,va_t,va_delta,va_Rlist=order_make_Rlist(val_x,val_t,val_c)
        test_x,test_t,test_delta,test_Rlist=order_make_Rlist(test_x,test_t,test_c)
        if name in ['as','fc','xgb']:
            tpe_algo = tpe.suggest
            tpe_trials = Trials()
            if name=='as':
                space = [hp.quniform('bottleneck', 10, 30,1),\
             hp.uniform('drop_out', 0.1, 0.5),hp.quniform('encoder_depth', 1, 4,1),\
            hp.quniform('fc_depth',1, 3,1),hp.loguniform('lambda1', -5, 1)]
                self.tr_x=tr_x
                self.tr_t=tr_t
                self.tr_c=tr_delta
                self.tr_Rlist=tr_Rlist
                self.va_x=va_x
                self.va_t=va_t
                self.va_c=va_delta
            if name=='fc':
                self.tr_x=tr_x
                self.tr_t=tr_t
                self.tr_c=tr_delta
                self.tr_Rlist=tr_Rlist
                self.va_x=va_x
                self.va_t=va_t
                self.va_c=va_delta
                space = [hp.quniform('bottleneck', 10, 30,1),\
                         hp.uniform('drop_out', 0.1, 0.5),\
            hp.quniform('encoder_depth', 1, 4,1)]
            if name=='xgb':
                self.tr_x=tr_x.data.numpy()
                self.tr_t=tr_t.data.numpy()
                self.tr_c=tr_delta.data.numpy()
                self.va_x=va_x.data.numpy()
                self.va_t=va_t.data.numpy()
                self.va_c=va_delta.data.numpy()
                space = [hp.quniform('max_depth',1,5,1),\
            hp.quniform('n_estimator', 10,1000,1)]
            self.name=name
            obj=self.training_mod
             
            tpe_best = fmin(fn=obj, space=space, 
                            algo=tpe_algo, trials=tpe_trials, 
                            max_evals=15,rstate=np.random.RandomState(self.seed))
            print('id_{}_d_{}_rd_{}_noi{}_sam_{}_estop_{}'.format(self.saveid,\
    self.features,self.realp,self.noise,self.n,int(self.earlystop)))
            #print(name)
            #print(tpe_best)
            if name=='xgb':
                test_x=test_x.data.numpy()
                test_t=test_t.data.numpy()
                test_c=test_c.data.numpy()
            if name in ['glmnet']:
                pass
         
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
         
        para_result=[]
        if name=='as':
            para_result=[tpe_best['encoder_depth'],\
            tpe_best['bottleneck'],tpe_best['fc_depth'],tpe_best['lambda1']
            ,tpe_best['drop_out']]
        if name=='fc':
            para_result=[tpe_best['encoder_depth'],\
            tpe_best['bottleneck'],tpe_best['drop_out']]
        if name=='xgb':
            para_result=[tpe_best['max_depth'],\
            tpe_best['n_estimator']]
        #print(tpe_best)
        self.training_mod(list(tpe_best.values()))
        loss,cidx,marker=self.predict(x=test_x, t=test_t,delta=test_c,\
    model=self.tmp_model,verbose=1,name=name)
        self.model_out_info={'name':name,'loss':loss,'cindex':cidx,'marker':marker,\
                           'model':self.tmp_model,'hyp_best':tpe_best}
        
        print('name={} cindex={} hyp_best={}'.format(name,cidx,tpe_best))
        return self.model_out_info
           
    
        
                    
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
            marker=model.predict(x).reshape(-1)
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
        epochs=self.maxepochs
        verbose=self.verbose
        device=self.device
        name=self.name
        
       
        
        loss_hist=[]
        if name in ['as','fc']:
            
            
            if name=='as':
                hidden_size=self.make_hidden_list(in_size=self.features,\
            depth=param[2],bottleneck_size=param[0])
                #[hp.quniform('bottleneck', 10, 30,1),\
             #hp.uniform('drop_out', 0.1, 0.5),hp.quniform('encoder_depth', 1, 4,1),\
            #hp.quniform('fc_depth',2, 3,1),hp.loguniform('lambda1', -3, 2)]
                param_dict={'encoder_hidden_size':hidden_size, \
       'cox_hidden_size':[int(param[0])]*int( param[3]) +[1],'bn':1,'dr_p':param[1],
       'num_out_layers':1,\
       'cat_type':'null','lambda1':param[4],'bn_cox':1,'dr_p_cox':param[1],'noise':None}
                print(param_dict)
                model=Autosurv(**param_dict) 
            if name=='fc':
                hidden_size=self.make_hidden_list(in_size=self.features,\
        depth=param[2],bottleneck_size=param[0])+[1]
                param_dict={'encoder_hidden_size':hidden_size,'bn':1,'dr_p':param[1]}
                #print(param_dict)
                model=Fc(**param_dict) 
            model=model.train()
            model=model.to(device)
            x=self.tr_x.to(device)
            optimizer=torch.optim.Adam(model.parameters(), lr=lr,amsgrad=True) 
            for iteration in range(epochs):
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
                if torch.isnan(loss).cpu().data.numpy():
                    print('nan!')
                    return 1
                     
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
                loss_hist.append(loss)
                if verbose==2:
                    print('training one epoch used {}s,loss={}'.format(time.time()-t0,loss))
                     
                if len(loss_hist)>4 and loss_hist[-1]>loss_hist[-2] and loss_hist[-2]>loss_hist[-3]:
                    print('converged')
                    break
                else:
                    if iteration==epochs-1:
                        print('non converge')
            if self.earlystop==False:
                loss,cidx,marker=self.predict(x=self.va_x, t=self.va_t,delta=self.va_c,\
                    model=model,verbose=1,name=name)
            self.tmp_model=model
            self.tmp_marker=marker
            print('after training cindex={}'.format(cidx))
            return -cidx
        if name in ['xgb']:
            xgb_tr_x=self.tr_x
             
            xgb_tr_t=np.array( [self.tr_t[idx] if self.tr_c[idx]==1 \
            else -self.tr_t[idx] \
    for idx in range(len(self.tr_t))] )
            xgb_va_x=self.tr_x 
            gbm=xgb.XGBRegressor(objective='survival:cox',booster='gbtree',max_depth=int(param[0]),\
        n_estimators=int(param[1])).fit(xgb_tr_x,xgb_tr_t)
            loss,cidx,marker=self.predict(x=xgb_va_x, t=self.va_t,delta=self.va_c,\
                model=gbm,verbose=1,name=name)
            self.tmp_model=model
            self.tmp_marker=marker
        return -cidx
    def save_model_out_info(self,info_list):
        cidx_result=pd.DataFrame()
        for name in self.model_out_info.keys:
            cidx_result[name]=self.model_out_info[name]['cindex']
        cidx_result.to_csv('id_{}_d_{}_rd_{}_noi{}_sam_{}_es_{}cidx.csv'.format(self.saveid,\
    self.features,self.realp,self.noise,self.n,int(self.earlystop)),index=False)
        p=pd.DataFrame()
        for name in self.model_out_info.keys:
            p[name]=self.model_out_info[name]['marker']
        p.to_csv('../simulationdata/id_{}_d_{}_rd_{}_noi{}_sam_{}_es_{}marker.csv'.\
    format(self.saveid,\
    self.features,self.realp,self.noise,self.n,int(self.earlystop)),index=False)
 
    def make_hidden_list(self,in_size,depth,bottleneck_size):
        step= ( np.log(bottleneck_size)-np.log(in_size) )/depth
        ret=list( np.exp( np.arange(np.log(in_size),np.log(bottleneck_size)-0.1*step,step) ).astype(int))
        ret[0]=int(in_size)
        ret[-1]=int(bottleneck_size)
        return ret         





class LinearBlock(torch.nn.Module):
    def __init__(self, in_channel,out_channel,bn,dr_p ,no_tail=False,bias=True):
        super(LinearBlock, self).__init__()
         
         
        mylist=ModuleList()
        mylist.append(Linear( in_channel,out_channel,bias=bias))
        if no_tail==False and out_channel>1:
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
        assert encoder_hidden_size[-1]==1
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
        if name=='as':
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
    if name in ['as','fc']:
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
        marker=model.predict(x)
        c_idx=cindex(marker,Rlist) 
        return None,c_idx,marker
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
