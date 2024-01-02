import numpy as np
import pickle 

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse.linalg import spsolve
from scipy import optimize
from scipy.signal import find_peaks
from scipy.integrate import cumtrapz
import matplotlib.cm as cm
from scipy.interpolate import UnivariateSpline, interp1d, splrep, splev
from scipy.optimize import minimize

from numpy.linalg import norm

import os
# %matplotlib notebook
import matplotlib
# plt.rcParams["figure.figsize"]=20,8


import matplotlib.pyplot as plt


import pandas as pd
import math

# df_cathode = pd.read_excel('PB183t2_NMC811Pr.xlsx',sheet_name='Processed Data',header=[0,1])
# df_anode = pd.read_excel('PB182t1_R3GrPr.xlsx',sheet_name='Processed Data',header=[0,1])
# df_fulcel_init = pd.read_excel('PB15Rt2_NMC-Gr Initial.xlsx',sheet_name='Processed Data',header=[0,1])
df_fulcel_final = pd.read_excel('MT629t1b.xlsx',sheet_name='Processed Data',header=[0,1])

# cathode = df_cathode['Discharge 4(Oxide Lithiation)'].iloc[:,:-2].dropna(axis=0)
# cathode = cathode.iloc[:-12,:]
# anode = df_anode['Charge 1 (Gr DeLithiation)'].iloc[:,:-2].dropna(axis=0)
# cyc_init = df_fulcel_init['Discharge 5'].iloc[:,:-2].dropna(axis=0)
# cyc_final = df_fulcel_final['Discharge 309'].iloc[:,:-2].dropna(axis=0)
#3, 54, 105, 156, 207, 258, 309

print('done loading data')

s_half = 8e-7
s_cyc=9e-7

# find peaks, and compare Frechet distance around the peaks
# def find_peak(intensity):
#     peak, _ = find_peaks(intensity,prominence=10)
#     max_peak_idx = np.argmax(_["prominences"])
# #     print(_["prominences"])
#     max_peak_loc = peak[max_peak_idx]
    
#     return max_peak_loc

def find_peak(intensity):
    peak, _ = find_peaks(intensity)#,prominence=0.1)    
    return peak


def deform(k,b,x):
    return x*k-b
    
def diff(X,Y,step):
    Y=np.array(Y)
    X=np.array(X)

    dY = Y[step:]-Y[:-step]
    dX = X[step:]-X[:-step]
    return dY/dX

def mesh(a,b,c):
    data_lst = [a,b,c]
    x1= max([min(i) for i in data_lst])
    x2= min([max(i) for i in data_lst])
    x=[]
    for i in data_lst:
        x+=[j for j in i if j>x1 and j<x2]
    x=list(set(x))
    x=np.array(np.sort(x))
    
    return x,x1,x2

def interp_mesh(X,Y,z, s_num=s_half):
    spl = splrep(X, Y,s=s_num)
    return splev(z, spl)

# find the min and max, choose between 5% to 95%
# linearly increase/decrease the weight beyond masks
# 1 within the mask
def weight_mask(x,cut_off1=0,cut_off2=0.193):
    a=min(x)
    b=max(x)
    d_ab = b-a
    a+=cut_off1*d_ab
    b-=cut_off2*d_ab
    mask1 = np.where(x<=a)
    mask2 = np.where(x>=b)
    weights = np.ones(len(x))
    weights[mask1[0]]=np.linspace(0,1,len(mask1[0]))
    weights[mask2[0]]=np.linspace(1,0,len(mask2[0]))
    return weights*weights*weights*weights


print('done loading funcs')

cyc_lst = [3, 54, 105, 156, 207, 258, 309]
# cyc_lst = [309]
ttl_lst = ['init','final']
acq_lst = ['ucb', 'ei', 'poi']
target = -np.inf


step=3
cscale=1e3

m_c = 25.524
m_a = 15.754
m_f = 25.7

# TODO!!!: 
# Write the weight function to linear increase weight at the beginning and end
# Use the weigth function for BO training
# Calculate the anode cathode mAh/g 

# print(os.getcwd())
# C015 NMC532  - da1959t1 - dchg.txt
# C015 NMC532 - da1959t1 - dchg.txt

#==============================================
# mc_cathode=pd.read_csv('C015 NMC532  - da1959t1 - chg.txt', sep='\t', header=None,skiprows=1, names=['mAhg','volts'])
# catx = mc_cathode['mAhg']#/1.76625/30
# caty = mc_cathode['volts']


# mc_anode=mc_cathode=pd.read_csv('A015 Graphite - MT864t3 - chg.txt', sep='\t', header=None,skiprows=1, names=['volts','mAhg'])
# anox = mc_cathode['mAhg']#/2/40
# anoy = mc_cathode['volts']

# mc_cathode=pd.read_csv('C015 NMC532  - da1959t1 - dchg.txt', sep='\t', header=None,skiprows=1, names=['mAhg','volts'])
# catx = mc_cathode['mAhg']#/1.76625/30
# caty = mc_cathode['volts']


# mc_anode=pd.read_csv('A015 Graphite - MT864t3 - dchg.txt', sep='\t', header=None,skiprows=1, names=['volts','mAhg'])
# anox = mc_anode['mAhg']#/2/40
# anoy = mc_anode['volts']
mc_cathode = pd.read_excel('AY137t1.xlsx',sheet_name='Processed Data',header=[0,1])
samp_cathode = mc_cathode['Discharge 2']
catx = samp_cathode['Amp_hr']*cscale#/1.76625/30
caty = samp_cathode['Volts']

mc_anode = pd.read_excel('AY181t1.xlsx',sheet_name='Processed Data',header=[0,1])
samp_anode = mc_anode['Charge 5']
anox = samp_anode['Amp_hr'].dropna()*cscale#/1.76625/30
anoy = samp_anode['Volts'].dropna()
#==============================================




# cycy = splev(cycx, spl3)

# cut based on lowest peak
# smooth with spline 

ifc=0
pbounds = {'kc': (0.8, 1.2), 'bc': (0, 0.8), 'ka': (0.8, 1.2),'ba': (0,2.5)}


# pbounds = {'kc': (0.7, 1),'bc': (0,100),'ka': (0.7, 1), 'ba': (0, 100)}

res_dict = {}

for cycnum in cyc_lst:
    target = -np.inf
#     cyc_name = "Charge "+str(cycnum)
    cyc_name = "Discharge "+str(cycnum)
    cyc = df_fulcel_final[cyc_name].iloc[:,:5].dropna(axis=0)
    res_dict[cycnum]={}

    for acq in acq_lst:
        res_dict[cycnum][acq]={}

        for idx in range(3):
            res_dict[cycnum][acq][idx]={}


    #         cyc = cycnum
    #         cycx = cyc['mAh/cm2']
    #         print(cyc.columns)
            cycx = cyc['Amp_hr']*cscale
            cycy = cyc['Volts']
    #         break



            def gain(kc,bc,ka,ba):
            #     k1=0.95
            #     k2=1
            #     b1=0.0000001
            #     b2=0

                # deform original data 
                catx_def = deform(kc,bc,catx)
                anox_def = deform(ka,ba,anox)



    #             data_lst = [f_cel_root,h_ca,h_an]

                x,x1,x2=mesh(cycx,catx_def,anox_def)

    #             cut_off = int(len(x)*0.01)
    #             x=x[cut_off:-cut_off]
    #             x = x[step:]

                # TODO: spline fit the original data and take derivatives 
                try:
                    caty_def = interp_mesh(catx_def,caty,x)
                    anoy_def =  interp_mesh(anox_def,anoy,x)
                    cycy_def = interp_mesh(cycx,cycy,x,s_num=s_cyc)
                except:
                    return -1e10

                dVdQ_cat = diff(x,caty_def,step)
                dVdQ_ano = diff(x,anoy_def,step)
                dVdQ_cyc = diff(x,cycy_def,step)
        #===========================================
                peaks=find_peak(dVdQ_cyc)
                cut_off_start = 10
                
                try:
                    cut_off_end = peaks[-1]+20
                    if peaks[-1]+20 >= len(x):
                        cut_off_end = -1
                    
                except: 
                    cut_off_end = -1
                
                    
        #===========================================


    #             weights = weight_mask(x[step:])


                # TODO: find the minimum of 10% and 90% of the data x_m
                # only choose data in (1.5*x_m,0)
        #===========================================
    #             x_len = len(dVdQ_cyc)
    #             peak_min = min(dVdQ_cyc[int(x_len*0.2):int(x_len*0.8)])
    #             lower_bound = -0.75#int(1.5*peak_min)
    #             mask = np.where(dVdQ_cyc>=lower_bound)
    #             x=x[mask]
    #             dVdQ_cyc=dVdQ_cyc[mask]
    #             dVdQ_cat=dVdQ_cat[mask]
    #             dVdQ_ano=dVdQ_ano[mask]
        #===========================================
                dVdQ_fit = dVdQ_cat-dVdQ_ano
                err_vec = dVdQ_fit - dVdQ_cyc


    #             return -(np.dot(err_vec*weights,err_vec))/len(err_vec)

                if len(err_vec)==0: 
                    return -1e10
                else: 
                    return -(np.dot(err_vec[cut_off_start:cut_off_end],err_vec[cut_off_start:cut_off_end]))/len(err_vec)




            # Generate exploring list
            kc_list = []
            ka_list = []
            ba_list = []
            count = 0

            #  Build optimizer
            optimizer = BayesianOptimization(
                f=gain,
                pbounds=pbounds,
                random_state=idx,
            )
            print('starting to optimize')

            # Set up exploration first
            for i in range(count):
                optimizer.probe(params = {'kc': kc_list[i],'bc':bc_list[i], 'ka': ka_list[i], 'ba': ba_list[i]},
                               lazy = True,)

            # Record run result
            # logger = JSONLogger(path=pref+"/logs.json")
            # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

            # Run optimizer
            optimizer.maximize(
                init_points=20,
                n_iter=400,
                acq = acq
            )



            print(optimizer.max['params'])

            res_dict[cycnum][acq][idx]=optimizer.max['params']

            kc=optimizer.max['params']['kc']
            bc=optimizer.max['params']['bc']        
            ka=optimizer.max['params']['ka']
            ba=optimizer.max['params']['ba']




            catx_def = deform(kc,bc,catx)
            anox_def = deform(ka,ba,anox)
            x,x1,x2=mesh(cycx,catx,anox)
            x = x[step:]

    #         normed_target = optimizer.max['target']/len(x) 
    #         if normed_target > target: 
    #             target = normed_target
    #             next_pbounds = {'kc': (0.5*kc, kc),'bc':(bc,2*bc), 'ka': (0.5*ka, ka), 'ba': ( ba, 2*ba)}


            caty_def = interp_mesh(catx_def,caty,x)
            anoy_def = interp_mesh(anox_def,anoy,x)
            cycy_def = interp_mesh(cycx,cycy,x,s_num=s_cyc)

            dVdQ_cat = diff(x,caty_def,step)
            dVdQ_ano = diff(x,anoy_def,step)
            dVdQ_cyc = diff(x,cycy_def,step)

            x_len = len(dVdQ_cyc)
    #         lower_bound = -1.9
    #         mask = np.where(dVdQ_cyc<lower_bound)
    #         weights = np.ones(x_len)
    #         weights[mask]=0.01


    #         mask = np.where(dVdQ_cyc>=lower_bound)
    #         x=x[mask]
    #         dVdQ_cyc=dVdQ_cyc[mask]
    #         dVdQ_cat=dVdQ_cat[mask]
    #         dVdQ_ano=dVdQ_ano[mask]

            dVdQ_fit = dVdQ_cat-dVdQ_ano

            fig = plt.subplot()

            fig.plot(x[step:],dVdQ_cyc,label=str(optimizer.max['target']))
            fig.plot(x[step:],dVdQ_fit,label=acq)

            rn = 6
            plt.legend(loc='best')
            plt.title(' %s, kc=%s, bc=%s, ka=%s, ba = %s' %(cyc_name,str(round(kc,rn)),str(round(bc,rn)), str(round(ka,rn)),str(round(ba,rn))))
            plt.ylim(-0.02,0)
            plt.show()
            plt.savefig('./plots/ %s acq %s idx %s' %(cyc_name,acq,idx))
            plt.figure().clear()
    #     ifc+=1
    #     pbounds = next_pbounds
    #     print(pbounds)




# create a binary pickle file 
f = open("res_dict.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(res_dict,f)

# close file
f.close()
