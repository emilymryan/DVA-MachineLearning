import numpy as np
import ast

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
from scipy.interpolate import CubicSpline


from numpy.linalg import norm

import os
import pickle
# %matplotlib notebook
import matplotlib
# plt.rcParams["figure.figsize"]=20,8


import matplotlib.pyplot as plt
import random


import pandas as pd
import math

task_id = int(os.getenv('SGE_TASK_ID'))-1

df_fulcel_final = pd.read_excel('MT629t1b_corrected.xlsx',header=[0,1]).drop(index=0).reset_index(drop=True)

print('done loading data')

s_half = 1e-12
s_cyc=9e-7


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

# def interp_mesh(X,Y,z, s_num=s_half):
#     spl = splrep(X, Y,s=s_num)
#     return splev(z, spl)

def interp_mesh(X, Y, z, s_num=s_half):
    # Check that X and Y have the same length
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
    
    # Check that X is in strictly increasing order
    if not all(X[i] < X[i + 1] for i in range(len(X) - 1)):
        raise ValueError("X must be strictly increasing")
    
    try:
        # Print input data for debugging
#         print("X:", X)
#         print("Y:", Y)
#         print("s_num:", s_num)
        
        spl = splrep(X, Y, s=s_num)
        return splev(z, spl)
    except ValueError as e:
        print("ValueError:", e)
        raise

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
# cyc_lst = [3, 54, 105, 156, 207, 258]
# cyc_lst = [309]
ttl_lst = ['init','final']
acq_lst = ['ucb', 'ei', 'poi']
target = -np.inf


step=3
cscale=1e3

m_c = 25.524
m_a = 15.754
m_f = 25.7


mc_cathode = pd.read_excel('AY137t1.xlsx',sheet_name='Processed Data',header=[0,1])
mc_anode = pd.read_excel('AY181t1.xlsx',sheet_name='Processed Data',header=[0,1])

samp_cathode_D = mc_cathode['Discharge 2']
catx_D = samp_cathode_D['Amp_hr']*cscale#/1.76625/30
caty_D = samp_cathode_D['Volts']

samp_anode_D = mc_anode['Charge 5']
anox_D = samp_anode_D['Amp_hr'].dropna()*cscale#/1.76625/30
anoy_D = samp_anode_D['Volts'].dropna()


samp_cathode_C = mc_cathode['Charge 2']
samp_cathode_C.dropna(subset=['Amp_hr'], inplace=True)
catx_C = samp_cathode_C['Amp_hr']*cscale#/1.76625/30
caty_C = samp_cathode_C['Volts']

samp_anode_C = mc_anode['Discharge 5']
samp_anode_C.dropna(subset=['Amp_hr'], inplace=True)
anox_C = samp_anode_C['Amp_hr'].dropna()*cscale#/1.76625/30
anoy_C = samp_anode_C['Volts'].dropna()


Q_cathode_max = mc_cathode['Charge 2']['Amp_hr'].dropna().iloc[-1]*cscale
Q_anode_max = mc_anode['Discharge 4']['Amp_hr'].dropna().iloc[-1]*cscale

catx_D = Q_cathode_max-catx_D
anox_D = Q_anode_max-anox_D

catx_D = np.flip(catx_D).reset_index(drop=True)
anox_D = np.flip(anox_D).reset_index(drop=True)
caty_D = np.flip(caty_D).reset_index(drop=True)
anoy_D = np.flip(anoy_D).reset_index(drop=True)
#==============================================

fast_lst = []

for i in range(len(cyc_lst)-1): 
    
    mid = (cyc_lst[i]+cyc_lst[i+1])//2
    
#     fast_lst+=[mid-1,mid,mid+1]
    fast_lst+=[mid]
    
# control the list being fit
slow_lst = cyc_lst[1:]
# fast_lst = slow_lst
    
ifc=0
pbounds = {'kc': (0.8, 1.2), 'bc': (-1.5,0), 'ka': (0.9, 1.2),'ba': (-1.5,0),'r': (0,0.1)}

# pbounds = {'kc': (0.8, 0.9), 'bc': (0.1, 0.12), 'ka': (0.8, 0.9),'ba': (0,0.2),'r': (0,0.1)}


# pbounds = {'kc': (0.7, 1),'bc': (0,100),'ka': (0.7, 1), 'ba': (0, 100)}

res_dict = {}


# plot the differential voltage curves given k, b, and r

def plot_DVQ(ka, kc,ba, bc, r,cycx,cycy,cycnum,acq,idx,cd,fs):
    if cd == 'Discharge':

        catx_def = deform(kc,bc,catx_D)
        anox_def = deform(ka,ba,anox_D)
        caty = caty_D
        anoy = anoy_D
        
        cyc_name_D = "Discharge "+str(cycnum)
        



        cyc = df_fulcel_final[cyc_name_D].iloc[:,list(range(0, 6)) + [-1]].dropna(axis=0)

        cycx = cyc['Amp_hr_actual']*cscale
        cycy = cyc['Volts']
        cycx = np.flip(cycx).reset_index(drop=True)
        cycy = np.flip(cycy).reset_index(drop=True)


    else:
        catx_def = deform(kc,bc,catx_C)
        anox_def = deform(ka,ba,anox_C)
        caty = caty_C
        anoy = anoy_C
        
        cyc_name_C = "Charge "+str(cycnum)
        cyc = df_fulcel_final[cyc_name_C].iloc[:,list(range(0, 6)) + [-1]].dropna(axis=0)
        cycx = cyc['Amp_hr_actual']*cscale
        cycy = cyc['Volts']

    x,x1,x2=mesh(cycx,catx_def,anox_def)
#         x = x[step:]

    caty_def = interp_mesh(catx_def,caty,x,s_num=s_half)
    anoy_def = interp_mesh(anox_def,anoy,x,s_num=s_half)            
    cycy_def = interp_mesh(cycx,cycy,x,s_num=s_cyc)

    dVdQ_cat = diff(x,caty_def,step)
    dVdQ_ano = diff(x,anoy_def,step)
    dVdQ_cyc = diff(x,cycy_def,step)




    dVdQ_fit = dVdQ_cat-dVdQ_ano

    fig = plt.subplot()

    fig.plot(x[step:],dVdQ_cyc,label='gt')
    fig.plot(x[step:],dVdQ_fit,label='fit')

    
    plt.title('DVQ '+ str(cycnum) + ' ' + cd)
    plt.legend()
    
    plt.ylim(0,0.5)
#     plt.show()
    plt.savefig('/projectnb/ryanlab/jackz/test/Diff_V/BO_vs_GD/NMC532-Gr_BO/combined_plots_%s/DVQ %s cyc %s acq %s idx %s.png'%(str(fs),cd,str(cycnum),str(acq),str(idx)))
    plt.figure().clear()

    
def loss_DVQ(ka, kc, ba, bc, r, cycx,cycy,cd):

    if cd == 'Discharge':
        catx_def = deform(kc,bc,catx_D)
        anox_def = deform(ka,ba,anox_D)
        caty = caty_D
        anoy = anoy_D
        
        cyc_name_D = "Discharge "+str(cycnum)
        cyc = df_fulcel_final[cyc_name_D].iloc[:,list(range(0, 6)) + [-1]].dropna(axis=0)
        cycx = cyc['Amp_hr_actual']*cscale
        cycy = cyc['Volts']
        cycx = np.flip(cycx).reset_index(drop=True)
        cycy = np.flip(cycy).reset_index(drop=True)
        

        
        
    else:
        catx_def = deform(kc,bc,catx_C)
        anox_def = deform(ka,ba,anox_C)
        caty = caty_C
        anoy = anoy_C
        
        cyc_name_C = "Charge "+str(cycnum)
        cyc = df_fulcel_final[cyc_name_C].iloc[:,list(range(0, 6)) + [-1]].dropna(axis=0)
        cycx = cyc['Amp_hr_actual']*cscale
        cycy = cyc['Volts']

    x,x1,x2=mesh(cycx,catx_def,anox_def)
#         x = x[step:]

    caty_def = interp_mesh(catx_def,caty,x,s_num=s_half)
    anoy_def = interp_mesh(anox_def,anoy,x,s_num=s_half)            
    cycy_def = interp_mesh(cycx,cycy,x,s_num=s_cyc)

    dVdQ_cat = diff(x,caty_def,step)
    dVdQ_ano = diff(x,anoy_def,step)
    dVdQ_cyc = diff(x,cycy_def,step)
    
    dVdQ_fit = dVdQ_cat-dVdQ_ano


#===========================================
    # Convert the list to a numpy array
    x_array = np.array(x)
    # Find the index of the first element greater than 0.45
#     cut_off_start = np.argmin(x_array < 0.05)
    cut_off_start = 0
    # Find the index of the last element less than 0.45
#     cut_off_end = np.argmin(x_array < 3.2)
    cut_off_end = -1
#     dVdQ_fit = dVdQ_cat - dVdQ_ano
    err_vec = dVdQ_fit  - dVdQ_cyc
    loss = np.dot(err_vec[cut_off_start:cut_off_end],err_vec[cut_off_start:cut_off_end])
        

    return loss




mod_D = 1e-4
mod_C = 1e-4
loss_start = 3.2
loss_end = 4.5

fs_lst = ['fast','slow']

for fs in fs_lst: 
    
    if fs == 'fast':
        cycnum = slow_lst[task_id//9]-1
        idx = task_id%9
        current = 0.001001144



    else: 
        cycnum = slow_lst[task_id//9]
        idx = task_id%9
        current = 0.00012
        
    target = -np.inf
    #     cyc_name = "Charge "+str(cycnum)
    cyc_name_D = "Discharge "+str(cycnum)
    cyc_name_C = "Charge "+str(cycnum)




    cyc_D = df_fulcel_final[cyc_name_D].iloc[:,list(range(0, 6)) + [-1]].dropna(axis=0)
    cyc_C = df_fulcel_final[cyc_name_C].iloc[:,list(range(0, 6)) + [-1]].dropna(axis=0)

    cycx_D = cyc_D['Amp_hr_actual']*cscale
    cycy_D = cyc_D['Volts']
    cycx_D = np.flip(cycx_D).reset_index(drop=True)
    cycy_D = np.flip(cycy_D).reset_index(drop=True)


    cycx_C = cyc_C['Amp_hr_actual']*cscale
    cycy_C = cyc_C['Volts']


    dVdQ_cat_C = diff(catx_C,caty_C,step)
    dVdQ_ano_C = diff(anox_C,anoy_C,step)
    dVdQ_cyc_C = diff(cycx_C,cycy_C,step)


    dVdQ_cat_D = diff(catx_D, caty_D, step)
    dVdQ_ano_D = diff(anox_D, anoy_D, step)
    dVdQ_cyc_D = diff(cycx_D, cycy_D, step)

    res_dict[cycnum]={}

    for acq in acq_lst:
        res_dict[cycnum][acq]={}

        res_dict[cycnum][acq][idx]={}


    #         cyc = cycnum
    #         cycx = cyc['mAh/cm2']
    #         print(cyc.columns)
    #     cycx = (Qc_end-cyc['Amp_hr'])*cscale
        


        def gain(ka, ba, kc,bc,r):
        #     k1=0.95
        #     k2=1
        #     b1=0.0000001
        #     b2=0

            # deform original data 
            catx_def_D = deform(kc,bc,catx_D)
            anox_def_D = deform(ka,ba,anox_D)
            
            catx_def_C = deform(kc,bc,catx_C)
            anox_def_C = deform(ka,ba,anox_C)

            # norm by max
    #                 caty_normed = norm_by_max(caty)
    #                 anoy_normed = norm_by_max(anoy)
            caty_normed_D = caty_D
            anoy_normed_D = anoy_D
            caty_normed_C = np.flip(caty_C)
            anoy_normed_C = np.flip(anoy_C)
 

    #             data_lst = [f_cel_root,h_ca,h_an]

            x_D,x1_D,x2_D=mesh(cycx_D,catx_def_D,anox_def_D)
            x_C,x1_C,x2_C=mesh(cycx_C,catx_def_C,anox_def_C)
            x_D  = x_D[(x_D > loss_start) & (x_D < loss_end)]
            x_C  = x_C[(x_C > loss_start) & (x_C < loss_end)]



            # TODO: spline fit the original data and take derivatives 
            try:
#                 print('a0')
                caty_def_D = interp_mesh(catx_def_D,caty_normed_D,x_D,s_num=s_half)
#                 print('a1')
                        
                anoy_def_D = interp_mesh(anox_def_D,anoy_normed_D,x_D,s_num=s_half)
#                 print('a2')
                cycy_def_D = interp_mesh(cycx_D,cycy_D,x_D,s_num=s_cyc)
#                 print('a3')
#                     print(len(catx_def_C))
#                     print(len(caty_normed_C))
#                     print(catx_def_C)


                caty_def_C = interp_mesh(catx_def_C,caty_normed_C,x_C,s_num=s_half)
#                 print('a4')
                anoy_def_C = interp_mesh(anox_def_C,anoy_normed_C,x_C,s_num=s_half)
#                 print('a5')
                cycy_def_C = interp_mesh(cycx_C,cycy_C,x_C,s_num=s_cyc)
                


            except:
                print(4)
                return -1e10

    #===========================================
    #                 peaks=find_peak(dVdQ_cyc)
    #                 cut_off_start = 10
    #                 try:
    #                     cut_off_end = peaks[-1]+10
    #                 except:
    #                     cut_off_end =-1
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
    # TODO: add resistance 
            if len(x_C)==0 or len(x_D)==0:
                print(5)
                return -1e10
            VQ_fit_D = caty_def_D-anoy_def_D - r*current
            err_vec_D = VQ_fit_D - cycy_def_D
            VQ_fit_C = caty_def_C-anoy_def_C + r*current
            err_vec_C = VQ_fit_C - cycy_def_C

            loss_D = (np.dot(err_vec_D,err_vec_D))/len(err_vec_D)+loss_DVQ(ka, kc, ba, bc, r,cycx_D,cycy_D,"Discharge")*mod_D
            loss_C = (np.dot(err_vec_C,err_vec_C))/len(err_vec_C)+loss_DVQ(ka, kc,ba, bc, r,cycx_C,cycy_C,"Charge")*mod_C         
#             print('Gain D, C: ')
#             print(gain_D)
#             print(gain_C)
            
            
            return -loss_D-loss_C
        
            #============================================





        # Generate exploring list
        kc_list = []
        ka_list = []
        ba_list = []
        bc_list = []
        r_list = []
    #             rb_list = []
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
            optimizer.probe(params = {'kc': kc_list[i],'bc':bc_list[i], 'ka': ka_list[i], 'ba': ba_list[i], 'r':rk_list[i]},
                           lazy = True,)

        # Record run result
        # logger = JSONLogger(path=pref+"/logs.json")
        # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        # Run optimizer
        optimizer.maximize(
            init_points=40,
            n_iter=600,
            acq = acq
        )



        print(optimizer.max['params'])

        res_dict[cycnum][acq][idx]=optimizer.max['params']

        kc=optimizer.max['params']['kc']
        bc=optimizer.max['params']['bc']        
        ka=optimizer.max['params']['ka']
        ba=optimizer.max['params']['ba']
        r=optimizer.max['params']['r']
 



     

        res_dict[cycnum][acq][idx]={'kc':kc,'bc':bc,'ka':ka,'ba':ba,'r':r}


        
        catx_def_D = deform(kc,bc,catx_D)
        anox_def_D = deform(ka,ba,anox_D)

        catx_def_C = deform(kc,bc,catx_C)
        anox_def_C = deform(ka,ba,anox_C)


        caty_normed_D = caty_D
        anoy_normed_D = anoy_D
        caty_normed_C = caty_C
        anoy_normed_C = anoy_C



        x_D,x1_D,x2_D=mesh(cycx_D,catx_def_D,anox_def_D)
        x_C,x1_C,x2_C=mesh(cycx_C,catx_def_C,anox_def_C)
        



        # TODO: spline fit the original data and take derivatives 
        caty_def_D = interp_mesh(catx_def_D,caty_normed_D,x_D,s_num=s_half)
        anoy_def_D = interp_mesh(anox_def_D,anoy_normed_D,x_D,s_num=s_half)
        cycy_def_D = interp_mesh(cycx_D,cycy_D,x_D,s_num=s_cyc)        
        caty_def_C = interp_mesh(catx_def_C,caty_normed_C,x_C,s_num=s_half)
        anoy_def_C = interp_mesh(anox_def_C,anoy_normed_C,x_C,s_num=s_half)
        cycy_def_C = interp_mesh(cycx_C,cycy_C,x_C,s_num=s_cyc)



        VQ_fit_D = caty_def_D-anoy_def_D - r*current #include I here. 
        err_vec_D = VQ_fit_D - cycy_def_D
        VQ_fit_C = caty_def_C-anoy_def_C + r*current
        err_vec_C = VQ_fit_C - cycy_def_C

        fig = plt.subplot()

        fig.plot(x_D,cycy_def_D,label=str(r*current))
        fig.plot(x_D,VQ_fit_D,label=acq)

        rn = 6
        plt.legend(loc='best')
        print(cycnum)
        print(count)
        print(ttl_lst[count])
        plt.title(' %s, kc=%s, bc=%s, ka=%s, ba = %s' %(cycnum,str(round(kc,rn)),str(round(bc,rn)), str(round(ka,rn)),str(round(ba,rn))))
    #             annotations = [
    #                 "length x = " +  str(len(dVdQ_cyc)),
    #                 "min x = " + str(min(x)),
    #                 "max x = " + str(max(x))
    #             ]
    #             for i, annotation in enumerate(annotations):
    #                 fig.text(0.1, -0.12 - i * 0.03, annotation, {'color': 'C0', 'fontsize': 13})

    #             fig.text(0.1, -0.12, eq, {'color': 'C0', 'fontsize': 13})

    #             plt.ylim(-0.4,0)
    #             plt.show()
        plt.savefig('/projectnb/ryanlab/jackz/test/Diff_V/BO_vs_GD/NMC532-Gr_BO/combined_plots_%s/VQ Discharge cyc %s acq %s idx %s.png'%(str(fs),str(cycnum),str(acq),str(idx)))
        plt.figure().clear()

        fig = plt.subplot()

        fig.plot(x_C,cycy_def_C,label=str(r*current))
        fig.plot(x_C,VQ_fit_C,label=acq)

        rn = 6
        plt.legend(loc='best')
        print(cycnum)
        print(count)
        print(ttl_lst[count])
        plt.title(' %s, kc=%s, bc=%s, ka=%s, ba = %s' %(cycnum,str(round(kc,rn)),str(round(bc,rn)), str(round(ka,rn)),str(round(ba,rn))))
        plt.savefig('/projectnb/ryanlab/jackz/test/Diff_V/BO_vs_GD/NMC532-Gr_BO/combined_plots_%s/VQ Charge cyc %s acq %s idx %s.png'%(str(fs),str(cycnum),str(acq),str(idx)))
    #             annotations = [
    #                 "length x = " +  str(len(dVdQ_cyc)),
    #                 "min x = " + str(min(x)),
    #                 "max x = " + str(max(x))
    #             ]
    #             for i, annotation in enumerate(annotations):
    #                 fig.text(0.1, -0.12 - i * 0.03, annotation, {'color': 'C0', 'fontsize': 13})

    #             fig.text(0.1, -0.12, eq, {'color': 'C0', 'fontsize': 13})

    #             plt.ylim(-0.4,0)
    #             plt.show()
#         plt.savefig('/projectnb/ryanlab/jackz/test/Diff_V/Resistance_order_0/combined_plots_%s/VQ Charge cyc %s acq %s idx %s.png'%(str(fs),str(cycnum),str(acq),str(idx)))
#         plt.figure().clear()
        plt.show()
        print('Discharge')
        plot_DVQ(ka, kc,ba, bc, r*current,cycx_D,cycy_D,cycnum,acq,idx,'Discharge',str(fs))
        print('Charge')
        plot_DVQ(ka, kc,ba, bc, r*current,cycx_C,cycy_C,cycnum,acq,idx,'Charge',str(fs))

     
    #     print('VQ')
    #     print(gain(kc,bc,ka,ba,r)-mod_dvq_gain)

    #     ifc+=1
    #     pbounds = next_pbounds
    #     print(pbounds)
    #     ifc+=1
    #     pbounds = next_pbounds
    #     print(pbounds)
    # # create a binary pickle file 


    f = open("/projectnb/ryanlab/jackz/test/Diff_V/BO_vs_GD/NMC532-Gr_BO/combined_dicts_%s/combined %s %s %s resistance_dict.pkl" %(str(fs), str(cycnum),str(acq),str(idx)),"wb")

    # write the python object (dict) to pickle file
    pickle.dump(res_dict,f)

    # close file
    f.close()

