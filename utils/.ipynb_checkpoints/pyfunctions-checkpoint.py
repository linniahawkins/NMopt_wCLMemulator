import os
import numpy as np
import xarray as xr
import cftime
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import glob
import dask

import gpflow
import tensorflow as tf
from sklearn.metrics import r2_score

utils_dir = os.path.dirname(__file__)

# ============================================================
# ================== general functions =======================

def amean(da):
    #annual mean of monthly data
    m  = da['time.daysinmonth']
    cf = 1/365
    xa = cf*(m*da).groupby('time.year').sum().compute()
    return xa

def amax(da):
    #annual max
    m  = da['time.daysinmonth']
    xa = da.groupby('time.year').max().compute()
    return xa

def gmean(da,la):
    # global mean
    if 'gridcell' in da.dims:
        dim='gridcell'
    else:
        dim=['lat','lon']
    x=(da*la).sum(dim=dim)/la.sum()
    return x.compute()

def fix_time(ds):
    yr0=str(ds['time.year'][0].values)
    nt=len(ds.time)
    ds['time'] = xr.cftime_range(yr0,periods=nt,freq='MS',calendar='noleap') #fix time bug
    return ds

def get_map(da,sgmap=None):
    if not sgmap:
        sgmap=xr.open_dataset(os.path.join(utils_dir,'sgmap_retrain_h0.nc'))
    return da.sel(gridcell=sgmap.cclass).where(sgmap.notnan).compute()

# ==============================================================
# ===================== load PPE  ==============================

def get_files(htape,yr0=1850,yr1=2014):
    d='/glade/campaign/asp/djk2120/PPEn11/transient/hist/'

    #find all files
    fs   = np.array(sorted(glob.glob(d+'*'+htape+'*')))
    yrs  = np.array([int(f.split(htape)[1][1:5]) for f in fs])
    keys = np.array([f.split('.clm2')[0][-7:] for f in fs])
    
    #bump back yr0, if needed
    uyrs=np.unique(yrs)
    yr0=uyrs[(uyrs/yr0)<=1][-1]
    
    #find index to subset files
    ix   = (yrs>=yr0)&(yrs<=yr1)
    yrs  = yrs[ix]
    keys = keys[ix]

    #subset and reshape files
    ny=np.sum(keys=='LHC0000')
    nens = int(len(keys)/ny)
    files = fs[ix].reshape([nens,ny])

    #convert to list of lists
    files = [list(f) for f in files]
    
    return files,np.unique(keys)

def add_params(ds,df,keys):
    mems=df['member'].values
    ix1=0*mems==1
    for key in keys:
        ix1=(ix1)|(mems==key)

    nens=len(ds.ens)    
    ix2=0*np.arange(nens)==1
    for mem in mems:
        ix2=(ix2)|(ds.key==mem)


    params=[]    
    for p in df.keys():
        if p!='member':
            x=xr.DataArray(np.nan+np.zeros(nens),dims='ens')
            x[ix2]=df[p][ix1]
            ds[p]=x
            params.append(p)
    ds['params']=xr.DataArray(params,dims='param')

def get_ds(dvs,htape,yr0=1850,yr1=2014,dropdef=False):
    
    def preprocess(ds):
        return ds[dvs]
    
    #read in the data
    files,keys = get_files(htape,yr0,yr1)
    if dropdef:
        files = files[1:]
        keys  = keys[1:]
    
    ds = xr.open_mfdataset(files,combine='nested',concat_dim=['ens','time'],
                       parallel=True,preprocess=preprocess,decode_timedelta=False)
    
    #fix the time dimension, if needed
    yr0=str(ds['time.year'].values[0])
    if (htape=='h0')|(htape=='h1'):
        ds['time']=xr.cftime_range(yr0,periods=len(ds.time),freq='MS',calendar='noleap')
    
    #add some param info, etc.
    df=pd.read_csv('/glade/campaign/asp/djk2120/PPEn11/csvs/lhc220926.txt')
    ds['key']=xr.DataArray(keys,dims=['ens'])
    add_params(ds,df,keys)
    
    #add landarea info
    la_file = '/glade/u/home/djk2120/clm5ppe/pyth/sparsegrid_landarea.nc'
    la = xr.open_dataset(la_file).landarea  #km2
    ds['la'] = la
    
    #add some extra variables, e.g. lat/lon
    tmp = xr.open_dataset(files[0][0],decode_timedelta=True)
    for v in tmp.data_vars:
        if 'time' not in tmp[v].dims:
            if v not in ds:
                ds[v]=tmp[v]
                
    if htape=='h1':
        ds['pft']=ds.pfts1d_itype_veg
    
    return ds

# ==============================================================
# ===================== Emulation ==============================

def build_kernel_dict(num_params):
    kernel_noise = gpflow.kernels.White(variance=1e-3)
    kernel_matern32 = gpflow.kernels.Matern32(active_dims=range(num_params), variance=10, lengthscales = np.tile(10,num_params))
    kernel_matern52 = gpflow.kernels.Matern52(active_dims=range(num_params),variance=1,lengthscales=np.tile(1,num_params))
    kernel_bias = gpflow.kernels.Bias(active_dims = range(num_params))
    kernel_linear = gpflow.kernels.Linear(active_dims=range(num_params),variance=[1.]*num_params)
    kernel_poly = gpflow.kernels.Polynomial(active_dims = range(num_params),variance=[1.]*num_params)
    kernel_RBF = gpflow.kernels.RBF(active_dims = range(num_params), lengthscales=np.tile(1,num_params))
    
    kernel_dict = {
        0:kernel_linear + kernel_noise,
        1:kernel_RBF,
        2:kernel_RBF + kernel_linear + kernel_noise,
        3:kernel_RBF + kernel_linear + kernel_noise + kernel_bias,
        4:kernel_poly,
        5:kernel_poly + kernel_linear + kernel_noise,
        6:kernel_RBF + kernel_linear + kernel_noise + kernel_bias + kernel_poly,
        7:kernel_matern32,
        8:kernel_matern32+kernel_linear+kernel_noise,
        9:kernel_matern32*kernel_linear+kernel_noise,
        10:kernel_linear*kernel_RBF+kernel_matern32 + kernel_noise
    }
    return kernel_dict


def train_val_save(X_train,X_test,y_train,y_test,kernel,outfile=None,savedir=None):

        model = gpflow.models.GPR(data=(X_train, np.float64(y_train)), kernel=kernel, mean_function=None)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=30))

        # plot validation
        y_pred, y_pred_var = model.predict_y(X_test)
        sd = y_pred_var.numpy().flatten()**0.5

        coef_deter = r2_score(y_test,y_pred.numpy())

        if (savedir):
            print('saving')
            num_params = np.shape(X_train)[1]
            model.predict = tf.function(model.predict_y, input_signature=[tf.TensorSpec(shape=[None, num_params], dtype=tf.float64)])
            tf.saved_model.save(model, savedir)

        if (outfile):
            plt.figure()
            plt.errorbar(y_test, y_pred.numpy().flatten(), yerr=2*sd, fmt="o")
            plt.text(0.02, 0.98, f'R² = {np.round(coef_deter, 2)}',fontsize=10,transform=plt.gca().transAxes,va='top',ha='left')
            plt.text(0.02, 0.93, f'Emulator stdev ≈ {np.round(np.mean(sd), 2)}',fontsize=10,transform=plt.gca().transAxes,va='top',ha='left')
            plt.plot([0,np.max(y_test)],[0,np.max(y_test)],linestyle='--',c='k')
            plt.xlabel('CLM')
            plt.ylabel('Emulated')
            plt.xlim([np.min(y_test)-.5,np.max(y_test)+.5])
            plt.ylim([np.min(y_test)-.5,np.max(y_test)+.5])
            plt.tight_layout()
            plt.savefig(outfile)
    
        return coef_deter, np.mean(sd)


def select_kernel(kernel_dict,X_train,X_test,y_train,y_test):
    stdev = []
    r2 = []
    for k in range(len(kernel_dict)):
        kernel = kernel_dict[k]
        cd, sd = train_val_save(X_train,X_test,y_train,y_test,kernel,outfile=None,savedir=None)
        stdev.append(sd)
        r2.append(cd)
      
    r2_norm = (r2 - np.min(r2)) / (np.max(r2) - np.min(r2))
    std_norm = 1 - (stdev - np.min(stdev)) / (np.max(stdev) - np.min(stdev))

    score = 0.8*r2_norm + 0.2*std_norm
    best_kernel = kernel_dict[np.argmax(score)]
    
    return best_kernel



