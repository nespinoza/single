from scipy.signal import medfilt
from matplotlib import rc,rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import pymultinest
import batman
import george
from george import kernels
import numpy as np
import argparse
try:
    import matplotlib
    ShowPlots = True
except:
    print 'No matplotlib, so no final plots!'
    ShowPlots = False
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('-lcfile', default=None)
parser.add_argument('-sdmean', default=None)
parser.add_argument('-sdsigma', default=None)
parser.add_argument('-t0mean', default=None)
parser.add_argument('-t0sigma', default=None)
parser.add_argument('-Pmin', default=1.)
parser.add_argument('-Pmax', default=1000.)
parser.add_argument('--resampling', dest='resampling', action='store_true')
parser.add_argument('--circular', dest='circular', action='store_true')
parser.set_defaults(resampling=False)
parser.set_defaults(circular=False)
parser.add_argument('-nresampling', default=20)
parser.add_argument('-texp', default=0.020434)
parser.add_argument('-nlive', default=500)
parser.add_argument('-ldlaw', default='linear')
args = parser.parse_args()
############ OPTIONAL INPUTS ###################
filename = args.lcfile
sd_mean,sd_sigma = args.sdmean, args.sdsigma
if sd_mean is None:
    stellar_density = False
else:
    sd_mean,sd_sigma = np.double(sdmean),np.double(sdsigma)
    stellar_density = True
P_low,P_up = np.double(args.Pmin),np.double(args.Pmax)
texp = np.double(args.texp)
RESAMPLING = args.resampling
CIRCULAR = args.circular
#if (args.resampling).lower() == 'true':
#    RESAMPLING = True
#else:
#    RESAMPLING = False
NRESAMPLING = int(args.nresampling)
n_live_points = int(args.nlive)
ld_law = args.ldlaw

# Extract data:
t,flux = np.loadtxt(filename,unpack=True,usecols=(0,1))
ndata = len(t)

# Count time from first time sample, to simplify parameter estimation:
tzero = t[0]
t = t - tzero

print 'RESAMPLING:',RESAMPLING,' LD LAW:',ld_law
################################################
# Transit model priors:
# Rp/Rs
rp_low,rp_up = 0.001,0.5 
# Impact parameter:
b_low,b_up = 0.,1.+rp_up
# a/Rs
aR_low,aR_up =  1.0,300.

# Define priors for q1 and q2, if you have any. If you do, prior is assumed truncated normal 
# between 0 and 1. If you don't, set all numbers to 0. This assumes you want a uniform distribution 
# on q1 and q2:
q10 = 0.
q1_sigma0 = 0.
q20 = 0.
q2_sigma0 = 0.

# Priors on t0:
t00,t0_sigma0 = np.double(args.t0mean)-tzero,np.double(args.t0sigma)
print 'Priors on t0:',t00+tzero,t0_sigma0

###############################################
G = 6.67408e-11 # mks

# Cook GP:
kernel = 1.*george.kernels.ExpSquaredKernel(metric=1.0)
jitter = george.modeling.ConstantModel(np.log((10*1e-6)**2.))
# Wrap GP object to compute likelihood
gp = george.GP(kernel, mean=0.0,fit_mean=False,white_noise=jitter,fit_white_noise=True)
gp.compute(t)

def get_quantiles(dist,alpha = 0.68, method = 'median'):
    """ 
    get_quantiles function

    DESCRIPTION

        This function returns, in the default case, the parameter median and the error% 
        credibility around it. This assumes you give a non-ordered 
        distribution of parameters.

    OUTPUTS

        Median of the parameter,upper credibility bound, lower credibility bound

    """
    ordered_dist = dist[np.argsort(dist)]
    param = 0.0 
    # Define the number of samples from posterior
    nsamples = len(dist)
    nsamples_at_each_side = int(nsamples*(alpha/2.)+1)
    if(method == 'median'):
       med_idx = 0 
       if(nsamples%2 == 0.0): # Number of points is even
          med_idx_up = int(nsamples/2.)+1
          med_idx_down = med_idx_up-1
          param = (ordered_dist[med_idx_up]+ordered_dist[med_idx_down])/2.
          return param,ordered_dist[med_idx_up+nsamples_at_each_side],\
                 ordered_dist[med_idx_down-nsamples_at_each_side]
       else:
          med_idx = int(nsamples/2.)
          param = ordered_dist[med_idx]
          return param,ordered_dist[med_idx+nsamples_at_each_side],\
                 ordered_dist[med_idx-nsamples_at_each_side]

def get_phases(t,P,t0):
    """ 
    Given input times, a period (or posterior dist of periods)
    and time of transit center (or posterior), returns the 
    phase at each time t.
    """
    if type(t) is not float:
        phase = ((t - np.median(t0))/np.median(P)) % 1 
        ii = np.where(phase>=0.5)[0]
        phase[ii] = phase[ii]-1.0
    else:
        phase = ((t - np.median(t0))/np.median(P)) % 1 
        if phase>=0.5:
            phase = phase - 1.0 
    return phase

from scipy.stats import norm,beta,truncnorm
def transform_uniform(x,a,b):
    return a + (b-a)*x

def transform_loguniform(x,a,b):
    la=np.log(a)
    lb=np.log(b)
    return np.exp(la + x*(lb-la))

def transform_normal(x,mu,sigma):
    return norm.ppf(x,loc=mu,scale=sigma)

def transform_beta(x,a,b):
    return beta.ppf(x,a,b)

def transform_truncated_normal(x,mu,sigma,a=0.,b=1.):
    ar, br = (a - mu) / sigma, (b - mu) / sigma
    return truncnorm.ppf(x,ar,br,loc=mu,scale=sigma)

def prior(cube, ndim, nparams):
    # Prior on "rp" is uniform:
    cube[0] = transform_uniform(cube[0],rp_low,rp_up)
    # Same for aR:
    cube[1] = transform_loguniform(cube[1],aR_low,aR_up)
    # And impact parameter:
    cube[2] = transform_uniform(cube[2],b_low,b_up)
    # And t0:
    cube[3] = transform_normal(cube[3],t00,t0_sigma0)
    # Uniform on the transformed LD coeffs (q1 and q2):
    if q10 == 0. and q20 == 0.:
        cube[4] = transform_uniform(cube[4],0.,1.)
        if ld_law != 'linear':
            cube[5] = transform_uniform(cube[5],0.,1.)
            nprior = 5
        else:
            nprior = 4
    else:
        cube[4] = utils.transform_truncated_normal(cube[4],q10,q1_sigma0)
        if ld_law != 'linear':
            cube[5] = utils.transform_truncated_normal(cube[5],q20,q2_sigma0)
            nprior = 5
        else:
            nprior = 4
    # Prior for flux normalization constant:
    nprior = nprior + 1
    cube[nprior] = transform_uniform(cube[nprior],-2.0,2.0)
    # GP parameters. First sigma of the jitter term in ppm, then scale parameter (in days), then GP sigma, also in ppm:
    nprior = nprior + 1
    cube[nprior] = transform_loguniform(cube[nprior],1.,10000)
    nprior = nprior + 1
    cube[nprior] = transform_loguniform(cube[nprior],1./(24.*60.),1.)
    nprior = nprior + 1
    cube[nprior] = transform_loguniform(cube[nprior],1.,10000)
    # Eccentricity and Omega if CIRCULAR is False:
    if not CIRCULAR:
        nprior = nprior + 1
        cube[nprior] = transform_uniform(cube[nprior],0.,1.)
        # Omega:
        nprior = nprior + 1
        cube[nprior] = transform_uniform(cube[nprior],0.,360.)
    # Period:
    nprior = nprior + 1 
    cube[nprior] = transform_loguniform(cube[nprior],P_low,P_up)

def reverse_ld_coeffs(ld_law, q1, q2): 
    if ld_law == 'quadratic':
        coeff1 = 2.*np.sqrt(q1)*q2
        coeff2 = np.sqrt(q1)*(1.-2.*q2)
    elif ld_law=='squareroot':
        coeff1 = np.sqrt(q1)*(1.-2.*q2)
        coeff2 = 2.*np.sqrt(q1)*q2
    elif ld_law=='logarithmic':
        coeff1 = 1.-np.sqrt(q1)*q2
        coeff2 = 1.-np.sqrt(q1)
    elif ld_law == 'linear':
        return q1,q2
    return coeff1,coeff2

def init_batman(t,law):
    """  
    This function initializes the batman code.
    """
    params = batman.TransitParams()
    params.t0 = 0. 
    params.per = 1. 
    params.rp = 0.1
    params.a = 15.
    params.inc = 87.
    params.ecc = 0. 
    params.w = 90.
    if law == 'linear':
        params.u = [0.5]
    else:
        params.u = [0.1,0.3]
    params.limb_dark = law
    m = batman.TransitModel(params,t)
    return params,m

def get_transit_model(t,t0,P,p,a,inc,q1,q2,ecc,omega,ld_law):
    params,m = init_batman(t,law=ld_law)
    coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
    params.t0 = t0
    params.per = P
    params.rp = p
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = omega
    if ld_law == 'linear':
        params.u = [coeff1]
    else:
        params.u = [coeff1,coeff2]
    return m.light_curve(params)

if RESAMPLING:
    tin = np.array([])
    for i in range(len(t)):
        for j in range(NRESAMPLING):
            jj = j+1 
            tin = np.append(tin,t[i] + (jj - (NRESAMPLING + 1)/2.)*(texp/NRESAMPLING))
else:
    tin = np.double(t)

# Batman doesn't like non-float64 numbers for some reason:
tin = tin.astype('float64') 

params,m = init_batman(tin,law=ld_law)
def transit_model(rp,aR,b,t0,q1,q2,ecc,omega,P,tt=None):
    # Extract parameters:
    if ld_law != 'linear':
        coeff1,coeff2 = reverse_ld_coeffs(ld_law,q1,q2)
    else:
        coeff1 = q1

    ecc_factor = (1. + ecc*np.sin(omega * np.pi/180.))/(1. - ecc**2)
    inc_inv_factor = (b/aR)*ecc_factor
    # Check that b and b/aR are in physically meaningful ranges:
    if b>1.+rp or inc_inv_factor >=1.:
        return np.ones(len(t))

    # Compute inclination of the orbit:
    inc = np.arccos(inc_inv_factor)*180./np.pi
    if tt is None:
        params.t0 = t0
        params.per = P
        params.rp = rp
        params.a = aR
        params.inc = inc
        params.ecc = ecc
        params.w = omega
        if ld_law != 'linear':
            params.u = [coeff1,coeff2]
        else:
            params.u = [coeff1]
        model = m.light_curve(params)
        if RESAMPLING:
            model_out = np.zeros(len(t))
            for i in range(len(t)):
                model_out[i] = np.mean(model[i*NRESAMPLING:NRESAMPLING*(i+1)])
            return model_out
        return model
    else:
        if RESAMPLING:
            ttin = np.array([])
            for i in range(len(tt)):
                for j in range(NRESAMPLING):
                    jj = j+1
                    ttin = np.append(ttin,tt[i] + (jj - (NRESAMPLING + 1)/2.)*(texp/NRESAMPLING))
        else:
            ttin = tt 
        if ld_law != 'linear':
            model = get_transit_model(ttin,t0,P,rp,aR,inc,q1,q2,ecc,omega,ld_law)
        else:
            model = get_transit_model(ttin,t0,P,rp,aR,inc,q1,0.0,ecc,omega,ld_law)
        if RESAMPLING:
            model_out = np.zeros(len(tt))
            for i in range(len(tt)):
                model_out[i] = np.mean(model[i*NRESAMPLING:NRESAMPLING*(i+1)])
            return model_out
        return model

def loglike(cube, ndim, nparams):
    # Extract parameters:
    if ld_law != 'linear':
        if not CIRCULAR:
            rp,aR,b,t0,q1,q2,f0,jitter,scale,gpsigma,ecc,omega,P = cube[0],cube[1],cube[2],cube[3],cube[4],\
                                               cube[5],cube[6],cube[7],cube[8],cube[9],cube[10],cube[11],cube[12]
        else:
            rp,aR,b,t0,q1,q2,f0,jitter,scale,gpsigma,P = cube[0],cube[1],cube[2],cube[3],cube[4],\
                                               cube[5],cube[6],cube[7],cube[8],cube[9],cube[10]
            ecc,omega = 0.,90.
    else:
        if not CIRCULAR:
            rp,aR,b,t0,q1,f0,jitter,scale,gpsigma,ecc,omega,P = cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],\
                                                                     cube[6],cube[7],cube[8],cube[9],cube[10],cube[11]
        else:
            rp,aR,b,t0,q1,f0,jitter,scale,gpsigma,P = cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],\
                                                                     cube[6],cube[7],cube[8],cube[9]
            ecc,omega = 0.,90.
        q2 = 0. 

    # Get residuals:
    residuals = flux - (f0 + transit_model(rp,aR,b,t0,q1,q2,ecc,omega,P))
    # Update parameter vector:
    gp.set_parameter_vector(np.array([np.log((jitter*1e-6)**2),np.log((gpsigma*1e-6)**2),np.log((scale**2))])) 
    # Evaluate the GP log-likelihood:
    loglikelihood = gp.log_likelihood(residuals)

    # Now, if stellar density is off, just return this. If it is on, calculate the corresponding log-likelihood for the density data:
    if (not stellar_density):
        return loglikelihood
    if stellar_density:
        model = ((3.*np.pi)/(G*(P*(24.*3600.0))**2))*(aR)**3
        return loglikelihood - 0.5*(np.log(2*np.pi) + 2.*np.log(sd_sigma) + ((model-sd_mean)/sd_sigma)**2)

if ld_law != 'linear':
    n_params = 13
else:
    n_params = 12
if CIRCULAR:
    n_params = n_params - 2
if os.path.exists('mnest_out_folder'):
    counter = 0
    while True:
        if not os.path.exists('mnest_out_folder_'+str(counter)):
            os.mkdir('mnest_out_folder_'+str(counter))
            out_mnest_folder = 'mnest_out_folder_'+str(counter)
            break
        else:
            counter = counter + 1
else:
    os.mkdir('mnest_out_folder')
    out_mnest_folder = 'mnest_out_folder'

if not stellar_density:
    out_file = out_mnest_folder+'/out_multinest_transit_GP'+ld_law+'_'
    out_pickle_name = 'POSTERIOR_SAMPLES_GP'+ld_law+'.pkl'
else:
    out_file = out_mnest_folder+'/out_multinest_transit_GP'+ld_law+'_stellar_density_'
    out_pickle_name = 'POSTERIOR_SAMPLES_GP'+ld_law+'_stellar_density.pkl'

if not os.path.exists(out_pickle_name):
    # Run MultiNest:
    pymultinest.run(loglike, prior, n_params, n_live_points = n_live_points,outputfiles_basename=out_file, resume = False, verbose = True)
    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params = n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    
    posterior_samples = output.get_equal_weighted_posterior()[:,:-1]
    a_lnZ = output.get_stats()['global evidence']
    logZ = (a_lnZ / np.log(10))
    out = {}
    out['posterior_samples'] = {} 
    out['posterior_samples']['p'] = posterior_samples[:,0]
    out['posterior_samples']['aR'] = posterior_samples[:,1]
    out['posterior_samples']['b'] = posterior_samples[:,2]
    posterior_samples[:,3] = posterior_samples[:,3] + tzero
    out['posterior_samples']['t0'] = posterior_samples[:,3]
    # Transformed LD coeffs (q1 and q2):
    out['posterior_samples']['q1'] = posterior_samples[:,4]
    if ld_law != 'linear': 
        out['posterior_samples']['q2'] = posterior_samples[:,5]
        nprior = 5
    else:
        nprior = 4
    nprior = nprior + 1
    out['posterior_samples']['norm_constant'] = posterior_samples[:,nprior]
    nprior = nprior + 1
    out['posterior_samples']['jitter'] = posterior_samples[:,nprior]
    nprior = nprior + 1
    out['posterior_samples']['scale'] = posterior_samples[:,nprior]
    nprior = nprior + 1
    out['posterior_samples']['gpsigma'] = posterior_samples[:,nprior]
    if not CIRCULAR:
        # Eccentricity:
        nprior = nprior + 1
        out['posterior_samples']['ecc'] = posterior_samples[:,nprior]
        # Omega:
        nprior = nprior + 1
        out['posterior_samples']['omega'] = posterior_samples[:,nprior]
    # Period:
    nprior = nprior + 1
    out['posterior_samples']['P'] = posterior_samples[:,nprior]
    out['lnZ'] = a_lnZ
    pickle.dump(out,open(out_pickle_name,'wb'))
else:
    out = pickle.load(open(out_pickle_name,'rb'))
    posterior_samples = np.zeros([len(out['posterior_samples']['p']),n_params])
    posterior_samples[:,0] = out['posterior_samples']['p']
    posterior_samples[:,1] = out['posterior_samples']['aR'] 
    posterior_samples[:,2] = out['posterior_samples']['b'] 
    posterior_samples[:,3] = out['posterior_samples']['t0']
    # Transformed LD coeffs (q1 and q2):
    posterior_samples[:,4] = out['posterior_samples']['q1']
    if ld_law != 'linear': 
        posterior_samples[:,5] = out['posterior_samples']['q2']
        nprior = 5 
    else:
        nprior = 4 
    nprior = nprior + 1 
    posterior_samples[:,nprior] = out['posterior_samples']['norm_constant'] 
    nprior = nprior + 1 
    posterior_samples[:,nprior] = out['posterior_samples']['jitter'] 
    nprior = nprior + 1 
    posterior_samples[:,nprior] = out['posterior_samples']['scale'] 
    nprior = nprior + 1 
    posterior_samples[:,nprior] = out['posterior_samples']['gpsigma']
    if not CIRCULAR:
        # Eccentricity:
        nprior = nprior + 1 
        posterior_samples[:,nprior] = out['posterior_samples']['ecc'] 
        # Omega:
        nprior = nprior + 1 
        posterior_samples[:,nprior] =out['posterior_samples']['omega'] 
    # Period:
    nprior = nprior + 1 
    posterior_samples[:,nprior] = out['posterior_samples']['P']

# Convert times back to normal:
t = t + tzero

if ShowPlots:
    sns.set_context("talk")
    sns.set_style("ticks")
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    rcParams['axes.linewidth'] = 1.2
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['lines.markeredgewidth'] = 1
    bin_kwargs = {"zorder":5}
    model_kwargs = {"zorder":0}
    error_kwargs = {"zorder":10}
    t0 = np.median(out['posterior_samples']['t0'])
    tmodel = np.linspace(np.min(t),np.max(t),1000)

    # Perform "detrending":
    theta = np.median(posterior_samples,axis=0)
    # Extract parameters:
    if ld_law != 'linear':
        if not CIRCULAR:
            rp,aR,b,t0,q1,q2,f0,jitter,scale,gpsigma,ecc,omega,P = theta
        else:
            rp,aR,b,t0,q1,q2,f0,jitter,scale,gpsigma,P = theta
            ecc,omega = 0.,90.
    else:
        if not CIRCULAR:
            rp,aR,b,t0,q1,f0,jitter,scale,gpsigma,ecc,omega,P = theta
        else:
            rp,aR,b,t0,q1,f0,jitter,scale,gpsigma,P = theta
            ecc,omega = 0.,90.
        q2 = 0.
  
    # Substract f0 to flux:
    flux = flux - f0
    # Compute transit model:
    model = transit_model(rp,aR,b,t0-tzero,q1,q2,ecc,omega,P)
    # Get residuals:
    residuals = flux - model
    # GP prediction: 
    gp.set_parameter_vector(np.array([np.log((jitter*1e-6)**2),np.log((gpsigma*1e-6)**2),np.log((scale**2))]))
    pred_mean, pred_var = gp.predict(residuals, t-tzero, return_var=True)
    pred_mean_model, pred_var_model = gp.predict(residuals, tmodel-tzero, return_var=True)
    # "Detrend" flux:
    flux = flux - pred_mean
    fout = open('transit_GP'+ld_law+'_ecc_data.dat','w')
    fout.write('# Time \t Transit \t Trend model \t Trend sigma\n')
    # Save model for further plotting:
    for i in range(len(t)):
        fout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n'.format(t[i],model[i],(pred_mean[i] + f0),np.sqrt(pred_var[i])))
    fout.close()
    matplotlib.rcParams.update({'font.size':12})
    if RESAMPLING:
        matplotlib.pyplot.errorbar((t-t0)*24.,(flux-1.0)*1e6,yerr=np.ones(len(t))*jitter,fmt='.',color='black',elinewidth=1)
    else:
        matplotlib.pyplot.errorbar((t-t0)*24.,(flux-1.0)*1e6,yerr=np.ones(len(t))*jitter,fmt='.',color='black',alpha=0.5,markersize=2,elinewidth=1)
    nsample = 250
    idx = np.random.choice(len(out['posterior_samples']['t0']),nsample,replace=False)
    mm = np.array([])
    for i in range(nsample):
        if ld_law != 'linear':
            if not CIRCULAR:
                rp,aR,b,t0,q1,q2,f0,jitter,scale,gpsigma,ecc,omega,P = [out['posterior_samples']['p'][idx[i]],out['posterior_samples']['aR'][idx[i]],\
                       out['posterior_samples']['b'][idx[i]],\
                       out['posterior_samples']['t0'][idx[i]],out['posterior_samples']['q1'][idx[i]],out['posterior_samples']['q2'][idx[i]],\
                       out['posterior_samples']['norm_constant'][idx[i]],out['posterior_samples']['jitter'][idx[i]],out['posterior_samples']['scale'][idx[i]],\
                       out['posterior_samples']['gpsigma'][idx[i]],\
                       out['posterior_samples']['ecc'][idx[i]],out['posterior_samples']['omega'][idx[i]],out['posterior_samples']['P'][idx[i]]]
            else:
                rp,aR,b,t0,q1,q2,f0,jitter,scale,gpsigma,P = [out['posterior_samples']['p'][idx[i]],out['posterior_samples']['aR'][idx[i]],\
                       out['posterior_samples']['b'][idx[i]],\
                       out['posterior_samples']['t0'][idx[i]],out['posterior_samples']['q1'][idx[i]],out['posterior_samples']['q2'][idx[i]],\
                       out['posterior_samples']['norm_constant'][idx[i]],out['posterior_samples']['jitter'][idx[i]],out['posterior_samples']['scale'][idx[i]],\
                       out['posterior_samples']['gpsigma'][idx[i]],out['posterior_samples']['P'][idx[i]]]
                ecc,omega = 0.,90.
        else:
            if not CIRCULAR:
                rp,aR,b,t0,q1,f0,jitter,scale,gpsigma,ecc,omega,P = [out['posterior_samples']['p'][idx[i]],out['posterior_samples']['aR'][idx[i]],\
                       out['posterior_samples']['b'][idx[i]],\
                       out['posterior_samples']['t0'][idx[i]],out['posterior_samples']['q1'][idx[i]],\
                       out['posterior_samples']['norm_constant'][idx[i]],out['posterior_samples']['jitter'][idx[i]],out['posterior_samples']['scale'][idx[i]],\
                       out['posterior_samples']['gpsigma'][idx[i]],\
                       out['posterior_samples']['ecc'][idx[i]],out['posterior_samples']['omega'][idx[i]],out['posterior_samples']['P'][idx[i]]]
            else:
                rp,aR,b,t0,q1,f0,jitter,scale,gpsigma,P = [out['posterior_samples']['p'][idx[i]],out['posterior_samples']['aR'][idx[i]],\
                       out['posterior_samples']['b'][idx[i]],\
                       out['posterior_samples']['t0'][idx[i]],out['posterior_samples']['q1'][idx[i]],\
                       out['posterior_samples']['norm_constant'][idx[i]],out['posterior_samples']['jitter'][idx[i]],out['posterior_samples']['scale'][idx[i]],\
                       out['posterior_samples']['gpsigma'][idx[i]], out['posterior_samples']['P'][idx[i]]]
                ecc,omega = 0.,90. 
            q2 = 0.
        tm = transit_model(rp,aR,b,t0,q1,q2,ecc,omega,P,tt=tmodel)
        #print 'tm.shape:',tm.shape
        if i == 0:
            mm = tm
        else:
            try:
                mm = np.vstack((mm,tm))
            except:
                continue
    if RESAMPLING:
        plt.xlim([-7,7])
    else:
        plt.xlim([np.min((t-t0)*24.-0.1),np.max((t-t0)*24.)+0.1])
    model_median = (np.median(mm,axis=0)-1.0)*1e6
    model_down = np.zeros(len(model_median))
    model_up = np.zeros(len(model_median))
    for i in range(mm.shape[1]):
        med,up,down = get_quantiles(mm[:,i]) 
        model_up[i],model_down[i] = (up-1.0)*1e6,(down-1.0)*1e6
        
    plt.plot((tmodel-t0)*24.,model_median,color='black',linewidth=2)
    plt.fill_between((tmodel-t0)*24.,model_down,model_up,color='cornflowerblue',alpha=0.5)
    matplotlib.pyplot.ylabel('Relative flux (ppm)')
    matplotlib.pyplot.xlabel('Time from transit center (hours)')
    plt.tight_layout()
    matplotlib.pyplot.savefig('transit_'+ld_law+'.pdf')
    fout = open('transit_'+ld_law+'.dat','w')
    fout.write('# Time \t Transit \t Trend model \t Trend sigma\n')
    # Save model for further plotting:
    for i in range(len(model_median)):
        fout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n'.format(tmodel[i],model_median[i],(pred_mean_model[i] + f0)*1e6,np.sqrt(pred_var_model[i])*1e6))
    fout.close()
os.system('rm -r '+out_mnest_folder)
print 'Done!'
