import sys,numpy as np,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from continual import fit_gmm
from model import log_density
from runtime import atomic_json,ROOT
from scipy.special import logsumexp
r=np.random.RandomState(812);x=np.r_[r.normal(-4,.5,(2000,4)),r.normal(4,.7,(2000,4))];fit=fit_gmm(x,2,np.full(4,1e-6),813);means=np.sort(fit['means'][:,0]);assert abs(means[0]+4)<.1 and abs(means[1]-4)<.1
h=torch.tensor(x[:10],dtype=torch.float32);mu=torch.tensor(fit['means'],dtype=torch.float32);var=torch.tensor(fit['variances'],dtype=torch.float32);weights=torch.tensor(np.log(fit['weights']),dtype=torch.float32);actual=log_density(h,mu,var,weights).numpy();expected=logsumexp(np.log(fit['weights'])[None]-.5*np.sum((x[:10,None,:]-fit['means'])**2/fit['variances']+np.log(fit['variances'])+np.log(2*np.pi),-1),-1);error=float(np.max(abs(actual-expected)));assert error<1e-4
atomic_json(ROOT/'tests/gmm_contracts.json',dict(passed=True,mean_recovery=means.tolist(),density_max_abs=error,converged=fit['converged']))
print('Density EM and logpdf contracts passed')
