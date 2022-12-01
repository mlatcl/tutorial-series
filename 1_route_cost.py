
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from s1_data_prep import plot_path
from cmdstanpy import CmdStanModel

plt.ion(); plt.style.use('seaborn-pastel')
np.random.seed(42)

##############################################################
# Load data

with open('1_data.pkl', 'rb') as file:
    path, ice_concentration = pkl.load(file)

# for efficiency
idxs_for_subselection = np.arange(0, len(path), len(path)//20)
path = path[idxs_for_subselection, :]
ice_concentration = ice_concentration[:, idxs_for_subselection]

"""
path is a numpy array containing (lon, lat) of a path a ship
could take in antarctica.

ice_concentration contains historical sea ice fraction data at
the lon, lats in our path in the month of december from 2012-16,
so there are 31 days * 5yrs = 155 observations
"""

plot_path(path)

##############################################################
# Data exploration

# plot of ice concentration
fig, axs = plt.subplots(5, 1, figsize=(6, 9))
cols = ["#11151c","#212d40","#364156","#7d4e57","#d66853"]
for i in range(5):
    axs[i].plot(ice_concentration[np.arange(31*i, 31*(i + 1)), :].T,
                alpha=0.1, c=cols[i])
    axs[i].set_ylabel(f'{np.arange(2012, 2017)[i]}')
axs[-1].set_xlabel('path index')
fig.suptitle('ice concentration by year')
plt.tight_layout()

# plot of mean ice concentration
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(ice_concentration.mean(axis=0), c=cols[0])
ax.set_xlabel('path index')
ax.set_ylabel('ice conc')
fig.suptitle('mean ice concentration')
plt.tight_layout()

# plot of ice concentration covariance
plt.imshow(np.cov(ice_concentration.T))

# ice concentration residuals
fig, axs = plt.subplots(5, 1, figsize=(6, 9))
idxs_to_plot = np.linspace(0, len(ice_concentration.T) - 1, 5).astype(int)
for i in range(5):
    axs[i].hist((ice_concentration - ice_concentration.mean(axis=0))[:, idxs_to_plot[i]],
            color=cols[i], alpha=0.2)
axs[-1].set_xlabel('residual')
axs[2].set_ylabel('freq')
fig.suptitle('histogram of residuals')
plt.tight_layout()

##############################################################
# Define model

model_code = """
functions {
    real cost_func(vector c) {
        int d = size(c);
        vector[d] v = fmin(14.0, 61*c^2 - 101*c + 46);
        vector[d] r = 255*c^2 - 57*c;
        return mean(0.113*v^2 - 0.132*v + 0.003*r^2 + 0.042*r + 6);
    }
}
data {
    int n;  // num observations of ice conc (days)
    int d;  // num points on path
    
    array[d] vector[2] coords;  // coords of path
    array[n] vector[d] ice_c;   // concentration
}
parameters {
    vector<lower=0, upper=1>[d] mu;
    vector<lower=0.01, upper=1>[d] sp_sd;
    real<lower=0, upper=1> noise_v;
    real<lower=0, upper=250> lscale;
}
transformed parameters {
    cholesky_factor_cov[d] sigma_factor = cholesky_decompose(
        diag_matrix(sp_sd) * gp_matern52_cov(coords, 1.0, lscale) * diag_matrix(sp_sd) +
        diag_matrix(rep_vector(noise_v, d))
    );
}
model {
    ice_c ~ multi_normal_cholesky(mu, sigma_factor);
}
generated quantities {
    vector[d] ice_c_sample = fmax(fmin(multi_normal_cholesky_rng(mu, sigma_factor), 1.0), 0.0);
    real cost = cost_func(ice_c_sample);
}
"""

with open('ice_concentration_model.stan', 'w') as f:
    f.write(model_code)

pd.Series(dict(
    n=len(ice_concentration),
    d=len(ice_concentration.T),
    coords=path,
    ice_c=ice_concentration
)).to_json('data.json')

model = CmdStanModel(stan_file='ice_concentration_model.stan')
map_estimate = model.optimize(data='data.json', seed=42)

plt.plot(map_estimate.stan_variable('mu'))

posterior = model.sample(data='data.json', seed=42, iter_warmup=10000, iter_sampling=1000)

# traceplot
plt.plot(posterior.stan_variable('mu'))

def cost_func(c):
    v = 61*c**2 - 101*c + 46
    v[v >= 14] = 14
    r = 255*c**2 - 57*c
    return np.mean(0.113*v**2 - 0.132*v + 0.003*r**2 + 0.042*r + 6, axis=-1)

plt.hist(posterior.stan_variable('cost'), bins=100, alpha=0.5, label='sim')
plt.hist(cost_func(ice_concentration), alpha=0.5, label='data')
plt.xlabel('log cost')
plt.ylabel('freq')
plt.legend()
plt.tight_layout()
