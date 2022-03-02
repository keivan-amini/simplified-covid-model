#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:38:19 2022

@author: Giulio Colombini
"""

import numpy as np
from   scipy import stats
from   tqdm  import tqdm

# Social activity rate
# The example is a quite strong lockdown 14 days after the introduction of patient zero.
m_test = (np.array([0., 14, 20, np.inf]), np.array([1., .5, .15, .5]))

# Symptomatic fraction
alpha_test = (np.array([0., np.inf]), np.array([.14, .14]))

# Parameters of \rho_E over time
pars_e_test = (np.array([0., np.inf]), np.array([2.,2.]), np.array([.1,.1]))

# Parameters of \rho_I over time
pars_i_test = (np.array([0., np.inf]), np.array([5.5,5.5]), np.array([2.3,2.3]))

# Parameters of \rho_A over time
pars_a_test = pars_i_test

def discrete_gamma(mean, std_dev, min_t = None, max_t = None):
    if (min_t == None) and (max_t == None):
        min_t = np.rint(mean) - np.rint(3 * std_dev)
        max_t = np.rint(mean) + np.rint(5 * std_dev)
    
    RV = stats.gamma(a = (mean/std_dev)**2, scale = (std_dev**2/mean))
    low_i, high_i = np.int32(np.round([min_t, max_t]))
    low_i = max([1, low_i])
    
    c = np.zeros(high_i, dtype = np.double)
    
    for j in range(low_i, high_i):
        c[j] = RV.cdf((j + 1/2)) - RV.cdf((j-1/2))
    c /= c.sum()
    return (c, low_i, high_i)

# Forward buffered convolution
def propagate_forward(t, max_t, donor, acceptors, kernel_tuple, branching_ratios = np.array([1.])):
    kernel, i0, lker = kernel_tuple
    if t + i0 > max_t:
        return
    if t + lker - 1 > max_t:
        k = kernel[i0 : max_t - t + 1]
        lk = len(k)
    else:
        k = kernel[i0:]
        lk = len(k)
    buffer = np.empty(shape = (lk,) + donor.shape)
    for i in range(lk):
        buffer[i] = donor * k[i]
    for a, r in zip(acceptors, branching_ratios):
        a[t + i0 : t + i0 + lk] += r * buffer

def run_simulation(days = 60, dt = 1./24., beta = 1/1.2, N = 886891, m = m_test, 
                   alpha = alpha_test, pars_e = pars_e_test, 
                   pars_i = pars_i_test, pars_a = pars_a_test):
    # Calculate number of iterations
    max_step = int(np.rint(days / dt))
    
    # Initialise compartments and flows memory locations
    S = np.zeros(max_step+1)
    E = np.zeros(max_step+1)
    I = np.zeros(max_step+1)
    A = np.zeros(max_step+1)
    R = np.zeros(max_step+1)
    TOT = np.zeros(max_step+1)
    
    Phi_SE = np.zeros(max_step+1)
    Phi_EI = np.zeros(max_step+1)
    Phi_EA = np.zeros(max_step+1)
    Phi_IR = np.zeros(max_step+1)
    Phi_AR = np.zeros(max_step+1)
    
    # Unpack parameter tuples and rescale them with dt.
    
    m_t    = m[0] / dt
    m_vals = m[1]
    
    m_array= np.array([m_vals[np.searchsorted(m_t, t, side = 'right') - 1] for t in range(max_step+1)])
    
    alpha_t    = alpha[0] / dt
    alpha_vals = alpha[1]
    
    alpha_array= np.array([alpha_vals[np.searchsorted(alpha_t, t, side = 'right') - 1] for t in range(max_step+1)])
    
    # Unpack distribution tuples and generate distributions
    
    rho_e_t      = pars_e[0] / dt
    rho_e_mus    = pars_e[1] / dt
    rho_e_sigmas = pars_e[2] / dt
    
    rho_es = [discrete_gamma(mu, sigma) for mu, sigma in zip(rho_e_mus, rho_e_sigmas)]

    rho_i_t      = pars_i[0] / dt
    rho_i_mus    = pars_i[1] / dt
    rho_i_sigmas = pars_i[2] / dt

    rho_is = [discrete_gamma(mu, sigma) for mu, sigma in zip(rho_i_mus, rho_i_sigmas)]

    rho_a_t      = pars_a[0] / dt
    rho_a_mus    = pars_a[1] / dt
    rho_a_sigmas = pars_a[2] / dt

    rho_as = [discrete_gamma(mu, sigma) for mu, sigma in zip(rho_a_mus, rho_a_sigmas)]
    
    # Add initial population to Susceptibles
    
    S[0]   = N
    TOT[0] = N
    
    # Add patient zero to flow
    
    Phi_SE[0] += 1.
    
    # Intialize indices for distribution selection
    
    cur_rho_e_idx = 0
    cur_rho_i_idx = 0
    cur_rho_a_idx = 0
    
    # Master simulation loop
    for t in tqdm(range(max_step)):
        # Update distribution indices
        cur_rho_e_idx = np.searchsorted(rho_e_t, t, side = 'right') - 1
        cur_rho_i_idx = np.searchsorted(rho_i_t, t, side = 'right') - 1
        cur_rho_a_idx = np.searchsorted(rho_a_t, t, side = 'right') - 1
        
        # Get current parameters
        cur_m     = m_array[t]
        cur_alpha = alpha_array[t]
        
        
        # Evaluate active population
        
        P = S[t] + E[t] + A[t] + R[t]
        
        # Evolve contagion flow
        Phi_SE[t] += beta * cur_m * S[t] * (E[t] + A[t]) * dt / P
        
        # Propagate flows
        
        propagate_forward(t, max_step, Phi_SE[t], [Phi_EI, Phi_EA], rho_es[cur_rho_e_idx],
                          branching_ratios = np.array([cur_alpha, 1 - cur_alpha]))
        propagate_forward(t, max_step, Phi_EI[t], [Phi_IR], rho_is[cur_rho_i_idx],
                          branching_ratios = np.array([1.]))
        propagate_forward(t, max_step, Phi_EA[t], [Phi_AR], rho_as[cur_rho_a_idx],
                          branching_ratios = np.array([1.]))
        
        # Evolve compartments
        
        S[t+1] = S[t] - Phi_SE[t]
        E[t+1] = E[t] + Phi_SE[t] - Phi_EA[t] - Phi_EI[t]
        I[t+1] = I[t] + Phi_EI[t] - Phi_IR[t]
        A[t+1] = A[t] + Phi_EA[t] - Phi_AR[t]
        R[t+1] = R[t] + Phi_IR[t] + Phi_AR[t]
        TOT[t+1] = S[t+1] + E[t+1] + I[t+1] + A[t+1] + R[t+1]
    t = np.array([t for t in range(max_step+1)])
    return (t, S, E, I, A, R, TOT)

t,s,e,i,a,r, tot = run_simulation()
#%% Graphics
from matplotlib import pyplot as plt

dt = 1/24

plt.plot(t * dt,s, label = 'S')
plt.plot(t * dt,e, label = 'E')
plt.plot(t * dt,i, label = 'I')
plt.plot(t * dt,a, label = 'A')
plt.plot(t * dt,r, label = 'R')
# plt.plot(t * dt, tot, label = 'TOT')


plt.legend()
plt.grid(True)
plt.ylim(bottom = 0)
plt.xlim([0, max(t * dt)])
plt.xlabel('Days since patient zero introduction')
plt.ylabel('People')