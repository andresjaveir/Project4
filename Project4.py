# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 16:29:18 2022

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

np.random.seed()

def pbc(x, L): ## Periodic boundary conditions
    x[(x>L/2)] -= L
    x[(x<-L/2)] += L
    return x

def ji_t_2D(ji, tau, dt, D, N): ## Evolution of the Non-Markovian Noise
    
    return ji + ((-1)/tau)*ji*dt + np.sqrt(
        (2*D*dt/(tau**2)))*np.random.normal(0, 1, (N, 2))

def v_t(v, F, dt): ## Evolution of the velocity
    
    return v + F*dt

L = 100 ## Box side
N = 10000 ## Number of particles
D = 1 ## D
dt = 0.01 ## Time step
tau = np.array([1, 10, 100]) ## Different values of tau
ji = np.zeros((len(tau), N, 2)) ## We set the array for the non-markovian noise
for k in range(len(tau)):
    ji[k] = np.ones((N, 2))*np.sqrt(D/tau[k]) ## We set the noise
ji_i = copy.deepcopy(ji) ## Copy the initial configurations
ri = np.zeros((N, 2)) ## Array for the initial positions (all set at the origin (0,0))
r = np.zeros((len(tau), N, 2)) ## Array for the positions
Nsteps = 700  ## Number of steps
MSD = np.zeros((3, Nsteps)) ## Array for the MSD
corr = np.zeros((3, Nsteps)) ## Array for the correlation
trajectory = np.zeros((3, Nsteps, 2)) ## Array for the typical trajectory

for step in range(Nsteps): ## We begin the simulation
    
    corr[0][step] = np.mean(ji[0]*ji_i)*2 ## Calculate the correlation for each tau
    r[0] = pbc(r[0] + ji[0]*dt, L) # Evolve the position of the particles for each tau
    ji[0] = ji_t_2D(ji[0], tau[0], dt, D, N) ## Evolve the noise for each tau
    MSD[0][step] = np.mean((pbc(r[0]-ri, L))**2)*2 ## Calculate the MSD for each tau
    trajectory[0][step] = r[0, 0] ## We save the position, to plot the trajectory
    
    corr[1][step] = np.mean(ji[1]*ji_i)*2
    r[1] = pbc(r[1] + ji[1]*dt, L)
    ji[1] = ji_t_2D(ji[1], tau[1], dt, D, N)
    MSD[1][step] = np.mean((pbc(r[1]-ri, L))**2)*2
    trajectory[1][step] = r[1, 0]
        
    corr[2][step] = np.mean(ji[2]*ji_i)*2
    r[2] = pbc(r[2] + ji[2]*dt, L)
    ji[2] = ji_t_2D(ji[2], tau[2], dt, D, N)
    MSD[2][step] = np.mean((pbc(r[2]-ri, L))**2)*2
    trajectory[2][step] = r[2, 0]
    
t = np.linspace(0, (Nsteps-1)*dt, Nsteps) ## The array of time
f = np.zeros((3, Nsteps)) ## The array for the theoretical MSD
g = np.zeros((3,  Nsteps)) ## The array for the theoretical correlation
for i in range(len(f)):
    f[i] = 4*D*(t+tau[i]*(np.exp(-t/tau[i])-1))
    g[i] = np.sqrt(D/tau[i])*np.exp(-t/tau[i])

colors = ["greenyellow", "yellowgreen", "g"] ## Colors to plot

plt.figure(figsize=(8,6)) ## Plot the trajectories
for i in range(len(r)):
    plt.title('Typical trajectory', fontsize = 16)
    plt.plot(trajectory[i, :, 0], trajectory[i, :, 1], label = '\u03C4=%i'%tau[i], color = colors[i])
    plt.xlabel('x', fontsize = 14)
    plt.ylabel('y', fontsize = 14)
    plt.legend(loc='best', fontsize = 14)
    plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
    #plt.savefig('trajectory.png', dpi=600)

for i in range(len(r)): ## Plot the MSDs
    plt.figure(figsize=(8,6))
    plt.title('MSD for \u03C4=%i'%tau[i], fontsize = 16)
    plt.plot(t, MSD[i], label = 'MSD', color = 'g')
    plt.plot(t, f[i], label = 'Theory', color = 'lime')
    plt.xlabel('t/(s)', fontsize = 14)
    plt.legend(loc='best', fontsize = 14)
    plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
    #plt.savefig('MSD_2_%i'%tau[i], dpi=600)

for i in range(len(ji)): ## Plot the correlation
    plt.figure(figsize=(8,6))
    plt.title('Correlation function for \u03C4=%i'%tau[i], fontsize = 16)
    plt.plot(t, corr[i], color = 'g', label = '\u03C4=%i'%tau[i])
    plt.plot(t, g[i], color = 'lime', label = 'Theory')
    plt.yscale('log')
    plt.xlabel('t/(s)', fontsize = 14)
    plt.legend(loc='best', fontsize = 14)
    plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlim(0,3)
    #plt.savefig('Corr_3_%i'%tau[i], dpi=600)
    

#%%
def ji_t_1D(ji, tau, dt, D, N): ## Redefine the evolution of the Non-Markovian 
                                # noise, now for 1 dimension
    return ji + ((-1)/tau)*ji*dt + np.sqrt(
        (2*D*dt/(tau**2)))*np.random.normal(0, 1, N)

np.random.seed()
L = 1000 ## Box side
N = 10000 ## Number of particles
D = 1 # D
dt = 0.01 ## Time-step
tau = 10 ## Tau
f = np.array([0, 0.001, 0.01, 0.1, 1]) ## Module of the forces
Nsteps = 700 ## Number of steps
F = np.zeros((len(f), N)) ## Array of the forces
xi = np.zeros(N) ## Array of the Non-Markovian noise
x = np.zeros((len(f), N)) ## Array of the positions
MSD = np.zeros((10, len(f), Nsteps)) ## Array for the MSD (10 repetitions)
Disp = np.zeros((10, len(f), Nsteps)) ## Array for the displacements (10 rep)
MSD_mean = np.zeros((len(f), Nsteps)) ## Array for the mean of the  MSD
Disp_mean = np.zeros((len(f), Nsteps)) ## Array for the mean of the displ
ji = np.ones((len(f), N))*np.sqrt(D/tau) ## Array for the noise
v = np.zeros((len(f), N)) ## Array for the velocities
e = np.zeros((len(f), N)) ## Array for the sign of the force on each particle

for s in range(10): ## We begin the simulation (we average 10 simulations)
    
    x = np.zeros((len(f), N)) ## We reset the position for each simulation
    ji = np.ones((len(f), N))*np.sqrt(D/tau) ## We reset the noise
    for i in range(len(F)): ## We reset the sign of thforces 
        e[i] = np.random.choice((-1,1),N)
        F[i] = f[i]*e[i]
    
    for step in range(Nsteps): ## The simulation begins
        for k in range(len(f)): ## For each value of the force
            x[k] = pbc(x[k] + ji[k]*dt + v[k]*dt, L) ## We evolve the position
            v[k] = v_t(v[k], F[k], dt) ## Evolution of the velocity
            ji[k] = ji_t_1D(ji[k], tau, dt, D, N) ## Evolution of the noise
            MSD[s][k][step] = np.mean((pbc(x[k]-xi, L))**2)*2 ## We save the MSD
            Disp[s][k][step] = np.mean(e[k]*(pbc((x[k]-xi), L)))*2 ## We save the disp

for i in range(len(f)): ## We average the simulations
    for k in range(10):
        MSD_mean[i] += MSD[k][i]/10
        Disp_mean[i] += Disp[k][i]/10

colors = ["greenyellow", "yellowgreen", "olive", "g", "darkgreen"] ## Colors for the plot
t = np.linspace(dt, Nsteps*dt, Nsteps) ## Time array 
plt.figure(figsize=(8,6)) 
for i in range(len(f)): ## Plot the MSD
    plt.title('MSD' , fontsize = 16)
    plt.plot(t, MSD_mean[i], label = 'f=%0.3f'%f[i], color = colors[i])
    plt.xlabel('t/(s)', fontsize = 14)
    plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', fontsize = 14)
#plt.savefig('MSD_4_f', dpi=600)

Disp_mean = np.abs(Disp_mean) ## We get the positive values of the displacement
                                # (since the scale is logaritmic, negative values cannot be plotted)

plt.figure(figsize=(8,6)) ## Displacement plot
for i in range(len(f)):
    plt.title('Displacement', fontsize = 16)
    plt.plot(t, Disp_mean[i], label = 'f=%0.3f'%f[i], color = colors[i])
    plt.xlabel('t/(s)', fontsize = 14)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
    plt.legend(loc='best', fontsize = 14)   
#plt.savefig('Disp3', dpi=600)



#%%


np.random.seed()
L = 1000
N = 10000
D = 1
dt = 0.01
f = 0.001 ## Modulus of the force
tau = np.array([0.5, 1, 10, 50, 100, 250, 500, 750, 1000]) ## Array for the tau
Nsteps = 700
F = np.zeros((len(tau), N))
xi = np.zeros(N)
x = np.zeros((len(tau), N))
MSD = np.zeros((10, len(tau), Nsteps))
Disp = np.zeros((10, len(tau), Nsteps))
MSD_mean = np.zeros((len(tau), Nsteps))
Disp_mean = np.zeros((len(tau), Nsteps))
ji = np.ones((len(tau), N))
v = np.zeros((len(tau), N))
e = np.zeros((len(tau), N))

for s in range(10):
    
    x = np.zeros((len(tau), N))
    ji = np.ones((len(tau), N))
    for k in range(len(tau)):
        e[k] = np.random.choice((-1,1),N)
        F[k] = f*e[k]
        ji[k] = np.ones(N)*np.sqrt(D/tau[k])
    
    for step in range(Nsteps):
        for k in range(len(tau)):
            x[k] = pbc(x[k] + ji[k]*dt + v[k]*dt, L)
            v[k] = v_t(v[k], F[k], dt)
            ji[k] = ji_t_1D(ji[k], tau[k], dt, D, N)
            MSD[s][k][step] = np.mean((pbc(x[k]-xi, L))**2)*2
            Disp[s][k][step] = np.mean(e[k]*(pbc((x[k]-xi), L)))*2


for i in range(len(tau)):
    for k in range(10):
        MSD_mean[i] += MSD[k][i]/10
        Disp_mean[i] += Disp[k][i]/10

t = np.linspace(0, (Nsteps-1)*dt, Nsteps)

mu = np.zeros(len(tau))
d = np.zeros(len(tau))

for i in range(len(tau)):
    mu[i] = Disp_mean[i][Nsteps-1]/(f*t[Nsteps-1])
    d[i] = MSD_mean[i][Nsteps-1]/(2*t[Nsteps-1])


plt.figure(figsize=(8,6))
plt.title("Mobility as a function of tau", fontsize = 16)
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.ylabel('Mobility')
plt.xlabel('\u03C4', fontsize=14)
plt.plot(tau, mu, '--o', color='lime')
#plt.savefig('Disp_tau_5_f0_01', dpi=600)

plt.figure(figsize=(8,6))
plt.title("Diffusion coefficient as a function of tau", fontsize=16)
plt.xlabel('\u03C4', fontsize=14)
plt.ylabel('Diffusion coefficient')
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.plot(tau, d, '--og')
#plt.savefig('MSD_tau_5_f0_01', dpi=600)


plt.figure(figsize=(8,6))
plt.title("Diffusion coefficient and mobility as a function of tau", fontsize=16)
plt.xlabel('\u03C4', fontsize=14)
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.plot(tau, d, '--og', label = 'D')
plt.plot(tau, mu, '--o', color='lime', label = '\u03BC')
plt.legend(loc='best')
#plt.savefig('Apartado_5_f0_01', dpi=600)

