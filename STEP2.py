import numpy as np
import matplotlib.pyplot as plt

# Définition des paramètres
B = 200e6
T = 0.1e-3
F_s = 2e6
F = 512e6
N_s = 2**18
F_c = 24e9
N = 512
K = 256
N_s_off = 5
R_max = 20
V_max = 2
c = 3e8
N_t = 10

Beta = B / T
Tau_max = 2 * R_max / c
Delta_R_0 = c / (2 * B)
Delta_v = c / (2 * K * T * F_c)

# Initialisation des paramètres step2
t_emission_tot = []
Fi_t_tot = []
t_reception_tot_vect = []
t_reception_tot = np.zeros((N_t, K * N))
t_emission_mat = np.zeros((K, N))
t_reception_mat = np.zeros((K, N))
Fi_t_mat = np.zeros((K, N))
matrices_t_reception = [np.zeros((K, N)) for _ in range(N_t)]
Tot_RDM_fig_4 = np.zeros((N, K))
Tot_RDM_eq_16 = np.zeros((N, K))

# Génération des retards et vitesses aléatoires
random_delays = np.random.rand(N_t) *Tau_max
random_speeds = np.random.rand(N_t) * V_max
R_0 = (c * random_delays) / 2
Kappa = np.exp(4 * np.pi * 1j * R_0 * F_c / c) * np.exp(-2 * np.pi * 1j * Beta**2 * R_0**2 / c**2)
F_d = 2 * random_speeds * F_c / c
F_b = 2 * R_0 * Beta / c


from fct_step2 import *
from fct_step3 import *

# Étape 2: Traitement radar

# Pour la tâche 1 et 2 avec plusieurs cibles (=> plusieurs temps de réception)

for r in range(N_t):
    for k in range(K):
        t_reception = np.arange(random_delays[r], T + random_delays[r], T / (N + N_s_off - 1))
        t_reception = t_reception[:N]
        t_reception_mat[k, :] = t_reception
    matrices_t_reception[r] = t_reception_mat

for k in range(K):
    t_emission = np.arange(0, T, T / (N + N_s_off - 1))
    F_i_t = Beta * t_emission
    t_emission = t_emission[:N]
    F_i_t = F_i_t[:N]
    t_emission_mat[k, :] = t_emission
    Fi_t_mat[k, :] = F_i_t

for k in range(K):
    F_i_t = Fi_t_mat[k, :]
    t_emission = t_emission_mat[k, :]
    Fi_t_tot = np.concatenate((Fi_t_tot, F_i_t))
    t_emission_tot = np.concatenate((t_emission_tot, t_emission + (k - 1) * T))

for r in range(N_t):
    current_t_reception_mat = matrices_t_reception[r]
    for k in range(K):
        t_reception = current_t_reception_mat[k, :]
        t_reception_tot_vect = np.concatenate((t_reception_tot_vect, t_reception + (k - 1) * T))
    t_reception_tot[r, :] = t_reception_tot_vect
    t_reception_tot_vect = []


Tot_RDM_fig_4 = np.zeros((N, K))
Tot_RDM_eq_16 = np.zeros((N, K))

for r in range(N_t):
    RDM_fig_4, RDM_eq_16 = get_RDM_test(K, N, T, c, F_c, Beta, t_emission_mat, random_speeds[r], random_delays[r], F_b[r], F_d[r], R_0[r], Kappa[r])
    Tot_RDM_fig_4 += RDM_fig_4
    Tot_RDM_eq_16 += RDM_eq_16

# Représentation 2D avec des couleurs pour l'amplitude

plt.figure(figsize=(16, 6))

# Sous-plot 1 pour Tot_RDM_fig_4
plt.subplot(1, 2, 1)
plt.imshow(Tot_RDM_fig_4, cmap='jet', aspect='auto', extent=(0, K, N, 0))
plt.colorbar(label='Amplitude')
plt.xlabel('Indice K')
plt.ylabel('Indice N')
plt.title('Amplitude basée sur Figure 4')

# Sous-plot 2 pour Tot_RDM_eq_16
plt.subplot(1, 2, 2)
plt.imshow(Tot_RDM_eq_16, cmap='jet', aspect='auto', extent=(0, K, N, 0))
plt.colorbar(label='Amplitude')
plt.xlabel('Indice K')
plt.ylabel('Indice N')
plt.title('Amplitude basée sur Equation 16')

plt.tight_layout()
plt.show()


plt.show()
