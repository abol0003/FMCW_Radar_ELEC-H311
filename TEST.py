import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Définition des paramètres
B = 200e6
T = 0.1e-3
F_s = 2e6
F = 512e6
N_s = 2 ** 18
F_c = 24e9
N = 512
K = 256
N_s_off = 5
R_max = 20
V_max = 2
c = 3e8
trgt_numb = 3  # nombre de cibles

Beta = B / T
Tau_max = 2 * R_max / c
Delta_R_0 = c / (2 * B)
Delta_v = c / (2 * K * T * F_c)

# Initialisation des paramètres totaux
t_emission_tot = []
Fi_t_tot = []
t_reception_tot_vect = []
t_reception_tot = np.zeros((trgt_numb, K * N))
t_emission_mat = np.zeros((K, N))
t_reception_mat = np.zeros((K, N))
Fi_t_mat = np.zeros((K, N))
matrices_t_reception = [np.zeros((K, N)) for _ in range(trgt_numb)]
Tot_RDM_fig_4 = np.zeros((N, K))
Tot_RDM_eq_16 = np.zeros((N, K))

# Génération des retards et vitesses aléatoires
random_delays = np.random.rand(trgt_numb) * Tau_max
random_speeds = np.random.rand(trgt_numb) * V_max
R_0 = (c * random_delays) / 2
Kappa = np.exp(4 * np.pi * 1j * R_0 * F_c / c) * np.exp(-2 * np.pi * 1j * Beta ** 2 * R_0 ** 2 / c ** 2)
F_d = 2 * random_speeds * F_c / c
F_b = 2 * R_0 * Beta / c
from fct_step2 import *

# Étape 2: Traitement radar

# Pour la tâche 1 et 2 avec plusieurs cibles (=> plusieurs temps de réception)

for r in range(trgt_numb):
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

for r in range(trgt_numb):
    current_t_reception_mat = matrices_t_reception[r]
    for k in range(K):
        t_reception = current_t_reception_mat[k, :]
        t_reception_tot_vect = np.concatenate((t_reception_tot_vect, t_reception + (k - 1) * T))
    t_reception_tot[r, :] = t_reception_tot_vect
    t_reception_tot_vect = []

Tot_RDM_fig_4 = np.zeros((N, K))
Tot_RDM_eq_16 = np.zeros((N, K))

for r in range(trgt_numb):
    RDM_fig_4, RDM_eq_16 = get_RDM_test(K, N, T, c, F_c, Beta, t_emission_mat, random_speeds[r], random_delays[r],
                                        F_b[r], F_d[r], R_0[r], Kappa[r])
    Tot_RDM_fig_4 += RDM_fig_4
    Tot_RDM_eq_16 += RDM_eq_16

# pour step 3
rdm_without_noise = Tot_RDM_eq_16
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

# step 3

from fct_step3 import *

# variables
snr_values = [2, 10, 50]  # Valeurs de SNR à évaluer
thresolds = 1e-5

# Initialisation des matrices totales
Tot_N_K_fig_4 = np.zeros((N, K), dtype=complex)
Tot_N_K_eq_16 = np.zeros((N, K), dtype=complex)
roc_data = []
# Étape 2 : Calculer la courbe ROC et l'AUC pour chaque scénario et SNR
for r in range(trgt_numb):
    N_K_fig_4, N_K_eq_16 = get_N_K_ref(K, N, T, c, F_c, Beta, t_emission_mat, random_speeds[r], random_delays[r],
                                       F_b[r], F_d[r], R_0[r], Kappa[r])

    # Ajouter les matrices actuelles aux sommes cumulatives
    Tot_N_K_fig_4 += N_K_fig_4
    Tot_N_K_eq_16 += N_K_eq_16

    for snr in snr_values:
        #pour le rdm
        N_K_noise_rdm = add_awgn(Tot_N_K_eq_16, snr) #probleme c'est que on ajoute du bruit à la matrice total mais donc mauvais roc
        rdm_wn_16 = get_RDM_wn(N_K_noise_rdm)
        RDM_wn_16_snr = [rdm_wn_16.copy() for snr in range(len(snr_values))]
        #pour le roc
        N_K_noise_roc = add_awgn(N_K_eq_16, snr)  # probleme c'est que on ajoute du bruit à la matrice total mais donc mauvais roc
        roc_wn_16 = get_RDM_wn(N_K_noise_roc)


        # Appliquer un seuil pour détecter les cibles
        normRDM_Won = rdm_without_noise / np.max(rdm_without_noise)
        binary_map_won = detect_targets(normRDM_Won, thresolds)

        # Étape 2 : Calculer la courbe ROC et l'AUC
        fpr, tpr, thresholds = roc_curve(binary_map_won.flatten(), np.abs(roc_wn_16).flatten())
        roc_auc = auc(fpr, tpr)

        # Stocker les résultats pour le scénario actuel, la valeur de SNR et le numéro de cible
        roc_data.append((fpr, tpr, roc_auc, r, snr))

# Plot de la courbe ROC
plt.figure(figsize=(10, 7))
for i, (fpr, tpr, roc_auc, scenario, snr) in enumerate(roc_data):
    plt.plot(fpr, tpr, lw=2, label=f'Scénario {scenario + 1} (SNR {snr} dB, AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Taux de faux positifs (FPR)')
plt.xlabel('Taux de vrais positifs (TPR)')
plt.title(f'Courbe ROC pour les différents scénarios')
plt.legend(loc="lower right")
plt.savefig("ROC.png")  # Sauvegarde de l'image
plt.show()

# Comparaison des RDM avec bruit et sans bruit pour Equation 16
plt.figure(figsize=(16, 6))

# Sous-plot 1 pour RDM sans bruit basée sur Equation 16
plt.subplot(1, 2, 1)
plt.imshow(rdm_without_noise, cmap='jet', aspect='auto', extent=(0, K, N, 0))
plt.colorbar(label='Amplitude')
plt.xlabel('Indice K')
plt.ylabel('Indice N')
plt.title('RDM sans bruit basée sur Equation 16')

# Sous-plot 2 pour RDM avec bruit basée sur Equation 16
plt.subplot(1, 2, 2)
plt.imshow(RDM_wn_16_snr[2], cmap='jet', aspect='auto',
           extent=(0, K, N, 0))  # Change [0] avec l'indice de la valeur de SNR souhaitée
plt.colorbar(label='Amplitude')
plt.xlabel('Indice K')
plt.ylabel('Indice N')
plt.title(f'RDM avec bruit (SNR={snr_values[2]} dB) basée sur Equation 16')

plt.tight_layout()
plt.show()
