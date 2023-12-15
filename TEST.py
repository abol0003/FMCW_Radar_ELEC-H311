import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

############ STEP1 ############
# Paramètres du chirp
B = 200e6  # Plage de fréquence en Hz
T = 0.1e-3  # Durée du chirp en secondes
F = 512e6  # Fréquence d'échantillonnage en Hz
num_samples = 2 ** 18  # Nombre d'échantillons

# Calcul de la pente β
beta = B / T

# Créez un vecteur de temps couvrant la durée du chirp
t = np.arange(0, T, T / num_samples)

# Calcul de la fréquence instantanée
fi = beta * t

# Calcul de la phase instantanée
phi_i = beta * np.pi * (t ** 2)

# Générez le signal en bande de base e^{jϕi(t)}
baseband_signal = np.exp(1j * phi_i)

# Afficher la partie réelle du signal en bande de base
plt.figure(1)
plt.plot(t, np.real(baseband_signal))
plt.xlabel('Temps (s)')
plt.ylabel('Partie réelle')
plt.title('Signal en bande de base (Partie réelle)')
plt.xlim(0, 2 * T)
plt.grid()
plt.show(block=False)
plt.savefig("BNDBR.png")

# Afficher la partie imaginaire du signal en bande de base
plt.figure(2)
plt.plot(t, np.imag(baseband_signal))
plt.xlabel('Temps (s)')
plt.ylabel('Partie imaginaire')
plt.title('Signal en bande de base (Partie imaginaire)')
plt.grid()
plt.xlim(0, 2 * T)
plt.show(block=False)
plt.savefig("BNDBI.png")

# Calcul de la transformée de Fourier
fft_result = np.fft.fftshift(np.fft.fft(baseband_signal))

# Calcul des fréquences associées aux échantillons de la transformée de Fourier
freq_range = np.arange(-F / 2, F / 2, F / num_samples)

# Calcul de l'amplitude du spectre de fréquence
amplitude = np.abs(fft_result)

# Afficher le spectre de fréquence
plt.figure(3)
plt.plot(freq_range, amplitude)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('|FFT(X)|')
plt.title('Spectre de fréquence du signal en bande de base')
plt.grid()
plt.show(block=False)
plt.savefig("SPCTR.png")

########### STEP2 ############
# Définition des paramètres
B = 200e6  # Plage de fréquence en Hz
T = 0.1e-3  # Durée du chirp en secondes
F_s = 2e6  # Fréquence d'échantillonnage radar en Hz
F = 512e6  # Fréquence d'échantillonnage de la simulation en Hz (512 MHz)
F_c = 24e9  # Fréquence porteuse en Hz
N = 512  # Taille FFT rapide en dimension rapide (fast-time)
K = 256  # Taille FFT rapide en dimension lente (slow-time)
guard_samples = 5  # Nombre d'échantillons de garde
R_max = 20  # Portée maximale en mètres
V_max = 2  # Vitesse maximale en m/s
c = 299792458.0  # Vitesse de la lumière en m/s (3e8 m/s)
trgt_numb = 10  # nombre de cibles
Beta = B / T  # Pente du chirp
Tau_max = 2 * R_max / c  # Temps de propagation maximal

# Initialisation des paramètres
t_emission_tot = []
Phi_t_tot = []
t_reception_tot_vect = []
t_reception_tot = np.zeros((trgt_numb, K * N))
t_emission_mat = np.zeros((K, N))
t_reception_mat = np.zeros((K, N))
Phi_t_mat = np.zeros((K, N))
matrices_t_reception = [np.zeros((K, N)) for _ in range(trgt_numb)]

# Génération des retards et vitesses aléatoires


R_0 = np.array([5.41, 11.11, 12.10, 15.96, 4.22, 13.80, 10.94, 3.65, 0.10, 15.89])
random_speeds = np.array([1.88, 0.33, 1.23, 1.01, 1.90, 1.16, 1.68, 0.18, 1.47, 1.57])
random_delays=(R_0*2)/c

print(R_0)
Kappa = np.exp(4 * np.pi * 1j * R_0 * F_c / c) * np.exp(-2 * np.pi * 1j * Beta ** 2 * R_0 ** 2 / c ** 2)
F_d = 2 * random_speeds * F_c / c
F_b = 2 * R_0 * Beta / c
from fct_step2 import *

# Étape 2: Traitement radar
# Génération des instants d'émission et de réception pour chaque cible et chaque chirp
for r in range(trgt_numb):
    # Instants de réception pour chaque échantillon de fréquence Doppler et chaque instant de temps
    for k in range(K):
        t_reception = np.arange(random_delays[r], T + random_delays[r], T / (N + guard_samples - 1))
        t_reception = t_reception[:N]
        t_reception_mat[k, :] = t_reception
    matrices_t_reception[r] = t_reception_mat

# Instants d'émission et fréquence instantanée pour chaque échantillon de fréquence Doppler et chaque instant de temps
for k in range(K):
    t_emission = np.arange(0, T, T / (N + guard_samples - 1))
    F_i_t = Beta * t_emission
    t_emission = t_emission[:N]
    F_i_t = F_i_t[:N]
    t_emission_mat[k, :] = t_emission
    Phi_t_mat[k, :] = F_i_t

# Rassemblement des instants d'émission pour chaque chirp
for k in range(K):
    F_i_t = Phi_t_mat[k, :]
    t_emission = t_emission_mat[k, :]
    Phi_t_tot = np.concatenate((Phi_t_tot, F_i_t))
    t_emission_tot = np.concatenate((t_emission_tot, t_emission + (k - 1) * T))

# Rassemblement des instants de réception pour chaque cible et chaque chirp
for r in range(trgt_numb):
    current_t_reception_mat = matrices_t_reception[r]
    for k in range(K):
        t_reception = current_t_reception_mat[k, :]
        t_reception_tot_vect = np.concatenate((t_reception_tot_vect, t_reception + (k - 1) * T))
    t_reception_tot[r, :] = t_reception_tot_vect
    t_reception_tot_vect = []

# Initialisation des matrices totales pour les RDM
Tot_RDM_fig_4 = np.zeros((N, K))
Tot_RDM_eq_16 = np.zeros((N, K))
# Pour step 3
Tot_N_K_fig_4 = np.zeros((N, K), dtype=complex)
Tot_N_K_eq_16 = np.zeros((N, K), dtype=complex)

for r in range(trgt_numb):
    # Appel à la fonction get_RDM_won pour obtenir les RDM pour la cible actuelle
    RDM_fig_4, RDM_eq_16 = get_RDM_won(K, N, T, c, F_c, Beta, t_emission_mat, random_speeds[r], random_delays[r],
                                       F_b[r], F_d[r], R_0[r], Kappa[r])

    # Ajout des RDM calculées à la somme cumulative
    Tot_RDM_fig_4 += RDM_fig_4
    Tot_RDM_eq_16 += RDM_eq_16

    # permet l'etape 3
    N_K_fig_4, N_K_eq_16 = get_N_K_ref(K, N, T, c, F_c, Beta, t_emission_mat, random_speeds[r], random_delays[r],
                                       F_b[r], F_d[r], R_0[r], Kappa[r])

    Tot_N_K_fig_4 += N_K_fig_4
    Tot_N_K_eq_16 += N_K_eq_16  # pour rdm

# sauvegarde pour step 3
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
plt.show(block=False)
plt.savefig("RDM_WITHOUT.png")

############ STEP3 ############

from fct_step3 import *

# variables
snr_values = [-25, 10, 80]  # Valeurs de SNR à évaluer
thresolds =6e-7  # Seuil pour la definir la binary map de ce qui est considéré comme un signal ou non
# tres bas pour pouvoir avoir un maximum de signaux binaire (1) à analyser

# Initialisation des matrices totales
roc_data = []
RDM_wn_16_snr = []
# Calculer la courbe ROC et l'AUC et RDM

for snr in snr_values:
    # pour le rdm
    N_K_noise_rdm_roc = add_awgn(Tot_N_K_eq_16, snr)  # Ajouter du bruit blanc gaussien à la matrice totale
    rdm_wn_16 = get_RDM_wn(N_K_noise_rdm_roc)  # Calculer la RDM avec bruit
    RDM_wn_16_snr.append(rdm_wn_16.copy())  # Stocker la RDM avec bruit pour chaque valeur de SNR

    # pour le roc

    # Appliquer un seuil pour détecter les cibles
    normRDM_Won = rdm_without_noise / np.max(rdm_without_noise)  # Normaliser la RDM sans bruit
    binary_map_won = detect_targets(normRDM_Won, thresolds)  # Appliquer le seuil pour détecter les cibles

    # Étape 2 : Calculer la courbe ROC et l'AUC

    # 1. En utilisant la fonction roc curve faite par nous même
    #fpr, tpr, roc_auc = roc_curve_custom(binary_map_won.flatten(), np.abs(rdm_wn_16).flatten())  # Calculer les taux de faux positifs et de vrais positifs

    # 2. En utilisant la fonction roc curve toute faite par la blackboxe sklearn.metrics

    fpr, tpr, thresholds = roc_curve(binary_map_won.flatten(), np.abs(rdm_wn_16).flatten())  # Calculer les taux de faux positifs et de vrais positifs
    roc_auc = auc(fpr, tpr)  # Calculer l'AUC

    # Les deux manières ont été comparés et affiche des résultats similaire,
    # juste un tant d'execution plus long pour la roc_curve_custom

    # Stocker les résultats pour le scénario actuel, la valeur de SNR et le numéro de cible
    roc_data.append((fpr, tpr, roc_auc, snr))

# Plot de la courbe ROC
plt.figure(figsize=(10, 7))
for i, (fpr, tpr, roc_auc, snr) in enumerate(roc_data):
    plt.plot(fpr, tpr, lw=2, label=f'SNR {snr} dB,( AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title(f'Courbe ROC pour les différents scénarios')
plt.legend(loc="lower right")
plt.savefig("ROC.png")

# Comparaison des RDM avec bruit et sans bruit pour Equation 16
plt.figure(figsize=(16, 6))
vl_snr = 0
max_amplitude = np.max(rdm_without_noise)

# Sous-plot 1 pour RDM sans bruit basée sur Equation 16
plt.subplot(1, 3, 1)
plt.imshow(rdm_without_noise, cmap='jet', aspect='auto', extent=(0, K, N, 0), vmax=max_amplitude)
plt.colorbar(label='Amplitude')
plt.xlabel('Indice K')
plt.ylabel('Indice N')
plt.title('RDM sans bruit ')

# Sous-plot 2 pour RDM avec bruit
plt.subplot(1, 3, 2)
plt.imshow(RDM_wn_16_snr[vl_snr], cmap='jet', aspect='auto', extent=(0, K, N, 0), vmax=max_amplitude)
plt.colorbar(label='Amplitude')
plt.xlabel('Indice K')
plt.ylabel('Indice N')
plt.title(f'RDM avec bruit (SNR={snr_values[vl_snr]} dB) ')

# Sous-plot 3 pour l'autre RDM avec bruit
plt.subplot(1, 3, 3)
plt.imshow(RDM_wn_16_snr[len(snr_values) - 1], cmap='jet', aspect='auto', extent=(0, K, N, 0), vmax=max_amplitude)
plt.colorbar(label='Amplitude')
plt.xlabel('Indice K')
plt.ylabel('Indice N')
plt.title(f'Autre RDM avec bruit (SNR={snr_values[len(snr_values) - 1]} dB) ')

plt.tight_layout()
plt.show(block=True)
plt.savefig("RDM_WITHOUT_WITH.png")
