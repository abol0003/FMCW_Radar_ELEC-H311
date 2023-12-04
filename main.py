import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
############ STEP1 ############
# Paramètres du chirp
B = 200e6  # Plage de fréquence en Hz
T = 0.2e-3  # Durée du chirp en secondes
F = 512e6  # Fréquence d'échantillonnage en Hz
num_samples = 2 ** 18  # Nombre d'échantillons

# Calcul de la pente β
beta = B / T

# Créez un vecteur de temps couvrant la durée du chirp
t = np.linspace(0, T, num_samples, endpoint=False)

# Calcul de la fréquence instantanée
fi = beta * t

# Calcul de la phase instantanée
phi_i = 2 * np.pi * np.cumsum(fi) * (1 / F)

# Générez le signal en bande de base e^{jϕi(t)}
baseband_signal = np.exp(1j * phi_i)

# Visualiser la partie réelle du signal en bande de base
plt.figure(1)
plt.plot(t, np.real(baseband_signal))
plt.xlabel('Temps (s)')
plt.ylabel('Partie réelle')
plt.title('Signal en bande de base (Partie réelle)')
plt.xlim(0, 2 * T)
plt.grid()
plt.show(block=False)
plt.savefig("BNDBR.png")

# Visualiser la partie imaginaire du signal en bande de base
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
freq_range = np.fft.fftshift(np.fft.fftfreq(num_samples, 1 / F))

# Calcul de l'amplitude du spectre de fréquence
amplitude = np.abs(fft_result)

# Afficher le spectre de fréquence
plt.figure(3)
plt.plot(freq_range, amplitude)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')
plt.title('Spectre de fréquence du signal en bande de base')
plt.grid()
plt.show(block=False)
plt.savefig("SPCTR.png")

# Trouver la fréquence à la moitié de l'amplitude maximale pour calculer la largeur de bande
max_amplitude = np.max(amplitude)
half_max_amplitude = max_amplitude / 2

# Trouver les indices où l'amplitude est proche de la moitié de l'amplitude maximale
indices = np.where(amplitude >= half_max_amplitude)

# Les fréquences correspondant à ces indices donnent la largeur de bande
bandwidth = freq_range[indices[-1]] - freq_range[indices[0]]
bandwidth_value = bandwidth[0]  # Extraction de la valeur de la largeur de bande
# print("Largeur de bande du signal FMCW en bande de base : {:.2f} Hz".format(bandwidth_value))


########### STEP2 ############
# Définition des paramètres
B = 200e6 # Plage de fréquence en Hz
T = 0.1e-3 # Durée du chirp en secondes
F_s = 2e6 # Fréquence d'échantillonnage radar en Hz
F = 512e6 # Fréquence d'échantillonnage de la simulation en Hz (512 MHz)
F_c = 24e9 # Fréquence porteuse en Hz
N = 512  # Taille FFT rapide en dimension rapide (fast-time)
K = 256  # Taille FFT rapide en dimension lente (slow-time)
guard_samples = 5 # Nombre d'échantillons de garde
R_max = 20  # Portée maximale en mètres
V_max = 2   # Vitesse maximale en m/s
c = 299792458.0  # Vitesse de la lumière en m/s (3e8 m/s)
trgt_numb = 5  # nombre de cibles
Beta = B / T    # Pente du chirp
Tau_max = 2 * R_max / c # Temps de propagation maximal


# Initialisation des paramètres
t_emission_tot = []
Phi_t_tot = []
t_reception_tot_vect = []
t_reception_tot = np.zeros((trgt_numb, K * N))
t_emission_mat = np.zeros((K, N))
t_reception_mat = np.zeros((K, N))
Phi_t_mat = np.zeros((K, N))
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

for r in range(trgt_numb):
    # Appel à la fonction get_RDM_won pour obtenir les RDM pour la cible actuelle
    RDM_fig_4, RDM_eq_16 = get_RDM_won(K, N, T, c, F_c, Beta, t_emission_mat, random_speeds[r], random_delays[r], F_b[r], F_d[r], R_0[r], Kappa[r])

    # Ajout des RDM calculées à la somme cumulative
    Tot_RDM_fig_4 += RDM_fig_4
    Tot_RDM_eq_16 += RDM_eq_16

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
snr_values = [2, 10, 500]  # Valeurs de SNR à évaluer
thresolds = 1e-5 # Seuil pour la détection

# Initialisation des matrices totales
Tot_N_K_fig_4 = np.zeros((N, K), dtype=complex)
Tot_N_K_eq_16 = np.zeros((N, K), dtype=complex)
roc_data = []
# Étape 2 : Calculer la courbe ROC et l'AUC et RDM
for r in range(trgt_numb):
    N_K_fig_4, N_K_eq_16 = get_N_K_ref(K, N, T, c, F_c, Beta, t_emission_mat, random_speeds[r], random_delays[r], F_b[r], F_d[r], R_0[r], Kappa[r])

    # Ajouter les matrices actuelles aux sommes cumulatives
    Tot_N_K_fig_4 += N_K_fig_4
    Tot_N_K_eq_16 += N_K_eq_16

    for snr in snr_values:
        #pour le rdm
        N_K_noise_rdm = add_awgn(Tot_N_K_eq_16, snr) # Ajouter du bruit blanc gaussien à la matrice totale
        rdm_wn_16 = get_RDM_wn(N_K_noise_rdm)      # Calculer la RDM avec bruit
        RDM_wn_16_snr = [rdm_wn_16.copy() for snr in range(len(snr_values))] # Stocker la RDM avec bruit pour chaque valeur de SNR
        #pour le roc
        N_K_noise_roc = add_awgn(N_K_eq_16, snr) # Ajouter du bruit blanc gaussien à la matrice du scenario actuel
        roc_wn_16 = get_RDM_wn(N_K_noise_roc)   # Calculer la RDM avec bruit pour le roc


        # Appliquer un seuil pour détecter les cibles
        normRDM_Won = rdm_without_noise / np.max(rdm_without_noise) # Normaliser la RDM sans bruit
        binary_map_won = detect_targets(normRDM_Won, thresolds) # Appliquer le seuil pour détecter les cibles

        # Étape 2 : Calculer la courbe ROC et l'AUC
        fpr, tpr, thresholds = roc_curve(binary_map_won.flatten(), np.abs(roc_wn_16).flatten()) # Calculer les taux de faux positifs et de vrais positifs
        roc_auc = auc(fpr, tpr) # Calculer l'AUC

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


# Comparaison des RDM avec bruit et sans bruit pour Equation 16
plt.figure(figsize=(16, 6))

# Sous-plot 1 pour RDM sans bruit basée sur Equation 16
plt.subplot(1, 2, 1)
plt.imshow(rdm_without_noise, cmap='jet', aspect='auto', extent=(0, K, N, 0))
plt.colorbar(label='Amplitude')
plt.xlabel('Indice K')
plt.ylabel('Indice N')
plt.title('RDM sans bruit ')

# Sous-plot 2 pour RDM avec bruit
vl_snr=2
plt.subplot(1, 2, 2)
plt.imshow(RDM_wn_16_snr[vl_snr], cmap='jet', aspect='auto',
           extent=(0, K, N, 0))  # Change [0] avec l'indice de la valeur de SNR souhaitée
plt.colorbar(label='Amplitude')
plt.xlabel('Indice K')
plt.ylabel('Indice N')
plt.title(f'RDM avec bruit (SNR={snr_values[vl_snr]} dB) ')
plt.tight_layout()
plt.show(block=False)
plt.savefig("RDM_WITHOUT_WITH.png")

