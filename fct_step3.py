import numpy as np
from fct_step2 import *
def add_awgn(signal, snr_dB):
    """
    Ajoute un bruit gaussien blanc à un signal complexe.

    :param signal: Le signal complexe auquel ajouter du bruit.
    :param snr_dB: Le rapport signal-sur-bruit en décibels.
    :return: Le signal avec le bruit ajouté.
    """

    snr_linear = 10 ** (snr_dB / 10)
    signal_power = np.mean(np.abs(signal ** 2))
    noise_power = signal_power / snr_linear
    # Générer du bruit gaussien complexe avec la puissance calculée

    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    sgnl_wn = signal + noise

    return sgnl_wn

def get_N_K_ref(K, N, T, c, F_c, Beta, t_emission, random_speed, random_delay, F_b, F_d, R_0, Kappa):
    """
        Génère les signaux radar pour chaque instant de temps et chaque échantillon de fréquence Doppler.
    :return: Deux matrices N_K_fig_4 et N_K_eq_16 qui représentent les signaux générés pour chaque méthode.
    """
    K_N_fig_4 = np.zeros((K, N), dtype=complex)  # Utilisation de la figure 4 dans les principes du radar FMCW
    K_N_eq_16 = np.zeros((K, N), dtype=complex)  # Utilisation de l'équation 16 dans les principes du radar FMCW

    for k in range(K):
        x_t_fig_4 = get_output_signal(T, c, F_c, Beta, t_emission, random_speed, random_delay, R_0, k)
        x_t_eq_16 = Kappa * np.exp(1j * 2 * np.pi * F_b * t_emission[k, :]) * np.exp(1j * 2 * np.pi * F_d * k * T)
        K_N_eq_16[k, :] = x_t_eq_16
        K_N_fig_4[k, :] = x_t_fig_4

    N_K_fig_4 = K_N_fig_4.T  # Comme présenté dans le cours (matrice NxK)
    N_K_eq_16 = K_N_eq_16.T  # Comme présenté dans le cours (matrice NxK)

    return N_K_fig_4, N_K_eq_16
import numpy as np


def get_RDM_wn(N_K):
    """
       Calcule la matrice de distance relative (RDM) en utilisant une transformation de Fourier.

       :param N_K: Matrice des signaux générés pour chaque échantillon de fréquence Doppler et chaque instant de temps.
       :return: La matrice de distance relative (RDM) obtenue par FFT 2D.
       """
    range_K_eq_16 = np.abs(np.fft.fftshift(np.fft.fft(N_K, axis=0)))  # FFT over each column as a vector
    RDM_eq_16 = np.abs(np.fft.fftshift(np.fft.fft(range_K_eq_16, axis=1)))  # FFT over each row
    RDM_wn = np.abs(np.fft.fft2(N_K))**2  # 2D FFT

    return RDM_wn

# Fonction pour détecter les cibles dans la RDM en utilisant un seuil
def detect_targets(rdm, threshold):
    """
       Détecte les cibles dans une matrice seuillée.

       :param rdm: La matrice de distance relative (RDM).
       :param threshold: Le seuil de détection.
       :return: Une matrice binaire où les valeurs au-dessus du seuil sont True et les autres sont False.
       """
    binary_map = (rdm > threshold).astype(bool)
    return binary_map