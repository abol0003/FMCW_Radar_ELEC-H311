import numpy as np
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


def get_RDM_wn(N_K):
    """
       Calcule la matrice de distance relative (RDM) en utilisant une transformation de Fourier.

       :param N_K: Matrice des signaux générés pour chaque échantillon de fréquence Doppler et chaque instant de temps.
       :return: La matrice de distance relative (RDM) obtenue par FFT 2D.
       """
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