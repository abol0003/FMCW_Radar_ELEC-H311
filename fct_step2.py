import numpy as np

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
def get_output_signal(T, c, F_c, Beta, t_emission, random_speed, random_delay, R_0, k):
    """
        Génère le signal vidéo pour un écho d'un chirp émis.
        plus précisement décrit chaque étape de la figure 4 du PDF FMCW SIGNAL
        :return: Le signal vidéo généré pour le chirp émis.
    """
    # Représentation en bande de base du signal pour un chirp émis
    BB_s_t = np.exp(1j * (np.pi * Beta * T ** 2 * k + np.pi * Beta * (t_emission[k - 1, :] ** 2)))
    # Représentation en bande de base du signal pour un chirp reçu
    BB_r_t = np.exp(1j * np.pi * Beta * ((t_emission[k - 1, :] - random_delay) ** 2))
    # Représentation en bande de base conjuguée du signal pour un chirp émis
    BB_s_t_star = np.conj(BB_s_t)
    # Multiplication de BB_r_t par le décalage de fréquence
    BB_r_t_shifted = BB_r_t * np.exp(-1j * 4 * np.pi * F_c / c * (R_0 + k * random_speed * T))
    # Multiplication de BB_r_t_shifted par BB_s_t_star pour obtenir le signal vidéo
    x_t_fig_4 = BB_r_t_shifted * BB_s_t_star
    x_t_fig_4 = np.conj(x_t_fig_4)

    return x_t_fig_4


def get_RDM_won(K, N, T, c, F_c, Beta, t_emission, random_speed, random_delay, F_b, F_d, R_0, Kappa):
    """
       Calcule les matrices de distance relative (RDM) en utilisant une transformation de Fourier.
        :return: Deux matrices RDM_fig_4 et RDM_eq_16 obtenues par FFT 2D pour chaque méthode.
    """

    # Effectuer tous les mélanges et le calcul des signaux différents
    K_N_fig_4 = np.zeros((K, N), dtype=complex)  # Using the figure 4 in FMCW radar principles
    K_N_eq_16 = np.zeros((K, N), dtype=complex)  # Using equation 16 in FMCW radar principles

    for k in range(1, K + 1):
        # signal vidéo
        x_t_fig_4 = get_output_signal(T, c, F_c, Beta, t_emission, random_speed, random_delay, R_0, k)
        # Signal vidéo complexe conjugué
        x_t_eq_16 = Kappa * np.exp(1j * 2 * np.pi * F_b * t_emission[k-1, :]) * np.exp(1j * 2 * np.pi * F_d * k * T)

        K_N_eq_16[k-1, :] = x_t_eq_16
        K_N_fig_4[k-1, :] = x_t_fig_4

    N_K_fig_4 = K_N_fig_4.T   # Comme présenté dans le cours (matrice NxK)
    N_K_eq_16 = K_N_eq_16.T   # Comme présenté dans le cours (matrice NxK)

    # FFT 2D
    RDM_fig_4 = np.abs(np.fft.fft2(N_K_fig_4))**2
    RDM_eq_16 = np.abs(np.fft.fft2(N_K_eq_16))**2

    return RDM_fig_4, RDM_eq_16
