import numpy as np
def get_output_signal(T, c, F_c, Beta, t_emission, random_speed, random_delay, R_0, k):
    # Représentation en bande de base du signal pour un chirp émis
    BB_s_t = np.exp(1j * (np.pi * Beta * T**2 * k + np.pi * Beta * (t_emission[k, :]**2)))
    # Représentation en bande de base du signal pour un chirp reçu
    BB_r_t = np.exp(1j * np.pi * Beta * ((t_emission[k, :] - random_delay)**2))
    # Représentation en bande de base conjuguée du signal pour un chirp émis
    BB_s_t_star = np.conj(BB_s_t)
    # Multiplication de BB_r_t avec le décalage de fréquence
    BB_r_t_shifted = BB_r_t * np.exp(-1j * 4 * np.pi * F_c / c * (R_0 + k * random_speed * T))
    # Multiplication de BB_r_t_shifted avec BB_s_t_star pour obtenir le signal vidéo
    x_t_fig_4 = BB_r_t_shifted * BB_s_t_star
    x_t_fig_4 = np.conj(x_t_fig_4)

    return x_t_fig_4


def get_RDM_test(K, N, T, c, F_c, Beta, t_emission, random_speed, random_delay, F_b, F_d, R_0, Kappa):
    # Make all the mixing and the calculation of different signals
    K_N_fig_4 = np.zeros((K, N), dtype=complex)  # Using the figure 4 in FMCW radar principles
    K_N_eq_16 = np.zeros((K, N), dtype=complex)  # Using equation 16 in FMCW radar principles

    for k in range(1, K + 1):
        # Baseband representation of the signal for one emitted chirp
        BB_s_t = np.exp(1j * (np.pi * Beta * T**2 * k + np.pi * Beta * (t_emission[k-1, :]**2)))
        # Baseband representation of the signal for one received chirp
        BB_r_t = np.exp(1j * np.pi * Beta * ((t_emission[k-1, :] - random_delay)**2))
        # Conjugated baseband representation of the signal for one emitted chirp
        BB_s_t_star = np.conj(BB_s_t)
        # Multiplication of BB_r_t with the frequency shift
        BB_r_t_shifted = BB_r_t * np.exp(-1j * 4 * np.pi * F_c / c * (R_0 + k * random_speed * T))
        # Multiplication of BB_r_t_shifted with BB_s_t_star to obtain the video signal
        x_t_fig_4 = BB_r_t_shifted * BB_s_t_star
        x_t_fig_4 = np.conj(x_t_fig_4)

        # For task 3 using equation 16

        # Complex conjugated video signal
        x_t_eq_16 = Kappa * np.exp(1j * 2 * np.pi * F_b * t_emission[k-1, :]) * np.exp(1j * 2 * np.pi * F_d * k * T)

        K_N_eq_16[k-1, :] = x_t_eq_16
        K_N_fig_4[k-1, :] = x_t_fig_4

    N_K_fig_4 = K_N_fig_4.T  # As presented in the course (NxK matrix)
    N_K_eq_16 = K_N_eq_16.T  # As presented in the course (NxK matrix)

    range_K_fig_4 = np.abs(np.fft.fftshift(np.fft.fft(N_K_fig_4, axis=0)))  # FFT over each column as a vector
    RDM_fig_4 = np.abs(np.fft.fftshift(np.fft.fft(range_K_fig_4, axis=1)))  # FFT over each line
    RDM_fig_4 = np.abs(np.fft.fft2(N_K_fig_4))**2
    range_K_eq_16 = np.abs(np.fft.fftshift(np.fft.fft(N_K_eq_16, axis=0)))  # FFT over each column as a vector
    RDM_eq_16 = np.abs(np.fft.fftshift(np.fft.fft(range_K_eq_16, axis=1)))  # FFT over each line
    RDM_eq_16 = np.abs(np.fft.fft2(N_K_eq_16))**2

    return RDM_fig_4, RDM_eq_16
