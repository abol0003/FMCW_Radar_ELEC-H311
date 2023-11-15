import numpy as np
import matplotlib.pyplot as plt

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
# plt.show()

# Visualiser la partie imaginaire du signal en bande de base
plt.figure(2)
plt.plot(t, np.imag(baseband_signal))
plt.xlabel('Temps (s)')
plt.ylabel('Partie imaginaire')
plt.title('Signal en bande de base (Partie imaginaire)')
plt.grid()
plt.xlim(0, 2 * T)
# plt.show()

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
plt.show()

# Trouver la fréquence à la moitié de l'amplitude maximale pour calculer la largeur de bande
max_amplitude = np.max(amplitude)
half_max_amplitude = max_amplitude / 2

# Trouver les indices où l'amplitude est proche de la moitié de l'amplitude maximale
indices = np.where(amplitude >= half_max_amplitude)

# Les fréquences correspondant à ces indices donnent la largeur de bande
bandwidth = freq_range[indices[-1]] - freq_range[indices[0]]
bandwidth_value = bandwidth[0]  # Extraction de la valeur de la largeur de bande
print("Largeur de bande du signal FMCW en bande de base : {:.2f} Hz".format(bandwidth_value))

##### STEP 2 ######
# Paramètres radar
fc = 24e9  # Fréquence porteuse en Hz (24 GHz)
B = 200e6  # Plage de fréquence en Hz (200 MHz)
Fs = 2e6  # Fréquence d'échantillonnage radar en Hz (2 MHz)
F = 512e6  # Fréquence d'échantillonnage de la simulation en Hz (512 MHz)
N = 512  # Taille FFT rapide en dimension rapide (fast-time)
K = 256  # Taille FFT rapide en dimension lente (slow-time)
guard_samples = 5  # Nombre d'échantillons de garde
c = 299792458.0  # Vitesse de la lumière en m/s
wavelength = c / fc  # Longueur d'onde

# Fonction pour générer le signal FMCW
def generate_fmcw_signal(T_chirp, num_chirps):
    t_chirp = np.linspace(0, T_chirp, N, endpoint=False)
    chirp_signal = np.exp(2j * np.pi * (fc * t_chirp + 0.5 * B * t_chirp ** 2))
    t_total = num_chirps * T_chirp
    t = np.linspace(0, t_total, num_chirps * N, endpoint=False)
    fmcw_signal = np.tile(chirp_signal, num_chirps)
    return t, fmcw_signal

# Fonction pour simuler l'impact du canal sur le signal FMCW (multi-cible)
def simulate_multi_target_channel(t, fmcw_signal, target_range, target_velocity):
    delay_samples = int(target_range * Fs / c)
    doppler_shift = target_velocity * (fc / c)
    received_signal = np.roll(fmcw_signal, delay_samples) * np.exp(1j * 2 * np.pi * doppler_shift * t)
    return received_signal

# Fonction pour calculer la Range-Doppler Map (RDM)
def get_RDM(K, N, T, c, F_c, Beta, t_emission, random_speed, random_delay, F_b, F_d, R_0, Kappa):
    # Réaliser tous les mélanges et les calculs des signaux différents
    K_N_fig_4 = np.zeros((K, N), dtype=complex)  # Utilisation de la figure 4 dans les principes du radar FMCW
    K_N_eq_16 = np.zeros((K, N), dtype=complex)  # Utilisation de l'équation 16 dans les principes du radar FMCW

    for k in range(K):
        # Représentation en bande de base du signal pour une émission de chrip
        BB_s_t = np.exp(1j * (np.pi * Beta * T ** 2 * k + np.pi * Beta * (t_emission[k, :] ** 2)))
        # Représentation en bande de base du signal pour une réception de chrip
        BB_r_t = np.exp(1j * np.pi * Beta * ((t_emission[k, :] - random_delay) ** 2))
        # Représentation en bande de base conjuguée du signal pour une émission de chrip
        BB_s_t_star = np.conj(BB_s_t)
        # Multiplication de BB_r_t avec le décalage de fréquence
        BB_r_t_shifted = BB_r_t * np.exp(-1j * 4 * np.pi * F_c / c * (R_0 + k * random_speed * T))
        # Multiplication de BB_r_t_shifted avec BB_s_t_star pour obtenir le signal vidéo
        x_t_fig_4 = BB_r_t_shifted * BB_s_t_star
        x_t_fig_4 = np.conj(x_t_fig_4)

        # Pour la tâche 3 en utilisant l'équation 16

        # Signal vidéo conjugué complexe
        x_t_eq_16 = Kappa * np.exp(1j * 2 * np.pi * F_b * t_emission[k, :]) * np.exp(1j * 2 * np.pi * F_d * k * T)

        # Affecter x_t_eq_16 à la ligne correspondante de K_N_eq_16
        K_N_eq_16[k, :] = x_t_eq_16

        # Affecter x_t_fig_4 à la ligne correspondante de K_N_fig_4
        K_N_fig_4[k, :] = x_t_fig_4

    N_K_fig_4 = K_N_fig_4.T  #matrice NxK
    N_K_eq_16 = K_N_eq_16.T  #matrice NxK

    range_K_fig_4 = np.abs(
        np.fft.fftshift(np.fft.fft(N_K_fig_4, axis=1), axes=1))  # FFT sur chaque colonne comme un vecteur
    RDM_fig_4 = np.abs(np.fft.fftshift(np.fft.fft(range_K_fig_4, axis=0), axes=0))  # FFT sur chaque ligne
    range_K_eq_16 = np.abs(
        np.fft.fftshift(np.fft.fft(N_K_eq_16, axis=1), axes=1))  # FFT sur chaque colonne comme un vecteur
    RDM_eq_16 = np.abs(np.fft.fftshift(np.fft.fft(range_K_eq_16, axis=0), axes=0))  # FFT sur chaque ligne

    return RDM_fig_4, RDM_eq_16


# Nombre de scénarios à simuler (chacun avec une cible)
num_scenarios = 5

plt.figure(figsize=(15, 10))

for scenario in range(num_scenarios):
    np.random.seed()

    # Générer une cible avec une portée et une vitesse aléatoires
    target_range = np.random.uniform(0, 20)  # Portée maximale en m
    target_velocity = np.random.uniform(0, 2)  # Vitesse maximale en m/s

    print(f"Scénario {scenario + 1} - Paramètres de la cible :")
    print(f"Cible 1 - Portée = {target_range:.2f} m, Vitesse = {target_velocity:.2f} m/s")

    # Étape 1 : Générer le signal FMCW
    T_chirp = 1 / B
    t, fmcw_signal = generate_fmcw_signal(T_chirp, K)

    # Simuler l'impact du canal
    received_signal = simulate_multi_target_channel(t, fmcw_signal, target_range, target_velocity)

    # Stocker le signal reçu pour une utilisation ultérieure
    received_signal_scenario = received_signal.copy()

    # Traitement radar
    received_signal_blocks = received_signal.reshape(K, -1)
    range_doppler_map = np.fft.fftshift(np.fft.fft(received_signal_blocks, axis=1), axes=1)
    range_doppler_map = np.fft.fftshift(np.fft.fft(range_doppler_map, axis=0), axes=0)

    # Étape 4 : Affichage et identification de la cible
    fig = plt.figure(figsize=(15, 5))

    # Plot 3D
    ax = fig.add_subplot(121, projection='3d')
    K_vals, N_vals = np.meshgrid(np.arange(N), np.arange(K))
    ax.plot_surface(N_vals, K_vals, np.abs(range_doppler_map), cmap='viridis', edgecolor='k')
    ax.set_xlabel('N')
    ax.set_ylabel('K')
    ax.set_zlabel('Amplitude')
    ax.set_title(f'RDM 3D - Scénario {scenario + 1}')

    # Plot 2D
    ax = fig.add_subplot(122)
    im = ax.imshow(np.abs(range_doppler_map), extent=[0, K, 0, N], cmap='viridis', origin='lower')
    ax.set_xlabel('K')
    ax.set_ylabel('N')
    ax.set_title(f'RDM 2D - Scénario {scenario + 1}')
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

############### STEP 3 ###############
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour simuler le bruit blanc gaussien
def simulate_gaussian_noise(shape, snr):
    # Puissance du signal en fonction du SNR
    signal_power = 1.0
    noise_power = signal_power / (10 ** (snr / 10.0))

    # Générer un bruit blanc gaussien complexe
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*shape) + 1j * np.random.randn(*shape))

    return noise

# Fonction pour ajouter du bruit à un signal
def add_noise(signal, snr):
    noise = simulate_gaussian_noise(signal.shape, snr)
    noisy_signal = signal + noise
    return noisy_signal

# Fonction pour détecter les cibles dans la RDM en utilisant un seuil
def detect_targets(rdm, threshold):
    binary_map = rdm > threshold
    return binary_map

# Fonction pour estimer la probabilité de fausse alarme et de détection
def estimate_probabilities(binary_map, true_targets):
    false_alarm_map = binary_map & ~true_targets
    miss_map = ~binary_map & true_targets

    probability_false_alarm = np.sum(false_alarm_map) / np.sum(~true_targets)
    probability_miss_detection = np.sum(miss_map) / np.sum(true_targets)

    return probability_false_alarm, probability_miss_detection

# Paramètres
snr_values = [10, 15, 20]  # Valeurs de SNR à évaluer
num_trials = 100  # Nombre de réalisations de bruit pour l'analyse des performances
# Répéter l'analyse pour chaque valeur de SNR
for snr in snr_values:
    # Initialiser les résultats
    probabilities_false_alarm = []
    probabilities_miss_detection = []

    for _ in range(num_trials):
        # Utilisation du signal reçu simulé à partir de l'étape 2
        rdm_with_targets = np.abs(np.fft.fftshift(np.fft.fft(received_signal_scenario)))

        # Étape 2: Ajouter du bruit au signal RDM
        rdm_with_noise = add_noise(rdm_with_targets, snr)

        # Étape 3: Appliquer un seuil pour détecter les cibles
        threshold = 0.5  # À ajuster selon vos besoins
        binary_map = detect_targets(rdm_with_noise, threshold)

        # Étape 4: Estimer les probabilités de fausse alarme et de détection
        true_targets = rdm_with_targets >0.5 # 0.5 est le seuil donc toute amplitude de la RDM supérieure à 0.5 est considérée comme une vraie cible.
        probability_false_alarm, probability_miss_detection = estimate_probabilities(binary_map, true_targets)

        # Stocker les résultats
        probabilities_false_alarm.append(probability_false_alarm)
        probabilities_miss_detection.append(probability_miss_detection)

    # Afficher les résultats pour cette valeur de SNR
    plt.figure()
    plt.scatter(probabilities_false_alarm, probabilities_miss_detection)
    plt.title(f'ROC Curve - SNR {snr} dB')
    plt.xlabel('Probability of False Alarm')
    plt.ylabel('Probability of Miss Detection')
    plt.show()
