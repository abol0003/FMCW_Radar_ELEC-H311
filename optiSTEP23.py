import numpy as np
import matplotlib.pyplot as plt

# Constantes
B = 200e6  # Plage de fréquence en Hz
T_chirp = 1 / B
F_c = 24e9  # Fréquence porteuse en Hz (24 GHz)
Fs_radar = 2e6  # Fréquence d'échantillonnage radar en Hz (2 MHz)
F_simulation = 512e6  # Fréquence d'échantillonnage de la simulation en Hz (512 MHz)
N = 512  # Taille FFT rapide en dimension rapide (fast-time)
K = 256  # Taille FFT rapide en dimension lente (slow-time)
guard_samples = 5  # Nombre d'échantillons de garde
c = 299792458.0  # Vitesse de la lumière en m/s
wavelength = c / F_c  # Longueur d'onde
snr_values = [10, 15, 20]  # Valeurs de SNR à évaluer
num_trials = 100  # Nombre de réalisations de bruit pour l'analyse des performances

# Fonction pour générer le signal FMCW
def generate_fmcw_signal(T_chirp, num_chirps):
    t_chirp = np.linspace(0, T_chirp, N, endpoint=False)
    chirp_signal = np.exp(2j * np.pi * (F_c * t_chirp + 0.5 * B * t_chirp ** 2))
    t_total = num_chirps * T_chirp
    t = np.linspace(0, t_total, num_chirps * N, endpoint=False)
    fmcw_signal = np.tile(chirp_signal, num_chirps)
    return t, fmcw_signal

# Fonction pour simuler l'impact du canal sur le signal FMCW (multi-cible)
def simulate_multi_target_channel(t, fmcw_signal, target_range, target_velocity):
    delay_samples = int(target_range * Fs_radar / c)
    doppler_shift = target_velocity * (F_c / c)
    received_signal = np.roll(fmcw_signal, delay_samples) * np.exp(1j * 2 * np.pi * doppler_shift * t)
    return received_signal

# Fonction pour simuler le bruit blanc gaussien
def simulate_gaussian_noise(shape, snr):
    signal_power = 1.0
    noise_power = signal_power / (10 ** (snr / 10.0))
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

    # Répéter l'analyse pour chaque valeur de SNR
    for snr in snr_values:
        # Utilisation du signal reçu simulé à partir de l'étape 2
        rdm_with_targets = np.abs(np.fft.fftshift(np.fft.fft(received_signal_scenario)))

        # Étape 2: Ajouter du bruit au signal RDM
        rdm_with_noise = add_noise(rdm_with_targets, snr)

        # Étape 3: Appliquer un seuil pour détecter les cibles
        threshold = 0.5  # À ajuster selon vos besoins
        binary_map = detect_targets(rdm_with_noise, threshold)

        # Étape 4: Estimer les probabilités de fausse alarme et de détection
        true_targets = rdm_with_targets > 0.5
        probability_false_alarm, probability_miss_detection = estimate_probabilities(binary_map, true_targets)

        # Afficher les résultats pour cette valeur de SNR
        plt.figure()
        plt.scatter(probability_false_alarm, probability_miss_detection)
        plt.title(f'ROC Curve - SNR {snr} dB')
        plt.xlabel('Probability of False Alarm')
        plt.ylabel('Probability of Miss Detection')
        plt.show()
