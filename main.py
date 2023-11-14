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
phi_i = 2 * np.pi * np.cumsum(fi) * (1 / F)  # Correction de la phase instantanée

# Générez le signal en bande de base e^{jϕi(t)}
baseband_signal = np.exp(1j * phi_i)

# Visualiser la partie réelle du signal en bande de base
plt.figure(1)
plt.plot(t, np.real(baseband_signal))
plt.xlabel('Temps (s)')
plt.ylabel('Partie réelle')
plt.title('Signal en bande de base (Partie réelle)')
plt.xlim(0, 2*T)
plt.grid()
#plt.show()

# Visualiser la partie imaginaire du signal en bande de base
plt.figure(2)
plt.plot(t, np.imag(baseband_signal))
plt.xlabel('Temps (s)')
plt.ylabel('Partie imaginaire')
plt.title('Signal en bande de base (Partie imaginaire)')
plt.grid()
plt.xlim(0, 2*T)
#plt.show()

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

import numpy as np
import matplotlib.pyplot as plt

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

    # Étape 2 : Simuler l'impact du canal (une cible)
    received_signal = simulate_multi_target_channel(t, fmcw_signal, target_range, target_velocity)

    # Étape 3 : Traitement radar
    received_signal_blocks = received_signal.reshape(K, -1)
    range_doppler_map = np.fft.fftshift(np.fft.fft(received_signal_blocks, axis=1), axes=1)
    range_doppler_map = np.fft.fftshift(np.fft.fft(range_doppler_map, axis=0), axes=0)

    # Étape 4 : Affichage et identification de la cible
    plt.subplot(2, 3, scenario + 1)
    plt.imshow(np.abs(range_doppler_map), extent=[-target_velocity*1.5, target_velocity*1.5 / 2, 0, target_range * 1.5], aspect='auto', cmap='jet')

    # Marquer l'emplacement de la cible détectée
    plt.scatter(0, target_range , c='red', marker='o', s=50, label='Cible détectée')

    plt.xlabel('Vitesse Doppler (Hz)')
    plt.ylabel('Portée (m)')
    plt.title(f'RDM - Scénario {scenario + 1}')
    plt.legend()
    plt.grid()

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
        # Étape 1: Générer le signal RDM avec des cibles
        # (remplacez cela par votre génération de RDM)
        rdm_with_targets = np.random.rand(100, 100)

        # Étape 2: Ajouter du bruit au signal RDM
        rdm_with_noise = add_noise(rdm_with_targets, snr)

        # Étape 3: Appliquer un seuil pour détecter les cibles
        threshold = 0.5  # À ajuster selon vos besoins
        binary_map = detect_targets(rdm_with_noise, threshold)

        # Étape 4: Estimer les probabilités de fausse alarme et de détection
        true_targets = rdm_with_targets > 0.5  # À ajuster selon vos cibles réelles
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

