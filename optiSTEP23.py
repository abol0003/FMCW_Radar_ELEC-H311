import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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


# Constantes STEP 2 ET STEP 3
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
snr_values = [2, 10, 20]  # Valeurs de SNR à évaluer


###### Fonction Step 2 #######
# Fonction pour générer le signal FMCW
def generate_fmcw_signal(T_chirp, num_chirps):
    t_chirp = np.linspace(0, T_chirp, N, endpoint=False)
    chirp_signal = np.exp(1j * np.pi * beta * t_chirp**2) * np.exp(1j * 2 * np.pi * F_c * t_chirp)
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


###### Fonction Step 3 #######
# Fonction pour simuler le bruit blanc gaussien
def simulate_gaussian_noise(shape, snr):
    signal_power = 1.0 # whe have to calcule the module square of the amplitude
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


###### Step 2 #######
# Nombre de scénarios à simuler (chacun avec une cible)
num_scenarios = 3

# Initialisation des variables pour stocker les résultats (STEP 2 et STEP 3)
rdm_with_noise_combined = np.zeros((K, N, num_scenarios, len(snr_values)), dtype=complex)
rdm_without_noise_combined = np.zeros((K, N, num_scenarios), dtype=complex)
probability_false_alarm_list = []
probability_miss_detection_list = []
roc_data=[]

plt.figure(figsize=(15, 10))

for scenario in range(num_scenarios):
    np.random.seed()

    # Générer une cible avec une portée et une vitesse aléatoires
    target_range = np.random.uniform(0, 20)  # Portée maximale en m
    target_velocity = np.random.uniform(0, 2)  # Vitesse maximale en m/s

    print(f"Scénario {scenario + 1} - Paramètres de la cible :")
    print(f"Cible {scenario+1} - Portée = {target_range:.2f} m, Vitesse = {target_velocity:.2f} m/s")

    # Étape 1 : Générer le signal FMCW
    t, fmcw_signal = generate_fmcw_signal(T_chirp, K)

    # Simuler l'impact du canal
    received_signal = simulate_multi_target_channel(t, fmcw_signal, target_range, target_velocity)

    # Stocker les signaux reçu pour STEP 3
    received_signal_scenario = received_signal.copy()

    # Traitement radar
    received_signal_blocks = received_signal.reshape(K, -1)
    range_doppler_map_without_noise = np.fft.fftshift(np.fft.fft(received_signal_blocks, axis=1), axes=1)
    range_doppler_map_without_noise = np.fft.fftshift(np.fft.fft(range_doppler_map_without_noise, axis=0), axes=0)

    # Ajouter la RDM sans bruit à la liste
    rdm_without_noise_combined[:, :, scenario] = np.abs(range_doppler_map_without_noise)

    ###### Step 3 #######
    # Répéter l'analyse pour chaque valeur de SNR
    for snr_index, snr in enumerate(snr_values):
        # Utilisation du signal reçu simulé à partir de l'étape 2
        rdm_with_targets = np.abs(np.fft.fftshift(np.fft.fft(received_signal_scenario)))

        #Ajouter du bruit au signal RDM
        rdm_with_noise = add_noise(rdm_with_targets, snr)

        # Ajouter la RDM avec bruit à la matrice combinée
        rdm_with_noise_combined[:, :, scenario, snr_index] = rdm_with_noise.reshape((K, N))

        #Appliquer un seuil pour détecter les cibles
        threshold = 0.5
        binary_map = detect_targets(rdm_with_noise, threshold)

        #Estimer les probabilités de fausse alarme et de détection
        true_targets = rdm_with_targets > threshold
        probability_false_alarm, probability_miss_detection = estimate_probabilities(binary_map, true_targets)

        # Stocker les résultats pour le scénario actuel et la valeur de SNR et des ROC
        probability_false_alarm_list.append(probability_false_alarm)
        probability_miss_detection_list.append(probability_miss_detection)

        fpr, tpr, thresholds = roc_curve(true_targets, np.abs(rdm_with_noise))
        roc_auc = auc(fpr, tpr)
        roc_data.append((fpr, tpr, roc_auc, scenario, snr))

# Convertir les listes en tableaux numpy pour faciliter la manipulation
probability_false_alarm_array = np.array(probability_false_alarm_list)
probability_miss_detection_array = np.array(probability_miss_detection_list)
# Créer une nouvelle figure

plt.figure(figsize=(12, 6))

# Afficher la RDM sans bruit combinée
plt.subplot(1, 2, 1)
plt.imshow(np.mean(np.real(rdm_without_noise_combined), axis=2), extent=[0, K, 0, N], cmap='viridis', origin='lower')
plt.xlabel('K')
plt.ylabel('N')
plt.title('RDM sans bruit combinée pour les 5 scénarios')
plt.colorbar()


# Afficher la RDM avec bruit combinée
plt.subplot(1, 2, 2)
plt.imshow(np.mean(np.real(rdm_with_noise_combined), axis=2), extent=[0, K, 0, N],cmap='viridis', origin='lower')
plt.xlabel('K')
plt.ylabel('N')
plt.title('RDM avec bruit combinée pour les 5 scénarios')
plt.colorbar()
plt.tight_layout()
plt.show(block=False)
plt.savefig("RDM_WITHOUT_WITH.png")


# Plot 2: Proba des fausses alarmes et des mauvaises détections avec les 3 valeurs sur le graphe
plt.figure(figsize=(8, 6))
for i, snr in enumerate(snr_values):
    plt.scatter(probability_false_alarm_array[i::len(snr_values)], probability_miss_detection_array[i::len(snr_values)],
                label=f'SNR {snr} dB')

plt.title('Probabilité de fausse alarme et de mauvaise détection')
plt.xlabel('Probabilité de fausse alarme')
plt.ylabel('Probabilité de mauvaise détection')
plt.legend()
plt.show(block=False)
plt.savefig("PROBA.png")

# Plot de la courbe ROC
plt.figure(figsize=(10, 7))
for i, (fpr, tpr, roc_auc, scenario, snr) in enumerate(roc_data):
    plt.plot(fpr, tpr, lw=2, label=f'Scénario {scenario+1} (SNR {snr} dB, AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC pour différents scénarios')
plt.legend(loc="lower right")
plt.show(block=False)
plt.savefig("ROC.png")