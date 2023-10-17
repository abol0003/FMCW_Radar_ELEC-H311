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

#####    STEP 2    ##########

# Paramètres radar
fc = 24e9  # Fréquence porteuse en Hz (24 GHz)
B = 200e6  # Plage de fréquence en Hz (200 MHz)
Fs = 2e6  # Fréquence d'échantillonnage radar en Hz (2 MHz)
F = 512e6  # Fréquence d'échantillonnage de la simulation en Hz (512 MHz)
N = 512  # Taille FFT rapide en dimension rapide (fast-time)
K = 256  # Taille FFT rapide en dimension lente (slow-time)
guard_samples = 5  # Nombre d'échantillons de garde

# Génération du signal FMCW composé de K chirps
T_chirp = 1 / B
t_chirp = np.linspace(0, T_chirp, N, endpoint=False)
chirp_signal = np.exp(2j * np.pi * (fc * t_chirp + 0.5 * B * t_chirp**2))

# Génération du signal complet avec K chirps
num_chirps = K
t_total = num_chirps * T_chirp
t = np.linspace(0, t_total, num_chirps * N, endpoint=False)
fmcw_signal = np.tile(chirp_signal, num_chirps)

# Simulation de l'impact du canal sur le signal
# Dans ce cas, nous supposons un simple retard de temps et de fréquence constants
delay_samples = int(Fs * 0.001)  # Retard de 1 ms (1 ms = 1000 µs)
doppler_shift = 1e3  # Changement Doppler de 1 kHz

received_signal = np.roll(fmcw_signal, delay_samples) * np.exp(1j * 2 * np.pi * doppler_shift * t)

# Traitement radar
# Sérialisation en parallèle (S/P)
received_signal_blocks = received_signal.reshape(K, -1)

# FFT en dimension rapide
range_doppler_map = np.fft.fftshift(np.fft.fft(received_signal_blocks, axis=1), axes=1)

# FFT en dimension lente
range_doppler_map = np.fft.fftshift(np.fft.fft(range_doppler_map, axis=0), axes=0)

# Visualisation de la carte de portée-Doppler (RDM)
plt.figure(4)
plt.imshow(np.abs(range_doppler_map), extent=[-Fs/2, Fs/2, 0, F/2], aspect='auto', cmap='jet')
plt.xlabel('Vitesse Doppler (Hz)')
plt.ylabel('Portée (m)')
plt.title('Carte de portée-Doppler (RDM)')
plt.grid()
plt.xlim(-2*Fs, 2*Fs) #ajuste l'axe des x
plt.ylim(0, 2*F) #ajuste l'axe des y
plt.show()

# Détection de cibles sur la RDM
detection_threshold = 0.5 * np.max(np.abs(range_doppler_map))
target_indices = np.where(np.abs(range_doppler_map) > detection_threshold)
targets = np.array(target_indices).T

# Calcul des résolutions de portée et de Doppler
c = 299792458.0  # Vitesse de la lumière en m/s
wavelength = c / fc  # Longueur d'onde
range_resolution = c / (2 * B)
doppler_resolution = wavelength / (2 * T_chirp)

# Discussion de la pertinence des paramètres radar
print(f"Résolution de portée : {range_resolution:.2f} m")
print(f"Résolution Doppler : {doppler_resolution:.2f} Hz")

