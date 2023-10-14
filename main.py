#Etape Une : Génération du signal FMCW en bande de base

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
phi_i = np.pi * beta * t ** 2

# Génére le signal en bande de base e^{jϕi(t)}
baseband_signal = np.exp(1j * phi_i)

# visualiser le signal en bande de base
plt.figure(1)
plt.plot(t, np.real(baseband_signal))
plt.xlabel('Temps (s)')
plt.ylabel('Partie réelle')
plt.title('Signal en bande de base (Partie réelle)')
plt.grid()
plt.show()

#  partie imaginaire
plt.figure(2)
plt.plot(t, np.imag(baseband_signal))
plt.xlabel('Temps (s)')
plt.ylabel('Partie imaginaire')
plt.title('Signal en bande de base (Partie imaginaire)')
plt.grid()
plt.show()

# Calcul de la transformée de Fourier

fft_result = np.fft.fftshift(np.fft.fft(baseband_signal))
# np.fft.fft(baseband_signal) :
# Cette partie calcule la transformée de Fourier du signal en bande de base (baseband_signal).
# La fonction np.fft.fft() effectue la transformation de Fourier discrète.
# Elle prend en entrée le signal que à transformer.

freq_range = np.fft.fftshift(np.fft.fftfreq(num_samples, 1/F))
# np.fft.fftfreq(num_samples, 1/F) génère un vecteur de fréquences
# qui couvre la plage des fréquences allant de -F/2 à F/2.
# Cela signifie que les fréquences négatives sont représentées, ainsi que les fréquences positives,
# et que zéro est au centre de ce vecteur.

# Amplitude du signal en bande de base dans le domaine de la fréquence
amplitude = np.abs(fft_result)

# afficher le spectre de fréquence
plt.figure(3)
plt.plot(freq_range, amplitude)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')
plt.title('Spectre de fréquence du signal en bande de base')
plt.grid()
plt.show()

# Pour déduire la largeur de bande, il faut trouver la fréquence à la moitié de l'amplitude maximale
max_amplitude = np.max(amplitude)
half_max_amplitude = max_amplitude / 2

# Trouver les indices où l'amplitude est proche de la moitié de l'amplitude maximale
indices = np.where(amplitude >= half_max_amplitude)

# Les fréquences correspondant à ces indices donnent la largeur de bande
bandwidth = freq_range[indices[-1]] - freq_range[indices[0]]
bandwidth_value = bandwidth[0]  # Extraction de la valeur de la largeur de bande
print("Largeur de bande du signal FMCW en bande de base : {:.2f} Hz".format(bandwidth_value))

