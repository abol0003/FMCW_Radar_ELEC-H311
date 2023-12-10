import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

############ STEP1 ############
# Paramètres du chirp
B = 200e6  # Plage de fréquence en Hz
T = 0.2e-3  # Durée du chirp en secondes
F = 512e6  # Fréquence d'échantillonnage en Hz
num_samples = 2 ** 18  # Nombre d'échantillons

# Calcul de la pente β
beta = B / T

# Créez un vecteur de temps couvrant la durée du chirp
t = np.arange(0, T, T/num_samples)

# Calcul de la fréquence instantanée
fi = beta * t

# Calcul de la phase instantanée
phi_i = beta * np.pi * (t**2)
#phi_i = 2 * np.pi * np.cumsum(fi) * (1 / F)


# Générez le signal en bande de base e^{jϕi(t)}
baseband_signal = np.exp(1j * phi_i)


# Calcul de la transformée de Fourier
fft_result = np.fft.fftshift(np.fft.fft(baseband_signal))

# Calcul des fréquences associées aux échantillons de la transformée de Fourier
freq_range = np.arange(-F/2,F/2,F/num_samples)
#freq_range = np.fft.fftshift(np.fft.fftfreq(num_samples, 1 / F))


# Calcul de l'amplitude du spectre de fréquence
amplitude = np.abs(fft_result)

# Afficher le spectre de fréquence
plt.figure(3)
plt.plot(freq_range,(amplitude))
plt.xlabel('Fréquence (Hz)')
plt.ylabel('|FFT(X)|')
plt.title('Spectre de fréquence du signal en bande de base')
plt.grid()
plt.show(block=True)
plt.savefig("SPCTR.png")
