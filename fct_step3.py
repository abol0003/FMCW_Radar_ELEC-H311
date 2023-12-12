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
       :return: Une matrice binaire où les valeurs au-dessus du seuil sont 1 et les autres 0.
       """
    normRDM_Won = rdm / np.max(rdm)  # Normaliser la RDM sans bruit
    binary_map = (normRDM_Won > threshold).astype(int)
    return binary_map
import numpy as np

def roc_curve_custom(binary_map, matrix_wn):

    # Triez les scores par ordre décroissant en donnant la possition dans la matrice 1x(N.K)
    sorted_indices = np.argsort(matrix_wn)[::-1]
    # rearange la binary map dans le même ordre
    sorted_ground_truth = binary_map[sorted_indices]

    # Initialisez les tableaux pour les taux de faux positifs (FPR) et les taux de vrais positifs (TPR)
    fpr = [0]
    tpr = [0]

    # Initialisez les compteurs pour les exemples positifs et négatifs
    num_positives = np.sum(binary_map)
    num_negatives = len(binary_map) - num_positives

    # Parcourez les scores triés
    for i in range(1, len(sorted_ground_truth) + 1):
        # Calculez le nombre actuel de faux positifs et vrais positifs
        fp = np.sum(sorted_ground_truth[:i] == 0)
        tp = np.sum(sorted_ground_truth[:i] == 1)

        # Ajoutez les taux de faux positifs et vrais positifs normalisés à la liste
        fpr.append(fp / num_negatives)
        tpr.append(tp / num_positives)

    # Calculez l'AUC en utilisant la méthode des trapèzes (intégrales)
    auc = np.trapz(tpr, fpr)

    return fpr, tpr, auc
