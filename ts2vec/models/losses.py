import torch
from torch import nn
import torch.nn.functional as F

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    """
    Hierarchische kontrastiver Verlust wird hier berechnet

    z1,z2: Eingabetensoren der Form (B,T,C)
    alpha: Gewicht für den instanzbasierten Verlust
    temporal_unit: Minimale Einheit ür den temporalen Kontrast
    """
    loss = torch.tensor(0., device=z1.device)
    d = 0

    #Schleife zur Berechnung des hierarchischen Verlusts, die läuft solange die Zeitdimension größer als 1 ist
    while z1.size(1) > 1:
        # Berechnet den Verlust für die jeweilige Instanz und fügt sie dem gesamten Loss hinzu
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)

        # Berechnet den temporalen Verlust und fügt ihn dem gesamten Verlust hinzu
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1

        #Reduzierung der Zeitdimension mittels max pooling
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    
    # Falls die Zeitdimension auf 1 liegt, wird der Verlust basieren auf der letzten Instanz berechnet
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    
    #Gibt den durchschnittlichen Verlust zurück
    return loss / d

def instance_contrastive_loss(z1, z2):
    """
    Instanzbasierter kontrastiver Verlust wird hier berechnet

    z1,z2: Eingabetensoren der Form (B,T,C)
    """
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    
    # Kombiniert Eingabetensor entlang der Batch-Dimension
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    #Berechnet Änhlichkeit zwischen den Paaren
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    #Berechnet Logits unter Ausschluss der Diagonale
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    # Verlust als Mittelwert der berechneten Logits
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    """
    Temporaler kontrastiver Verlust wird hier berechnet

    z1,z2: Eingabetensoren der Form (B,T,C)
    """
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    
    #Kombiniert Eingabetensor entlang der Zeitdimension
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    #Berrechnet  Änhlichkeit zwischen allemn Zeitpunkten
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    #Berechnet Logits unter Ausschluss der Diagonale
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    # Verlust als Mittelwert der berechneten Logits
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss
