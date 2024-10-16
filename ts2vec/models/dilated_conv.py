import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        """
        Eine Dilatationsschicht wird hier initialisiert

        in_channels: Anzahl der Eingabekanäle
        out_channels: Anzahl der Ausgabekanäle
        kernel_size: Größe des Kernels 
        dilation: Dilatationsrate der Faltung
        groups: Anzahl der Gruppen bei der gruppierten Faltung
        """
        super().__init__()
        # Rezeptionsfeld der Faltung wird berechnet, das die effektive Größe des Bereichs beschreibt
        # auf diesen greift der Faltungskern beim Durchlauf über die Eingabe zugreift
        self.receptive_field = (kernel_size - 1) * dilation + 1

        # Berechnet das erforderliche Padding, um sicherzustellen, dass die Eingangs- und Ausgangsgröße der Faltung konsistent bleibt
        # Padding ist die Hälfte des Rezeptionsfelds, abgerundet
        padding = self.receptive_field // 2

        # Definiert die 1D.Faltungsschicht mit dem berechneten Padding + Dilatation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )

        # Wenn das Rezeptionsfeld eine gerade Zahl ist, muss ein Wert entfernt werden, damit die vorgesehene Größe erhalten bleibt
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        """
        Initialisiert einen Faltungsblock mit zwei Faltungsschichten 

        in_channels: Anzahl der Eingabekanäle
        out_channels: Anzahl der Ausgabekanäle
        kernel_size: Größe des Kernels 
        dilation: Dilatationsrate der Faltung
        final: Angabe ob es sich um den letzten Block handelt
        """
        # Erste Faltungsschicht
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        # Zweite Faltungsschicht
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        # Eingabe wird beim Residual beibehalten
        residual = x if self.projector is None else self.projector(x)

        # GELU-Aktivierungsfunktion angwendet -> glatte Aktivierungsfunktion, Vermeidung von Vanishing Gradient Problem
        # , basiert auf Gaußschen Fehlerfunktion somit probabilistische Interpretation
        x = F.gelu(x)
        # Erste Faltungsschicht
        x = self.conv1(x)
        x = F.gelu(x)
        # Zweite Faltungsschicht
        x = self.conv2(x)
        # Gibt die Summe aus der Faltungsschicht und Residuum zurück
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        # Sequenz von Faltungsblöcken werden erzeugt, damit nimmt die Dilatation exponentiell zu
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)
