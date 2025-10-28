import numpy as np
import matplotlib.pyplot as plt
from ControlSystem import ControlSystem

# Ga(s) = 100/((s+10)(s+20)) = 100/(s^2 + 30s + 200)
# PI: Gc(s) = Kp * (Ti*s + 1)/(Ti*s)
# Ti = 0.1
# Kp = 4*sqrt(2)

Ga_num = [100.0]
Ga_den = [1.0, 30.0, 200.0]

str_Gc = 'PI'
Ti = 0.1
Kp = 4 * np.sqrt(2)

reg = ControlSystem(Ga_num, Ga_den, str_Gc, Kp=Kp, Ti=Ti)

# ========== Informations sur le système ==========
print("="*60)
print("ANALYSE DU SYSTÈME DE CONTRÔLE")
print("="*60)

# Pôles et zéros
print("\n--- Pôles et Zéros ---")
print(f"Pôles en boucle ouverte: {reg.poles_open_loop}")
print(f"Zéros en boucle ouverte: {reg.zeros_open_loop}")
print(f"Pôles en boucle fermée: {reg.poles_closed_loop}")
print(f"Zéros en boucle fermée: {reg.zeros_closed_loop}")

# Erreur statique
print("\n--- Erreur Statique ---")
error_step = reg.static_error('step')
error_ramp = reg.static_error('ramp')
print(f"Erreur statique (échelon): {error_step:.6e}")
print(f"Erreur statique (rampe): {error_ramp:.6e}")

# Diagramme de Bode
omega = np.logspace(-1, 3, 2000)

# Calcul des marges de stabilité
gm, pm, wgc, wpc = reg.margin(omega)
print("\n--- Marges de Stabilité ---")
print(f"Marge de gain (Gm): {gm:.2f} dB")
print(f"Marge de phase (φ_m): {pm:.2f}°")
print(f"Pulsation de coupure de gain (ω_co): {wgc:.4f} rad/s")
print(f"Pulsation de coupure de phase (ω_pc): {wpc:.4f} rad/s")
print("="*60)
print()

# ========== Affichage de tous les diagrammes sur une seule fenêtre ==========
print("Affichage de tous les diagrammes sur une fenêtre...")
reg.plot_all(omega)