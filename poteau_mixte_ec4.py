"""
CALCUL POTEAU MIXTE EUROCODE 4 - APPROCHE PAR COURBE D'INTERACTION
Selon EN 1994-1-1 (NF EN 1994-1-1 juin 2005)

MÉTHODE : 
1. Calculer N_pl,Rd (résistance plastique en compression pure)
2. Calculer M_pl,Rd (résistance plastique en flexion pure)
3. Établir courbe d'interaction N-M
4. Pour N+M combinés, déduire les contraintes depuis la courbe
5. Flexion biaxiale selon article 6.7.3.7

TYPE DE SECTION : Profil creux circulaire rempli de béton (6.7.3.6)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple
import json
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import io
from contextlib import redirect_stdout

# ============================================================================
# BASES DE DONNÉES MATÉRIAUX (Eurocode 4)
# ============================================================================

CLASSES_ACIER = {
    'S235': {'fy': 235e6, 'fu': 360e6, 'E': 210e9},
    'S275': {'fy': 275e6, 'fu': 430e6, 'E': 210e9},
    'S355': {'fy': 355e6, 'fu': 510e6, 'E': 210e9},
    'S420': {'fy': 420e6, 'fu': 520e6, 'E': 210e9},
    'S460': {'fy': 460e6, 'fu': 540e6, 'E': 210e9},
}

CLASSES_BETON = {
    'C20': {'fck': 20e6, 'E': 30e9},
    'C25': {'fck': 25e6, 'E': 31e9},
    'C30': {'fck': 30e6, 'E': 33e9},
    'C35': {'fck': 35e6, 'E': 34e9},
    'C40': {'fck': 40e6, 'E': 35e9},
    'C45': {'fck': 45e6, 'E': 36e9},
    'C50': {'fck': 50e6, 'E': 37e9},
}

CLASSE_ARMATURE = {
    'FeE500': {'fyk': 500e6, 'E': 200e9},
    'FeE400': {'fyk': 400e6, 'E': 200e9},
}

GAMMA_PARTIEL = {
    'acier': 1.0,
    'beton': 1.5,
    'armature': 1.15,
}


# ============================================================================
# CLASSE MATÉRIAUX
# ============================================================================

@dataclass
class MateriauX:
    """Propriétés des matériaux selon EC4"""
    
    classe_acier: str = 'S355'
    classe_beton: str = 'C25'
    classe_armature: str = 'FeE500'
    
    def __post_init__(self):
        # Acier
        if self.classe_acier not in CLASSES_ACIER:
            raise ValueError(f"Classe acier {self.classe_acier} non reconnue")
        acier_props = CLASSES_ACIER[self.classe_acier]
        self.fy_acier = acier_props['fy']
        self.E_acier = acier_props['E']
        self.f_yd_acier = self.fy_acier / GAMMA_PARTIEL['acier']
        
        # Béton
        if self.classe_beton not in CLASSES_BETON:
            raise ValueError(f"Classe béton {self.classe_beton} non reconnue")
        beton_props = CLASSES_BETON[self.classe_beton]
        self.fck_beton = beton_props['fck']
        self.E_beton = beton_props['E']
        self.f_cd_beton = 0.85 * self.fck_beton / GAMMA_PARTIEL['beton']
        
        # Armature
        if self.classe_armature not in CLASSE_ARMATURE:
            raise ValueError(f"Classe armature {self.classe_armature} non reconnue")
        arm_props = CLASSE_ARMATURE[self.classe_armature]
        self.fyk_armature = arm_props['fyk']
        self.E_armature = arm_props['E']
        self.f_yd_armature = self.fyk_armature / GAMMA_PARTIEL['armature']


# ============================================================================
# CLASSE PRINCIPALE - EC4 COURBE D'INTERACTION
# ============================================================================

class PoteauMixteEC4:
    """
    Calcul poteau mixte circulaire selon EN 1994-1-1 (6.7.3.6)
    
    MÉTHODE :
    1. Calcul N_pl,Rd (résistance plastique compression)
    2. Calcul M_pl,Rd (résistance plastique flexion)
    3. Courbe d'interaction N-M
    4. Déduction des contraintes
    5. Vérification flexion biaxiale (6.7.3.7)
    """
    
    def __init__(self, 
                 D_ext: float,           # Diamètre extérieur (m)
                 t_paroi: float,         # Épaisseur paroi (m)
                 d_barre: float,         # Diamètre barre (m)
                 n_barres: int,          # Nombre barres
                 enrobage: float,        # Enrobage (m)
                 N: float,               # Effort normal (N)
                 My: float,              # Moment Y (N.m)
                 Mz: float,              # Moment Z (N.m)
                 classe_acier: str = 'S355',
                 classe_beton: str = 'C25',
                 classe_armature: str = 'FeE500'):
        
        self.D_ext = D_ext
        self.t_paroi = t_paroi
        self.d_barre = d_barre
        self.n_barres = n_barres
        self.enrobage = enrobage
        self.N = N
        self.My = My
        self.Mz = Mz
        
        self.mat = MateriauX(classe_acier=classe_acier, 
                           classe_beton=classe_beton,
                           classe_armature=classe_armature)
        
        self._verifier_entrees()
        self._calculer_geometrie()
        
    def _verifier_entrees(self):
        """Vérification des données"""
        assert self.D_ext > 0, "Diamètre > 0"
        assert self.t_paroi > 0, "Épaisseur > 0"
        assert self.t_paroi < self.D_ext / 2, "Épaisseur cohérente"
        assert self.d_barre > 0, "Diamètre barre > 0"
        assert self.n_barres > 0, "Nombre barres > 0"
        assert self.enrobage > 0, "Enrobage > 0"
        
    def _calculer_geometrie(self):
        """Géométrie de la section"""
        
        self.D_int = self.D_ext - 2 * self.t_paroi
        self.R_ext = self.D_ext / 2
        self.R_int = self.D_int / 2
        
        # Tube acier
        self.A_tube = np.pi * (self.R_ext**2 - self.R_int**2)
        self.I_tube = np.pi * (self.R_ext**4 - self.R_int**4) / 4
        
        # Béton
        self.A_beton = np.pi * self.R_int**2
        self.I_beton = np.pi * self.R_int**4 / 4
        
        # Armature
        self.r_barres = self.R_int - self.enrobage - self.d_barre / 2
        self.A_barre = np.pi * (self.d_barre / 2)**2
        self.A_armature = self.n_barres * self.A_barre
        self.I_armature = self.n_barres * self.A_barre * (self.r_barres**2)
        
        self.A_total = self.A_tube + self.A_beton + self.A_armature
        self.I_total = self.I_tube + self.I_beton + self.I_armature
        
    # ========================================================================
    # CALCULS EC4 - RÉSISTANCES PLASTIQUES
    # ========================================================================
    
    def calculer_N_pl_Rd(self) -> float:
        """
        Résistance plastique à la compression pure (EC4 6.7.3.2 et 6.7.3.6)
        
        FORMULE (6.33) - Section circulaire remplie de béton :
        ┌─────────────────────────────────────┐
        │ N_pl,Rd = A_a·f_yd + A_c·f_cd + A_s·f_sd │
        └─────────────────────────────────────┘
        
        où :
        - A_a = aire de l'acier du tube
        - A_c = aire du béton
        - A_s = aire de l'armature
        - f_yd = limite élastique acier / γ_M = f_y / 1.0
        - f_cd = résistance béton = 0.85·f_ck / γ_C = 0.85·f_ck / 1.5
        - f_sd = limite armature / γ_S = f_yk / 1.15
        """
        
        # Composante acier du tube
        N_a = self.A_tube * self.mat.f_yd_acier
        
        # Composante béton
        N_c = self.A_beton * self.mat.f_cd_beton
        
        # Composante armature
        N_s = self.A_armature * self.mat.f_yd_armature
        
        N_pl_Rd = N_a + N_c + N_s
        
        return N_pl_Rd, N_a, N_c, N_s
    
    def calculer_M_pl_Rd(self) -> float:
        """
        Moment résistant plastique en flexion uniaxiale (EC4 6.7.3.2)
        
        APPROCHE :
        La section travaille en flexion pure. L'axe neutre plastique divise
        la section en zones de compression et traction équilibrées.
        
        Pour une section circulaire :
        ┌──────────────────────────────────────────┐
        │ M_pl,Rd = Σ(force_i × bras_i)            │
        │ = N_a·R_ext + N_c·R_c + N_s·R_s           │
        └──────────────────────────────────────────┘
        
        où R_i sont les distances du centre de gravité au point d'application
        """
        
        # Acier du tube : travaille à la limite en traction/compression
        # Distance effective : rayon extérieur
        M_a = (self.A_tube * self.mat.f_yd_acier) * self.R_ext
        
        # Béton : distribution parabolique en compression
        # Distance moyenne pour section circulaire = 4R/(3π)
        R_c_avg = (4 * self.R_int) / (3 * np.pi)
        M_c = (self.A_beton * self.mat.f_cd_beton) * R_c_avg
        
        # Armature : répartie sur cercle de rayon r_barres
        M_s = (self.A_armature * self.mat.f_yd_armature) * self.r_barres
        
        M_pl_Rd = M_a + M_c + M_s
        
        return M_pl_Rd, M_a, M_c, M_s, R_c_avg
    
    def calculer_contraintes(self) -> Dict:
        """
        Calcul des contraintes par déduction de la courbe d'interaction
        
        APPROCHE EC4 :
        - Calculer N_pl,Rd et M_pl,Rd
        - Établir distribution linéaire des contraintes
        - Pour N+M combinés, déduire les contraintes
        """
        
        resultats = {}
        
        # Résistances plastiques
        N_pl_Rd, N_a, N_c, N_s = self.calculer_N_pl_Rd()
        M_pl_Rd, M_a, M_c, M_s, R_c_avg = self.calculer_M_pl_Rd()
        
        resultats['N_pl_Rd'] = N_pl_Rd
        resultats['M_pl_Rd'] = M_pl_Rd
        resultats['N_a'] = N_a
        resultats['N_c'] = N_c
        resultats['N_s'] = N_s
        resultats['M_a'] = M_a
        resultats['M_c'] = M_c
        resultats['M_s'] = M_s
        resultats['R_c_avg'] = R_c_avg
        
        # Inertie pour calcul des contraintes élastiques
        n_ac = self.mat.E_acier / self.mat.E_beton
        n_as = self.mat.E_armature / self.mat.E_beton
        
        I_eff = (self.I_beton + 
                 self.I_tube * n_ac + 
                 self.I_armature * n_as)
        
        resultats['I_eff'] = I_eff
        resultats['n_ac'] = n_ac
        resultats['n_as'] = n_as
        
        # Points critiques
        points = {
            'Top': (0, self.R_ext, 0),
            'Bottom': (0, -self.R_ext, 0),
            'Right': (0, 0, self.R_ext),
            'Left': (0, 0, -self.R_ext),
        }
        
        contraintes_beton = {}
        contraintes_tube = {}
        contraintes_armature = {}
        
        # Calcul des contraintes élastiques en chaque point
        # FORMULE (6.39-6.40) : Contraintes par flexion combinée
        for nom, (x, y, z) in points.items():
            
            # Contrainte axiale (effort normal seul)
            # σ_N = N / A_total
            if self.A_total > 0:
                sigma_N_beton = self.N / self.A_total
            else:
                sigma_N_beton = 0
            
            # Contraintes de flexion uniaxiale
            # σ_My = -M_y·z / I_eff  (moment autour axe Y)
            # σ_Mz = M_z·y / I_eff   (moment autour axe Z)
            if I_eff > 0:
                sigma_My_beton = -(self.My * z) / I_eff
                sigma_Mz_beton = (self.Mz * y) / I_eff
            else:
                sigma_My_beton = 0
                sigma_Mz_beton = 0
            
            # Contrainte totale dans béton (superposition)
            # σ_béton_total = σ_N + σ_My + σ_Mz
            sigma_beton = sigma_N_beton + sigma_My_beton + sigma_Mz_beton
            
            # Contraintes dans autres matériaux (section transformée)
            # σ_acier = n_ac · σ_béton
            # σ_armature = n_as · σ_béton
            sigma_tube = n_ac * sigma_beton
            sigma_armature = n_as * sigma_beton
            
            contraintes_beton[nom] = {
                'sigma_N': sigma_N_beton,
                'sigma_My': sigma_My_beton,
                'sigma_Mz': sigma_Mz_beton,
                'sigma_total': sigma_beton,
                'y': y,
                'z': z
            }
            
            contraintes_tube[nom] = {
                'sigma_total': sigma_tube,
                'y': y,
                'z': z
            }
            
            contraintes_armature[nom] = {
                'sigma_total': sigma_armature,
                'y': y,
                'z': z
            }
        
        resultats['contraintes_beton'] = contraintes_beton
        resultats['contraintes_tube'] = contraintes_tube
        resultats['contraintes_armature'] = contraintes_armature
        
        # Extrêmes
        sigma_beton_vals = [c['sigma_total'] for c in contraintes_beton.values()]
        sigma_tube_vals = [c['sigma_total'] for c in contraintes_tube.values()]
        sigma_arm_vals = [c['sigma_total'] for c in contraintes_armature.values()]
        
        resultats['sigma_beton_max'] = max(sigma_beton_vals)
        resultats['sigma_beton_min'] = min(sigma_beton_vals)
        resultats['sigma_tube_max'] = max(sigma_tube_vals)
        resultats['sigma_tube_min'] = min(sigma_tube_vals)
        resultats['sigma_armature_max'] = max(sigma_arm_vals)
        resultats['sigma_armature_min'] = min(sigma_arm_vals)
        
        # Ratios ELU
        resultats['ratio_beton'] = max(abs(resultats['sigma_beton_max']), abs(resultats['sigma_beton_min'])) / self.mat.f_cd_beton if self.mat.f_cd_beton > 0 else 0
        resultats['ratio_tube'] = max(abs(resultats['sigma_tube_max']), abs(resultats['sigma_tube_min'])) / self.mat.f_yd_acier if self.mat.f_yd_acier > 0 else 0
        resultats['ratio_armature'] = max(abs(resultats['sigma_armature_max']), abs(resultats['sigma_armature_min'])) / self.mat.f_yd_armature if self.mat.f_yd_armature > 0 else 0
        
        # Vérification flexion biaxiale (EC4 6.7.3.7)
        M_total = np.sqrt(self.My**2 + self.Mz**2)
        resultats['M_total'] = M_total
        resultats['ratio_moment'] = M_total / M_pl_Rd if M_pl_Rd > 0 else 0
        resultats['ratio_effort'] = abs(self.N) / N_pl_Rd if N_pl_Rd > 0 else 0
        
        return resultats
    
    def generer_courbe_interaction(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère la courbe d'interaction N-M
        
        Points : de N_pl,Rd pur à M_pl,Rd pur
        Interpolation linéaire selon EC4
        """
        
        N_pl_Rd, _, _, _ = self.calculer_N_pl_Rd()
        M_pl_Rd, _, _, _, _ = self.calculer_M_pl_Rd()
        
        # Points de la courbe
        n_points = 50
        
        # Point A : N_pl,Rd pur (M=0)
        # Point B-D : Interaction N-M (interpolation)
        # Point C : M_pl,Rd pur (N=0)
        
        N_values = np.linspace(0, N_pl_Rd, n_points)
        M_values = M_pl_Rd * (1 - N_values / N_pl_Rd)  # Interpolation linéaire
        
        return N_values, M_values
    
    def afficher_resultats(self):
        """Affichage complet des résultats avec détails des formules"""
        
        print("\n" + "╔" + "═"*100 + "╗")
        print("║" + " POTEAU MIXTE EUROCODE 4 - COURBE D'INTERACTION N-M ".center(100) + "║")
        print("║" + " EN 1994-1-1 (juin 2005) - Section 6.7.3 ".center(100) + "║")
        print("╚" + "═"*100 + "╝")
        
        print("\n📋 PARAMÈTRES D'ENTRÉE")
        print("─" * 102)
        print(f"  GÉOMÉTRIE:")
        print(f"    Diamètre extérieur:      {self.D_ext*1000:.1f} mm")
        print(f"    Diamètre intérieur:      {self.D_int*1000:.1f} mm")
        print(f"    Épaisseur paroi:         {self.t_paroi*1000:.2f} mm")
        print(f"    Barres armature:         {self.n_barres} × Ø{self.d_barre*1000:.1f} mm")
        print(f"    Rayon des barres:        {self.r_barres*1000:.1f} mm (du centre)")
        print(f"    Enrobage:                {self.enrobage*1000:.1f} mm")
        
        print(f"\n  MATÉRIAUX:")
        print(f"    Acier tube:              {self.mat.classe_acier}")
        print(f"      - f_y = {self.mat.fy_acier/1e6:.0f} MPa | f_yd = {self.mat.f_yd_acier/1e6:.0f} MPa (γ_M = {GAMMA_PARTIEL['acier']})")
        print(f"      - E = {self.mat.E_acier/1e9:.0f} GPa")
        
        print(f"\n    Béton:                   {self.mat.classe_beton}")
        print(f"      - f_ck = {self.mat.fck_beton/1e6:.0f} MPa | f_cd = 0.85·f_ck/γ_C = {self.mat.f_cd_beton/1e6:.2f} MPa (γ_C = {GAMMA_PARTIEL['beton']})")
        print(f"      - E = {self.mat.E_beton/1e9:.0f} GPa")
        
        print(f"\n    Armature:                {self.mat.classe_armature}")
        print(f"      - f_yk = {self.mat.fyk_armature/1e6:.0f} MPa | f_yd = {self.mat.f_yd_armature/1e6:.0f} MPa (γ_S = {GAMMA_PARTIEL['armature']})")
        print(f"      - E = {self.mat.E_armature/1e9:.0f} GPa")
        
        print(f"\n  SOLLICITATIONS:")
        print(f"    Effort normal:           {self.N/1000:+.2f} kN")
        print(f"    Moment Y:                {self.My/1e6:+.2f} kN.m")
        print(f"    Moment Z:                {self.Mz/1e6:+.2f} kN.m")
        
        print("\n" + "─"*102)
        print("📊 SECTIONS DES COMPOSANTS")
        print("─"*102)
        print(f"  A_tube = π(R_ext² - R_int²) = π({self.R_ext*1000:.2f}² - {self.R_int*1000:.2f}²) = {self.A_tube*1e6:.2f} mm²")
        print(f"  A_béton = π·R_int² = π·{self.R_int*1000:.2f}² = {self.A_beton*1e6:.2f} mm²")
        print(f"  A_armature = {self.n_barres}·π·(Ø/2)² = {self.n_barres}·π·({self.d_barre*1000:.1f}/2)² = {self.A_armature*1e6:.2f} mm²")
        print(f"  A_total = {self.A_total*1e6:.2f} mm²")
        
        print(f"\n  I_tube = π(R_ext⁴ - R_int⁴)/4 = {self.I_tube*1e12:.2f} mm⁴")
        print(f"  I_béton = π·R_int⁴/4 = {self.I_beton*1e12:.2f} mm⁴")
        print(f"  I_armature = {self.n_barres}·A_barre·r_barres² = {self.I_armature*1e12:.2f} mm⁴")
        print(f"  I_total = {self.I_total*1e12:.2f} mm⁴")
        
        resultats = self.calculer_contraintes()
        
        print("\n" + "─"*102)
        print("📊 RÉSISTANCES PLASTIQUES (EC4 6.7.3.2 - 6.7.3.6)")
        print("─"*102)
        
        print(f"\n  EFFORT NORMAL RÉSISTANT - Formule (6.33):")
        print(f"  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │ N_pl,Rd = A_a·f_yd + A_c·f_cd + A_s·f_sd                     │")
        print(f"  └─────────────────────────────────────────────────────────────┘")
        
        print(f"\n  Composantes:")
        print(f"    N_a (tube) = A_tube·f_yd = {self.A_tube*1e6:.2f}·{self.mat.f_yd_acier/1e6:.0f} = {resultats['N_a']/1000:.2f} kN")
        print(f"    N_c (béton) = A_béton·f_cd = {self.A_beton*1e6:.2f}·{self.mat.f_cd_beton/1e6:.2f} = {resultats['N_c']/1000:.2f} kN")
        print(f"    N_s (arm.) = A_arm·f_yd = {self.A_armature*1e6:.2f}·{self.mat.f_yd_armature/1e6:.0f} = {resultats['N_s']/1000:.2f} kN")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  N_pl,Rd = {resultats['N_pl_Rd']/1000:.2f} kN")
        
        print(f"\n  MOMENT RÉSISTANT EN FLEXION - Formule (dérivée EC4):")
        print(f"  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │ M_pl,Rd = N_a·R_ext + N_c·R_c_avg + N_s·r_barres            │")
        print(f"  └─────────────────────────────────────────────────────────────┘")
        
        print(f"\n  Bras de levier:")
        print(f"    R_ext = {self.R_ext*1000:.2f} mm (rayon extérieur)")
        print(f"    R_c_avg = 4·R_int/(3π) = 4·{self.R_int*1000:.2f}/(3π) = {resultats['R_c_avg']*1000:.2f} mm (béton)")
        print(f"    r_barres = {self.r_barres*1000:.2f} mm (armature)")
        
        print(f"\n  Composantes du moment:")
        print(f"    M_a = N_a·R_ext = {resultats['N_a']/1000:.2f}·{self.R_ext*1000:.2f} = {resultats['M_a']/1e6:.2f} kN.m")
        print(f"    M_c = N_c·R_c_avg = {resultats['N_c']/1000:.2f}·{resultats['R_c_avg']*1000:.2f} = {resultats['M_c']/1e6:.2f} kN.m")
        print(f"    M_s = N_s·r_barres = {resultats['N_s']/1000:.2f}·{self.r_barres*1000:.2f} = {resultats['M_s']/1e6:.2f} kN.m")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  M_pl,Rd = {resultats['M_pl_Rd']/1e6:.2f} kN.m")
        
        print("\n" + "─"*102)
        print("📊 INERTIE ÉQUIVALENTE (Section transformée)")
        print("─"*102)
        
        print(f"\n  Coefficients d'équivalence:")
        print(f"    n_ac = E_acier / E_béton = {self.mat.E_acier/1e9:.0f} / {self.mat.E_beton/1e9:.0f} = {resultats['n_ac']:.2f}")
        print(f"    n_as = E_armature / E_béton = {self.mat.E_armature/1e9:.0f} / {self.mat.E_beton/1e9:.0f} = {resultats['n_as']:.2f}")
        
        print(f"\n  FORMULE (6.40):")
        print(f"  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │ (EI)_eff = E_béton·(I_béton + n_ac·I_tube + n_as·I_arm.)    │")
        print(f"  └─────────────────────────────────────────────────────────────┘")
        
        print(f"\n  Calcul:")
        print(f"    I_eff = {self.I_beton*1e12:.2f} + {resultats['n_ac']:.2f}·{self.I_tube*1e12:.2f} + {resultats['n_as']:.2f}·{self.I_armature*1e12:.2f}")
        print(f"    I_eff = {resultats['I_eff']*1e12:.2f} mm⁴")
        
        print("\n" + "─"*102)
        print("⚙️ CONTRAINTES ÉLASTIQUES - BÉTON")
        print("─"*102)
        
        print(f"\n  FORMULE (superposition):")
        print(f"  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │ σ_béton(y,z) = N/A + My·z/I_eff + Mz·y/I_eff               │")
        print(f"  └─────────────────────────────────────────────────────────────┘")
        
        print(f"\n  {'Point':<12} {'σ_N (MPa)':>12} {'σ_My (MPa)':>12} {'σ_Mz (MPa)':>12} {'σ_total (MPa)':>15}")
        print("  " + "─" * 63)
        
        for nom, data in resultats['contraintes_beton'].items():
            print(f"  {nom:<12} {data['sigma_N']/1e6:>12.2f} {data['sigma_My']/1e6:>12.2f} "
                  f"{data['sigma_Mz']/1e6:>12.2f} {data['sigma_total']/1e6:>15.2f}")
        
        print(f"\n  Min/Max:")
        print(f"    σ_béton_max = {resultats['sigma_beton_max']/1e6:+.2f} MPa")
        print(f"    σ_béton_min = {resultats['sigma_beton_min']/1e6:+.2f} MPa")
        
        print("\n" + "─"*102)
        print("⚙️ CONTRAINTES ÉLASTIQUES - TUBE (Acier) et ARMATURE")
        print("─"*102)
        
        print(f"\n  FORMULE (section transformée):")
        print(f"  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │ σ_acier = n_ac · σ_béton                                    │")
        print(f"  │ σ_armature = n_as · σ_béton                                 │")
        print(f"  └─────────────────────────────────────────────────────────────┘")
        
        print(f"\n  TUBE (Acier):")
        for nom, data in resultats['contraintes_tube'].items():
            print(f"    {nom:<12} {data['sigma_total']/1e6:>25.2f} MPa")
        print(f"    Max/Min:     {resultats['sigma_tube_max']/1e6:>25.2f} / {resultats['sigma_tube_min']/1e6:.2f} MPa")
        
        print(f"\n  ARMATURE:")
        for nom, data in resultats['contraintes_armature'].items():
            print(f"    {nom:<12} {data['sigma_total']/1e6:>25.2f} MPa")
        print(f"    Max/Min:     {resultats['sigma_armature_max']/1e6:>25.2f} / {resultats['sigma_armature_min']/1e6:.2f} MPa")
        
        print("\n" + "─"*102)
        print("✅ VÉRIFICATIONS ELU (EC4 6.7.3.7 - FLEXION BIAXIALE)")
        print("─"*102)
        
        print(f"\n  TUBE ({self.mat.classe_acier}):")
        print(f"    - Résistance de calcul f_yd = {self.mat.f_yd_acier/1e6:.0f} MPa")
        print(f"    - Contrainte max (abs) = {max(abs(resultats['sigma_tube_max']), abs(resultats['sigma_tube_min']))/1e6:.2f} MPa")
        print(f"    - Ratio σ/f_yd = {resultats['ratio_tube']:.4f} {'✓ OK' if resultats['ratio_tube'] <= 1.0 else '✗ NON OK'}")
        
        print(f"\n  BÉTON ({self.mat.classe_beton}):")
        print(f"    - Résistance de calcul f_cd = {self.mat.f_cd_beton/1e6:.2f} MPa")
        print(f"    - Contrainte max (abs) = {max(abs(resultats['sigma_beton_max']), abs(resultats['sigma_beton_min']))/1e6:.2f} MPa")
        print(f"    - Ratio σ/f_cd = {resultats['ratio_beton']:.4f} {'✓ OK' if resultats['ratio_beton'] <= 1.0 else '✗ NON OK'}")
        
        print(f"\n  ARMATURE ({self.mat.classe_armature}):")
        print(f"    - Résistance de calcul f_yd = {self.mat.f_yd_armature/1e6:.0f} MPa")
        print(f"    - Contrainte max (abs) = {max(abs(resultats['sigma_armature_max']), abs(resultats['sigma_armature_min']))/1e6:.2f} MPa")
        print(f"    - Ratio σ/f_yd = {resultats['ratio_armature']:.4f} {'✓ OK' if resultats['ratio_armature'] <= 1.0 else '✗ NON OK'}")
        
        print(f"\n  INTERACTION N-M (Équations 6.46-6.47):")
        print(f"    - M_total = √(My² + Mz²) = √({self.My/1e6:.2f}² + {self.Mz/1e6:.2f}²) = {resultats['M_total']/1e6:.2f} kN.m")
        print(f"    - Ratio M/M_pl,Rd = {resultats['M_total']/1e6:.2f} / {resultats['M_pl_Rd']/1e6:.2f} = {resultats['ratio_moment']:.4f}")
        print(f"    - Ratio N/N_pl,Rd = {abs(self.N)/1000:.2f} / {resultats['N_pl_Rd']/1000:.2f} = {resultats['ratio_effort']:.4f}")
        
        global_ok = (resultats['ratio_tube'] <= 1.0 and 
                    resultats['ratio_beton'] <= 1.0 and 
                    resultats['ratio_armature'] <= 1.0)
        
        print(f"\n  {'🎉 TOUS LES CRITÈRES SATISFAITS ✓' if global_ok else '⚠️  CRITÈRES NON SATISFAITS ✗'}")
        
        print("╚" + "═"*100 + "╝\n")
        
        return resultats
    
    def tracer_courbe_interaction(self, resultats: Dict):
        """Trace la courbe d'interaction N-M"""
        
        N_pl_Rd = resultats['N_pl_Rd']
        M_pl_Rd = resultats['M_pl_Rd']
        
        # Génère la courbe
        N_courbe, M_courbe = self.generer_courbe_interaction()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Courbe d'interaction
        ax.plot(M_courbe/1e6, N_courbe/1000, 'b-', linewidth=2.5, label='Courbe d\'interaction')
        ax.fill_between(M_courbe/1e6, 0, N_courbe/1000, alpha=0.1, color='blue')
        
        # Points caractéristiques
        ax.plot(0, N_pl_Rd/1000, 'go', markersize=10, label=f'A: N_pl,Rd = {N_pl_Rd/1000:.0f} kN')
        ax.plot(M_pl_Rd/1e6, 0, 'ro', markersize=10, label=f'C: M_pl,Rd = {M_pl_Rd/1e6:.2f} kN.m')
        
        # Point de calcul
        M_total = np.sqrt(self.My**2 + self.Mz**2)
        ax.plot(M_total/1e6, abs(self.N)/1000, 'k*', markersize=20, label=f'Point de calcul: N={self.N/1000:.0f}kN, M={M_total/1e6:.2f}kN.m')
        
        ax.set_xlabel('Moment (kN.m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Effort normal (kN)', fontsize=12, fontweight='bold')
        ax.set_title('Courbe d\'Interaction N-M\nPoteau Mixte Circulaire - EC4', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper right')
        ax.set_xlim(0, M_pl_Rd/1e6 * 1.1)
        ax.set_ylim(0, N_pl_Rd/1000 * 1.1)
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# INTERFACE GRAPHIQUE
# ============================================================================

def calculer():
    try:
        N_input = ent_N.get().strip()
        if not N_input:
            N_value = 0
        else:
            N_value = float(N_input)
        
        poteau = PoteauMixteEC4(
            D_ext=float(ent_Dext.get()) / 1000,
            t_paroi=float(ent_t.get()) / 1000,
            d_barre=float(ent_barre.get()) / 1000,
            n_barres=int(ent_nb.get()),
            enrobage=float(ent_enrobage.get()) / 1000,
            
            N=N_value * 1000,
            My=float(ent_My.get()) * 1000,
            Mz=float(ent_Mz.get()) * 1000,
            
            classe_acier=cb_acier.get(),
            classe_beton=cb_beton.get(),
            classe_armature=cb_armature.get()
        )
        
        buffer = io.StringIO()
        
        with redirect_stdout(buffer):
            resultats = poteau.afficher_resultats()
        
        txt_resultats.delete("1.0", tk.END)
        txt_resultats.insert(tk.END, buffer.getvalue())
        
        if chk_graph.get():
            poteau.tracer_courbe_interaction(resultats)
    
    except Exception as e:
        messagebox.showerror("Erreur", str(e))


root = tk.Tk()
root.title("POTEAU MIXTE EUROCODE 4 - Courbe d'Interaction N-M")
root.geometry("1600x1000")

frame = ttk.LabelFrame(root, text="Données d'entrée")
frame.pack(fill="x", padx=10, pady=10)

ttk.Label(frame, text="Diamètre extérieur Dext (mm)").grid(row=0, column=0, padx=5, pady=5, sticky="w")
ent_Dext = ttk.Entry(frame, width=15)
ent_Dext.insert(0, "323")
ent_Dext.grid(row=0, column=1)

ttk.Label(frame, text="Épaisseur tube t (mm)").grid(row=0, column=2, padx=5)
ent_t = ttk.Entry(frame, width=15)
ent_t.insert(0, "10")
ent_t.grid(row=0, column=3)

ttk.Label(frame, text="Diamètre barre (mm)").grid(row=0, column=4, padx=5)
ent_barre = ttk.Entry(frame, width=15)
ent_barre.insert(0, "16")
ent_barre.grid(row=0, column=5)

ttk.Label(frame, text="Nombre de barres").grid(row=1, column=0, padx=5, pady=5)
ent_nb = ttk.Entry(frame, width=15)
ent_nb.insert(0, "4")
ent_nb.grid(row=1, column=1)

ttk.Label(frame, text="Enrobage (mm)").grid(row=1, column=2)
ent_enrobage = ttk.Entry(frame, width=15)
ent_enrobage.insert(0, "40")
ent_enrobage.grid(row=1, column=3)

ttk.Label(frame, text="Effort N (kN) [vide = 0]").grid(row=2, column=0)
ent_N = ttk.Entry(frame, width=15)
ent_N.insert(0, "")
ent_N.grid(row=2, column=1)

ttk.Label(frame, text="Moment My (kN.m)").grid(row=2, column=2)
ent_My = ttk.Entry(frame, width=15)
ent_My.insert(0, "50")
ent_My.grid(row=2, column=3)

ttk.Label(frame, text="Moment Mz (kN.m)").grid(row=2, column=4)
ent_Mz = ttk.Entry(frame, width=15)
ent_Mz.insert(0, "30")
ent_Mz.grid(row=2, column=5)

ttk.Label(frame, text="Classe acier").grid(row=3, column=0)
cb_acier = ttk.Combobox(frame, values=list(CLASSES_ACIER.keys()), width=12)
cb_acier.set("S355")
cb_acier.grid(row=3, column=1)

ttk.Label(frame, text="Classe béton").grid(row=3, column=2)
cb_beton = ttk.Combobox(frame, values=list(CLASSES_BETON.keys()), width=12)
cb_beton.set("C25")
cb_beton.grid(row=3, column=3)

ttk.Label(frame, text="Classe armature").grid(row=3, column=4)
cb_armature = ttk.Combobox(frame, values=list(CLASSE_ARMATURE.keys()), width=12)
cb_armature.set("FeE500")
cb_armature.grid(row=3, column=5)

frame_btn = ttk.Frame(root)
frame_btn.pack(fill="x")

chk_graph = tk.BooleanVar(value=True)
ttk.Checkbutton(frame_btn, text="Afficher courbe d'interaction", variable=chk_graph).pack(side="left", padx=10)
ttk.Button(frame_btn, text="CALCULER", command=calculer).pack(side="left", padx=10)

frame_resultats = ttk.LabelFrame(root, text="Résultats détaillés EC4 - Formules et Calculs")
frame_resultats.pack(fill="both", expand=True, padx=10, pady=10)

scroll = tk.Scrollbar(frame_resultats)
scroll.pack(side="right", fill="y")

txt_resultats = tk.Text(frame_resultats, font=("Courier", 8.5), yscrollcommand=scroll.set)
txt_resultats.pack(fill="both", expand=True)

scroll.config(command=txt_resultats.yview)

root.mainloop()
