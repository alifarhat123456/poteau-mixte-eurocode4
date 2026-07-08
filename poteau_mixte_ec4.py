"""
CALCUL DES CONTRAINTES DANS UN POTEAU MIXTE (TUBE REMPLI DE BÉTON)
Selon l'Eurocode 4 (EN 1994-1-1)

Type: Poteau mixte circulaire type "E" (tube enrobé)

ENTRÉES PRINCIPALES:
  Géométrie:
    - Diamètre extérieur du tube (D_ext)
    - Épaisseur de la paroi du tube (t_paroi)
    - Diamètre des barres de ferraillage (d_barre)
    - Nombre de barres de ferraillage (n_barres) - réparties équitablement
    - Enrobage du béton (c)

  Matériaux (VARIABLES):
    - Classe d'acier du tube (S235, S275, S355, S420, S460...)
    - Classe de béton (C20, C25, C30, C35, C40, C45, C50...)

  Sollicitations (TORSEUR):
    - Effort normal (N) - positif = compression
    - Moment fléchissant MY (autour axe Y)
    - Moment fléchissant MZ (autour axe Z)

SORTIES:
  - Contraintes dans le tube métallique
  - Contraintes dans le béton
  - Contraintes dans l'armature
  - Vérifications ELU biaxiales
  - Graphiques de distribution
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

# Propriétés de l'acier selon la classe
CLASSES_ACIER = {
    'S235': {'fy': 235e6, 'fu': 360e6, 'E': 210e9},
    'S275': {'fy': 275e6, 'fu': 430e6, 'E': 210e9},
    'S355': {'fy': 355e6, 'fu': 510e6, 'E': 210e9},
    'S420': {'fy': 420e6, 'fu': 520e6, 'E': 210e9},
    'S460': {'fy': 460e6, 'fu': 540e6, 'E': 210e9},
}

# Propriétés du béton selon la classe (EN 206)
CLASSES_BETON = {
    'C20': {'fck': 20e6, 'E': 30e9},
    'C25': {'fck': 25e6, 'E': 31e9},
    'C30': {'fck': 30e6, 'E': 33e9},
    'C35': {'fck': 35e6, 'E': 34e9},
    'C40': {'fck': 40e6, 'E': 35e9},
    'C45': {'fck': 45e6, 'E': 36e9},
    'C50': {'fck': 50e6, 'E': 37e9},
}

# Classe d'armature
CLASSE_ARMATURE = {
    'FeE500': {'fyk': 500e6, 'E': 200e9},
    'FeE400': {'fyk': 400e6, 'E': 200e9},
}

# Coefficients partiels de sécurité (Eurocode 4)
GAMMA_PARTIEL = {
    'acier': 1.0,      # γ_M (acier)
    'beton': 1.5,      # γ_C (béton)
    'armature': 1.15,  # γ_S (armature)
}


# ============================================================================
# CLASSE POUR LES MATÉRIAUX
# ============================================================================

@dataclass
class MateriauX:
    """Paramètres des matériaux - Version paramétrable"""
    
    classe_acier: str = 'S355'      # Classe d'acier du tube
    classe_beton: str = 'C25'       # Classe de béton
    classe_armature: str = 'FeE500' # Classe d'armature
    
    def __post_init__(self):
        """Initialisation basée sur les classes"""
        
        # Acier du tube
        if self.classe_acier not in CLASSES_ACIER:
            raise ValueError(f"Classe acier {self.classe_acier} non reconnue. Choix: {list(CLASSES_ACIER.keys())}")
        
        acier_props = CLASSES_ACIER[self.classe_acier]
        self.fy_acier = acier_props['fy']
        self.fu_acier = acier_props['fu']
        self.E_acier = acier_props['E']
        self.gamma_ma = GAMMA_PARTIEL['acier']
        
        # Béton
        if self.classe_beton not in CLASSES_BETON:
            raise ValueError(f"Classe béton {self.classe_beton} non reconnue. Choix: {list(CLASSES_BETON.keys())}")
        
        beton_props = CLASSES_BETON[self.classe_beton]
        self.fck_beton = beton_props['fck']
        self.E_beton = beton_props['E']
        self.fcd_beton = 0.85 * self.fck_beton / GAMMA_PARTIEL['beton']
        self.gamma_c = GAMMA_PARTIEL['beton']
        
        # Armature
        if self.classe_armature not in CLASSE_ARMATURE:
            raise ValueError(f"Classe armature {self.classe_armature} non reconnue. Choix: {list(CLASSE_ARMATURE.keys())}")
        
        armature_props = CLASSE_ARMATURE[self.classe_armature]
        self.fyk_armature = armature_props['fyk']
        self.E_armature = armature_props['E']
        self.fyd_armature = self.fyk_armature / GAMMA_PARTIEL['armature']
        self.gamma_s = GAMMA_PARTIEL['armature']


# ============================================================================
# CLASSE PRINCIPALE: POTEAU MIXTE AVEC FLEXION BIAXIALE - EC4
# ============================================================================

class PoteauMixteEC4:
    """
    Calcul des contraintes dans un poteau mixte circulaire (tube enrobé)
    avec flexion biaxiale selon l'Eurocode 4 (EN 1994-1-1)
    
    Méthode: Sections transformées élastiques avec calcul des contraintes
    dans chaque matériau (béton, acier tube, armatures)
    """
    
    def __init__(self, 
                 D_ext: float,           # Diamètre extérieur du tube (m)
                 t_paroi: float,         # Épaisseur de la paroi du tube (m)
                 d_barre: float,         # Diamètre des barres (m)
                 n_barres: int,          # Nombre de barres
                 enrobage: float,        # Enrobage du béton (m)
                 N: float,               # Effort normal (N) - positif = compression
                 My: float,              # Moment fléchissant autour axe Y (N.m)
                 Mz: float,              # Moment fléchissant autour axe Z (N.m)
                 classe_acier: str = 'S355',
                 classe_beton: str = 'C25',
                 classe_armature: str = 'FeE500'):
        """
        Initialisation du poteau mixte avec flexion biaxiale (EC4)
        """
        
        self.D_ext = D_ext
        self.t_paroi = t_paroi
        self.d_barre = d_barre
        self.n_barres = n_barres
        self.enrobage = enrobage
        self.N = N
        self.My = My
        self.Mz = Mz
        
        # Création de l'objet matériaux
        self.mat = MateriauX(classe_acier=classe_acier, 
                           classe_beton=classe_beton,
                           classe_armature=classe_armature)
        
        # Vérifications d'entrée
        self._verifier_entrees()
        
        # Calcul des propriétés géométriques
        self._calculer_geometrie()
        
    def _verifier_entrees(self):
        """Vérification des données d'entrée"""
        assert self.D_ext > 0, "Diamètre extérieur doit être > 0"
        assert self.t_paroi > 0, "Épaisseur doit être > 0"
        assert self.t_paroi < self.D_ext / 2, "Épaisseur trop grande"
        assert self.d_barre > 0, "Diamètre barre doit être > 0"
        assert self.n_barres > 0, "Nombre de barres doit être > 0"
        assert self.enrobage > 0, "Enrobage doit être > 0"
        
    def _calculer_geometrie(self):
        """Calcul des propriétés géométriques"""
        
        # Diamètre intérieur
        self.D_int = self.D_ext - 2 * self.t_paroi
        
        # Rayons
        self.R_ext = self.D_ext / 2
        self.R_int = self.D_int / 2
        
        # SECTION TUBE MÉTALLIQUE
        self.A_tube = np.pi * (self.R_ext**2 - self.R_int**2)
        self.I_tube = np.pi * (self.R_ext**4 - self.R_int**4) / 4
        
        # BÉTON INTÉRIEUR
        self.A_beton = np.pi * self.R_int**2
        self.I_beton = np.pi * self.R_int**4 / 4
        
        # ARMATURE (barres uniformément réparties)
        # Rayon des barres (distance du centre aux barres)
        self.r_barres = self.R_int - self.enrobage - self.d_barre / 2
        
        # Section d'une barre
        self.A_barre = np.pi * (self.d_barre / 2)**2
        
        # Aire totale d'armature
        self.A_armature = self.n_barres * self.A_barre
        
        # Inertie de l'armature (barres uniformément réparties sur cercle)
        self.I_armature = self.n_barres * self.A_barre * (self.r_barres**2)
        
        # SECTION TOTALE ET INERTIE (béton seul)
        self.A_total = self.A_tube + self.A_beton + self.A_armature
        self.I_total = self.I_tube + self.I_beton + self.I_armature
        
    def calculer_contraintes(self) -> Dict:
        """
        Calcul des contraintes avec flexion biaxiale selon Eurocode 4
        
        Méthode EC4: Section transformée élastique
        - Les modules d'élasticité différents sont pris en compte
        - Coefficients d'équivalence: n_ac = E_acier/E_beton, n_as = E_armature/E_beton
        - Inertie et aire équivalentes (rapportées au béton)
        - Calcul des contraintes dans chaque matériau
        
        Returns:
            Dict contenant les contraintes et résultats détaillés
        """
        
        resultats = {}
        
        # 1. COEFFICIENTS D'ÉQUIVALENCE (EC4 - Section transformée)
        n_ac = self.mat.E_acier / self.mat.E_beton  # Acier tube / Béton
        n_as = self.mat.E_armature / self.mat.E_beton  # Armature / Béton
        
        # 2. AIRE ÉQUIVALENTE (rapportée au béton)
        A_eq_tube = self.A_tube * n_ac
        A_eq_armature = self.A_armature * n_as
        A_eq_total = self.A_beton + A_eq_tube + A_eq_armature
        
        # 3. INERTIE ÉQUIVALENTE (rapportée au béton)
        I_eq = (self.I_beton + 
                self.I_tube * n_ac + 
                self.I_armature * n_as)
        
        resultats['n_ac'] = n_ac
        resultats['n_as'] = n_as
        resultats['A_eq_total'] = A_eq_total
        resultats['I_eq'] = I_eq
        
        # 4. DÉFORMATION AXIALE MOYENNE (effort normal)
        eps_0 = self.N / (A_eq_total * self.mat.E_beton) if A_eq_total > 0 else 0
        
        # 5. COURBURES DE FLEXION (pour MY et MZ)
        # κ = M / (EI)  => EI équivalent en béton
        kappa_y = self.My / I_eq if I_eq > 0 else 0  # Courbure autour Y
        kappa_z = self.Mz / I_eq if I_eq > 0 else 0  # Courbure autour Z
        
        # 6. OBSERVATION AUX 4 POINTS CRITIQUES
        points_obs = {
            'Top': (0, self.R_ext, 0),      # (x, y, z)
            'Bottom': (0, -self.R_ext, 0),
            'Right': (0, 0, self.R_ext),
            'Left': (0, 0, -self.R_ext),
        }
        
        contraintes_beton = {}
        contraintes_tube = {}
        contraintes_armature = {}
        
        sigma_max_all = -np.inf
        sigma_min_all = np.inf
        
        for nom, (x, y, z) in points_obs.items():
            # ===== CONTRAINTE DANS LE BÉTON =====
            # σ_beton = (N/Aeq) - kappa_y * z - kappa_z * y
            sigma_N_beton = (self.N / A_eq_total) if A_eq_total > 0 else 0
            sigma_My_beton = -kappa_y * z * self.mat.E_beton  # Convention: - pour flexion
            sigma_Mz_beton = -kappa_z * y * self.mat.E_beton
            
            sigma_total_beton = sigma_N_beton + sigma_My_beton + sigma_Mz_beton
            
            # ===== CONTRAINTE DANS LE TUBE (acier) =====
            # Même déformation qu'au béton, mais E_acier au lieu E_beton
            # σ_tube = n_ac * σ_beton (à la même position)
            sigma_total_tube = n_ac * sigma_total_beton
            
            # ===== CONTRAINTE DANS L'ARMATURE =====
            # Même déformation qu'au béton, mais E_armature au lieu E_beton
            sigma_total_armature = n_as * sigma_total_beton
            
            contraintes_beton[nom] = {
                'sigma_N': sigma_N_beton,
                'sigma_My': sigma_My_beton,
                'sigma_Mz': sigma_Mz_beton,
                'sigma_total': sigma_total_beton,
                'y': y,
                'z': z
            }
            
            contraintes_tube[nom] = {
                'sigma_total': sigma_total_tube,
                'y': y,
                'z': z
            }
            
            contraintes_armature[nom] = {
                'sigma_total': sigma_total_armature,
                'y': y,
                'z': z
            }
            
            # Mettre à jour les extrêmes
            sigma_max_all = max(sigma_max_all, sigma_total_beton, sigma_total_tube, sigma_total_armature)
            sigma_min_all = min(sigma_min_all, sigma_total_beton, sigma_total_tube, sigma_total_armature)
        
        resultats['deformation_axiale'] = eps_0
        resultats['kappa_y'] = kappa_y
        resultats['kappa_z'] = kappa_z
        resultats['contraintes_beton'] = contraintes_beton
        resultats['contraintes_tube'] = contraintes_tube
        resultats['contraintes_armature'] = contraintes_armature
        resultats['sigma_max'] = sigma_max_all
        resultats['sigma_min'] = sigma_min_all
        
        # Contraintes extrêmes par matériau
        sigma_tube_max = max([abs(c['sigma_total']) for c in contraintes_tube.values()])
        sigma_beton_max = max([abs(c['sigma_total']) for c in contraintes_beton.values()])
        sigma_armature_max = max([abs(c['sigma_total']) for c in contraintes_armature.values()])
        
        resultats['sigma_tube_max'] = sigma_tube_max
        resultats['sigma_beton_max'] = sigma_beton_max
        resultats['sigma_armature_max'] = sigma_armature_max
        
        # Ratios ELU
        f_yd_tube = self.mat.fy_acier / self.mat.gamma_ma
        f_cd = self.mat.fcd_beton
        f_yd_arm = self.mat.fyd_armature
        
        resultats['ratio_tube'] = sigma_tube_max / f_yd_tube if f_yd_tube > 0 else 0
        resultats['ratio_beton'] = sigma_beton_max / f_cd if f_cd > 0 else 0
        resultats['ratio_armature'] = sigma_armature_max / f_yd_arm if f_yd_arm > 0 else 0
        
        resultats['f_yd_tube'] = f_yd_tube
        resultats['f_cd'] = f_cd
        resultats['f_yd_arm'] = f_yd_arm
        
        return resultats
    
    def verifications_ELU(self, resultats: Dict) -> Dict:
        """
        Vérifications des états limites ultimes (ELU) - EC4
        """
        
        verif = {}
        
        # TUBE MÉTALLIQUE
        sigma_tube_max = resultats['sigma_tube_max']
        f_yd_tube = resultats['f_yd_tube']
        ratio_tube = resultats['ratio_tube']
        verif['tube_OK'] = ratio_tube <= 1.0
        verif['ratio_tube'] = ratio_tube
        verif['sigma_tube_max_MPa'] = sigma_tube_max / 1e6
        verif['f_yd_tube_MPa'] = f_yd_tube / 1e6
        
        # BÉTON
        sigma_beton_max = resultats['sigma_beton_max']
        f_cd = resultats['f_cd']
        ratio_beton = resultats['ratio_beton']
        verif['beton_OK'] = ratio_beton <= 1.0
        verif['ratio_beton'] = ratio_beton
        verif['sigma_beton_max_MPa'] = sigma_beton_max / 1e6
        verif['f_cd_beton_MPa'] = f_cd / 1e6
        
        # ARMATURE
        sigma_armature_max = resultats['sigma_armature_max']
        f_yd_arm = resultats['f_yd_arm']
        ratio_armature = resultats['ratio_armature']
        verif['armature_OK'] = ratio_armature <= 1.0
        verif['ratio_armature'] = ratio_armature
        verif['sigma_armature_max_MPa'] = sigma_armature_max / 1e6
        verif['f_yd_armature_MPa'] = f_yd_arm / 1e6
        
        # VÉRIFICATION GLOBALE
        verif['tous_OK'] = verif['tube_OK'] and verif['beton_OK'] and verif['armature_OK']
        
        return verif
    
    def afficher_resultats(self):
        """Affichage complet des résultats"""
        
        print("\n" + "╔" + "═"*86 + "╗")
        print("║" + " CALCUL POTEAU MIXTE SELON EUROCODE 4 (EN 1994-1-1) - FLEXION BIAXIALE ".center(86) + "║")
        print("╚" + "═"*86 + "╝")
        
        # Paramètres d'entrée
        print("\n📋 PARAMÈTRES D'ENTRÉE")
        print("─" * 88)
        print(f"  GÉOMÉTRIE:")
        print(f"    Diamètre extérieur du tube (D_ext):     {self.D_ext*1000:.1f} mm")
        print(f"    Épaisseur de la paroi (t):              {self.t_paroi*1000:.2f} mm")
        print(f"    Diamètre des barres de ferraillage:     {self.d_barre*1000:.1f} mm")
        print(f"    Nombre de barres:                       {self.n_barres}")
        print(f"    Enrobage du béton:                      {self.enrobage*1000:.1f} mm")
        
        print(f"\n  MATÉRIAUX:")
        print(f"    Classe d'acier du tube:                 {self.mat.classe_acier}")
        print(f"    Classe de béton:                        {self.mat.classe_beton}")
        print(f"    Classe d'armature:                      {self.mat.classe_armature}")
        
        print(f"\n  SOLLICITATIONS (TORSEUR):")
        print(f"    Effort normal (N):                      {self.N/1000:+.2f} kN (compression)")
        print(f"    Moment fléchissant autour Y (MY):       {self.My/1e6:+.2f} kN.m")
        print(f"    Moment fléchissant autour Z (MZ):       {self.Mz/1e6:+.2f} kN.m")
        moment_resultant = np.sqrt(self.My**2 + self.Mz**2)
        print(f"    Moment résultant (M_tot):               {moment_resultant/1e6:.2f} kN.m")
        
        # Propriétés géométriques
        print("\n📐 PROPRIÉTÉS GÉOMÉTRIQUES")
        print("─" * 88)
        print(f"  Diamètre intérieur (D_int):            {self.D_int*1000:.1f} mm")
        print(f"  Section tube (A_tube):                 {self.A_tube*1e6:.2f} mm²")
        print(f"  Section béton (A_beton):               {self.A_beton*1e6:.2f} mm²")
        print(f"  Aire armature totale:                  {self.A_armature*1e6:.2f} mm²")
        print(f"  Section totale (A_tot):                {self.A_total*1e6:.2f} mm²")
        print(f"  Inertie totale (I):                    {self.I_total*1e12:.2f} mm⁴")
        print(f"  Rayon des barres (depuis centre):      {self.r_barres*1000:.1f} mm")
        
        # Propriétés matériaux
        print("\n🔧 PROPRIÉTÉS DES MATÉRIAUX")
        print("─" * 88)
        print(f"  Acier tube ({self.mat.classe_acier}):")
        print(f"    - Limite élastique fy:               {self.mat.fy_acier/1e6:.0f} MPa")
        print(f"    - Résistance de calcul f_yd:         {self.mat.fy_acier/self.mat.gamma_ma/1e6:.0f} MPa")
        print(f"    - Module d'Young E:                  {self.mat.E_acier/1e9:.0f} GPa")
        
        print(f"\n  Béton ({self.mat.classe_beton}):")
        print(f"    - Résistance caractéristique fck:    {self.mat.fck_beton/1e6:.0f} MPa")
        print(f"    - Résistance de calcul fcd:          {self.mat.fcd_beton/1e6:.2f} MPa")
        print(f"    - Module d'Young E:                  {self.mat.E_beton/1e9:.0f} GPa")
        
        print(f"\n  Armature ({self.mat.classe_armature}):")
        print(f"    - Limite élastique fyk:              {self.mat.fyk_armature/1e6:.0f} MPa")
        print(f"    - Résistance de calcul fyd:          {self.mat.fyd_armature/1e6:.0f} MPa")
        print(f"    - Module d'Young E:                  {self.mat.E_armature/1e9:.0f} GPa")
        
        # Calcul des contraintes
        resultats = self.calculer_contraintes()
        
        print("\n⚙️ CONTRAINTES CALCULÉES (DOMAINE ÉLASTIQUE - EC4)")
        print("─" * 88)
        print(f"  Déformation axiale (ε):                {resultats['deformation_axiale']*1e6:.2f} µm/m")
        print(f"  Coefficient d'équivalence n_ac:        {resultats['n_ac']:.2f}")
        print(f"  Coefficient d'équivalence n_as:        {resultats['n_as']:.2f}")
        print(f"  Courbure autour Y (κy):                {resultats['kappa_y']:.6f} m⁻¹")
        print(f"  Courbure autour Z (κz):                {resultats['kappa_z']:.6f} m⁻¹")
        
        print(f"\n  CONTRAINTES DANS LE BÉTON:")
        print(f"  {'Point':<12} {'σ_N (MPa)':>12} {'σ_My (MPa)':>12} {'σ_Mz (MPa)':>12} {'σ_total (MPa)':>15}")
        print("  " + "─" * 63)
        
        for nom, data in resultats['contraintes_beton'].items():
            print(f"  {nom:<12} {data['sigma_N']/1e6:>12.2f} {data['sigma_My']/1e6:>12.2f} "
                  f"{data['sigma_Mz']/1e6:>12.2f} {data['sigma_total']/1e6:>15.2f}")
        
        print(f"\n  CONTRAINTES DANS LE TUBE (Acier):")
        print(f"  {'Point':<12} {'σ_total (MPa)':>25}")
        print("  " + "─" * 40)
        
        for nom, data in resultats['contraintes_tube'].items():
            print(f"  {nom:<12} {data['sigma_total']/1e6:>25.2f}")
        
        print(f"\n  CONTRAINTES DANS L'ARMATURE:")
        print(f"  {'Point':<12} {'σ_total (MPa)':>25}")
        print("  " + "─" * 40)
        
        for nom, data in resultats['contraintes_armature'].items():
            print(f"  {nom:<12} {data['sigma_total']/1e6:>25.2f}")
        
        print(f"\n  CONTRAINTES EXTRÊMES:")
        print(f"    Contrainte max globale:              {resultats['sigma_max']/1e6:+.2f} MPa")
        print(f"    Contrainte min globale:              {resultats['sigma_min']/1e6:+.2f} MPa")
        
        # Vérifications ELU
        verif = self.verifications_ELU(resultats)
        
        print("\n✅ VÉRIFICATIONS ÉTATS LIMITES ULTIMES (ELU) - EC4")
        print("─" * 88)
        
        print(f"\n  TUBE MÉTALLIQUE ({self.mat.classe_acier}):")
        print(f"    - Contrainte max (abs):              {verif['sigma_tube_max_MPa']:.2f} MPa")
        print(f"    - Résistance de calcul f_yd:         {verif['f_yd_tube_MPa']:.0f} MPa")
        print(f"    - Ratio (σ/f_yd):                    {verif['ratio_tube']:.4f}")
        print(f"    - Marge de sécurité:                 {(1 - verif['ratio_tube'])*100:.1f}%")
        print(f"    - Statut:                            {'✓ OK (ADMIS)' if verif['tube_OK'] else '✗ NON OK (DÉPASSEMENT)'}")
        
        print(f"\n  BÉTON ({self.mat.classe_beton}):")
        print(f"    - Contrainte max (abs):              {verif['sigma_beton_max_MPa']:.2f} MPa")
        print(f"    - Résistance de calcul fcd:          {verif['f_cd_beton_MPa']:.2f} MPa")
        print(f"    - Ratio (σ/fcd):                     {verif['ratio_beton']:.4f}")
        print(f"    - Marge de sécurité:                 {(1 - verif['ratio_beton'])*100:.1f}%")
        print(f"    - Statut:                            {'✓ OK (ADMIS)' if verif['beton_OK'] else '✗ NON OK (DÉPASSEMENT)'}")
        
        print(f"\n  ARMATURE ({self.mat.classe_armature}):")
        print(f"    - Contrainte max (abs):              {verif['sigma_armature_max_MPa']:.2f} MPa")
        print(f"    - Résistance de calcul fyd:          {verif['f_yd_armature_MPa']:.0f} MPa")
        print(f"    - Ratio (σ/fyd):                     {verif['ratio_armature']:.4f}")
        print(f"    - Marge de sécurité:                 {(1 - verif['ratio_armature'])*100:.1f}%")
        print(f"    - Statut:                            {'✓ OK (ADMIS)' if verif['armature_OK'] else '✗ NON OK (DÉPASSEMENT)'}")
        
        if verif['tous_OK']:
            print(f"\n  🎉 CONCLUSION: ✓ TOUS LES CRITÈRES SONT SATISFAITS")
        else:
            print(f"\n  ⚠️  CONCLUSION: ✗ CRITÈRES NON SATISFAITS - VÉRIFIER LES DIMENSIONNEMENTS")
        
        print("╚" + "═"*86 + "╝\n")
        
        return resultats, verif
    
    def visualiser_distribution_contraintes(self, resultats: Dict):
        """Visualisation des distributions de contraintes"""
        
        fig = plt.figure(figsize=(18, 6))
        
        # 1. Distribution circumférentielle
        ax1 = plt.subplot(131)
        
        angles = np.linspace(0, 2*np.pi, 200)
        sigma_beton_pts = []
        sigma_tube_pts = []
        
        for angle in angles:
            y = self.R_ext * np.cos(angle)
            z = self.R_ext * np.sin(angle)
            
            # Béton
            sigma_N_beton = self.N / resultats['A_eq_total']
            sigma_My_beton = -resultats['kappa_y'] * z * self.mat.E_beton
            sigma_Mz_beton = -resultats['kappa_z'] * y * self.mat.E_beton
            sigma_beton_tot = sigma_N_beton + sigma_My_beton + sigma_Mz_beton
            sigma_beton_pts.append(sigma_beton_tot / 1e6)
            
            # Tube
            sigma_tube_pts.append(resultats['n_ac'] * sigma_beton_tot / 1e6)
        
        ax1.plot(angles * 180 / np.pi, sigma_beton_pts, 'b-', linewidth=2.5, label='Béton')
        ax1.plot(angles * 180 / np.pi, sigma_tube_pts, 'r--', linewidth=2.5, label='Tube (Acier)')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.fill_between(angles * 180 / np.pi, 0, sigma_beton_pts, alpha=0.2, color='blue')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Angle (°)', fontsize=11)
        ax1.set_ylabel('Contrainte (MPa)', fontsize=11)
        ax1.set_title('Distribution Circumférentielle', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.set_xlim(0, 360)
        
        # 2. Torseur 3D
        ax2 = plt.subplot(132, projection='3d')
        
        ax2.quiver(0, 0, 0, 0, 0, 3*self.N/1e6, color='green', arrow_length_ratio=0.1, 
                   linewidth=3, label=f'N = {self.N/1000:.0f} kN')
        ax2.quiver(0, 0, 0, self.My/1e6, 0, 0, color='red', arrow_length_ratio=0.1,
                   linewidth=3, label=f'My = {self.My/1e6:.1f} kN.m')
        ax2.quiver(0, 0, 0, 0, self.Mz/1e6, 0, color='blue', arrow_length_ratio=0.1,
                   linewidth=3, label=f'Mz = {self.Mz/1e6:.1f} kN.m')
        
        ax2.set_xlabel('My (kN.m)', fontsize=10)
        ax2.set_ylabel('Mz (kN.m)', fontsize=10)
        ax2.set_zlabel('N (kN)', fontsize=10)
        ax2.set_title('Torseur de Sollicitation', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        
        # 3. Section transversale
        ax3 = plt.subplot(133)
        
        circle_ext = plt.Circle((0, 0), self.R_ext*1000, fill=False, edgecolor='black', 
                               linewidth=2, label='Tube')
        ax3.add_patch(circle_ext)
        
        circle_int = plt.Circle((0, 0), self.R_int*1000, fill=False, edgecolor='blue',
                               linewidth=1.5, linestyle='--', label='Béton')
        ax3.add_patch(circle_int)
        
        angles_barres = np.linspace(0, 2*np.pi, self.n_barres+1)
        x_barres = self.r_barres * np.cos(angles_barres[:-1]) * 1000
        y_barres = self.r_barres * np.sin(angles_barres[:-1]) * 1000
        ax3.plot(x_barres, y_barres, 'ks', markersize=8, label='Armatures')
        
        ax3.set_xlim(-self.R_ext*1200, self.R_ext*1200)
        ax3.set_ylim(-self.R_ext*1200, self.R_ext*1200)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Y (mm)', fontsize=11)
        ax3.set_ylabel('Z (mm)', fontsize=11)
        ax3.set_title('Section Transversale', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def exporter_resultats(self, nom_fichier: str = 'resultats_poteau.json'):
        """Exporte les résultats en JSON"""
        
        resultats = self.calculer_contraintes()
        verif = self.verifications_ELU(resultats)
        
        export = {
            'entrees': {
                'geometrie': {
                    'D_ext_mm': self.D_ext * 1000,
                    't_paroi_mm': self.t_paroi * 1000,
                    'd_barre_mm': self.d_barre * 1000,
                    'n_barres': self.n_barres,
                    'enrobage_mm': self.enrobage * 1000,
                },
                'materiaux': {
                    'classe_acier': self.mat.classe_acier,
                    'classe_beton': self.mat.classe_beton,
                    'classe_armature': self.mat.classe_armature,
                },
                'sollicitations': {
                    'N_kN': self.N / 1000,
                    'My_kNm': self.My / 1e6,
                    'Mz_kNm': self.Mz / 1e6,
                }
            },
            'verifications_ELU': {
                'tube': {
                    'ratio': verif['ratio_tube'],
                    'OK': verif['tube_OK'],
                },
                'beton': {
                    'ratio': verif['ratio_beton'],
                    'OK': verif['beton_OK'],
                },
                'armature': {
                    'ratio': verif['ratio_armature'],
                    'OK': verif['armature_OK'],
                },
                'tous_OK': verif['tous_OK'],
            }
        }
        
        with open(nom_fichier, 'w') as f:
            json.dump(export, f, indent=2)
        
        print(f"✓ Résultats exportés dans: {nom_fichier}")


# ============================================================================
# INTERFACE GRAPHIQUE
# ============================================================================

def calculer():
    try:
        poteau = PoteauMixteEC4(
            D_ext=float(ent_Dext.get()) / 1000,
            t_paroi=float(ent_t.get()) / 1000,
            d_barre=float(ent_barre.get()) / 1000,
            n_barres=int(ent_nb.get()),
            enrobage=float(ent_enrobage.get()) / 1000,
            
            N=float(ent_N.get()) * 1000,
            My=float(ent_My.get()) * 1000,
            Mz=float(ent_Mz.get()) * 1000,
            
            classe_acier=cb_acier.get(),
            classe_beton=cb_beton.get(),
            classe_armature=cb_armature.get()
        )
        
        buffer = io.StringIO()
        
        with redirect_stdout(buffer):
            resultats, verif = poteau.afficher_resultats()
        
        txt_resultats.delete("1.0", tk.END)
        txt_resultats.insert(tk.END, buffer.getvalue())
        
        if chk_graph.get():
            poteau.visualiser_distribution_contraintes(resultats)
    
    except Exception as e:
        messagebox.showerror("Erreur", str(e))


# Fenêtre principale
root = tk.Tk()
root.title("POTEAU MIXTE EUROCODE 4 - Flexion Biaxiale")
root.geometry("1400x900")

# Cadre saisie
frame = ttk.LabelFrame(root, text="Données d'entrée")
frame.pack(fill="x", padx=10, pady=10)

# Ligne 1
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

# Ligne 2
ttk.Label(frame, text="Nombre de barres").grid(row=1, column=0, padx=5, pady=5)
ent_nb = ttk.Entry(frame, width=15)
ent_nb.insert(0, "4")
ent_nb.grid(row=1, column=1)

ttk.Label(frame, text="Enrobage (mm)").grid(row=1, column=2)
ent_enrobage = ttk.Entry(frame, width=15)
ent_enrobage.insert(0, "40")
ent_enrobage.grid(row=1, column=3)

# Ligne 3
ttk.Label(frame, text="Effort N (kN)").grid(row=2, column=0)
ent_N = ttk.Entry(frame, width=15)
ent_N.insert(0, "1500")
ent_N.grid(row=2, column=1)

ttk.Label(frame, text="Moment My (kN.m)").grid(row=2, column=2)
ent_My = ttk.Entry(frame, width=15)
ent_My.insert(0, "50")
ent_My.grid(row=2, column=3)

ttk.Label(frame, text="Moment Mz (kN.m)").grid(row=2, column=4)
ent_Mz = ttk.Entry(frame, width=15)
ent_Mz.insert(0, "30")
ent_Mz.grid(row=2, column=5)

# Ligne 4
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

# Boutons
frame_btn = ttk.Frame(root)
frame_btn.pack(fill="x")

chk_graph = tk.BooleanVar(value=True)

ttk.Checkbutton(frame_btn, text="Afficher graphiques", variable=chk_graph).pack(side="left", padx=10)
ttk.Button(frame_btn, text="CALCULER", command=calculer).pack(side="left", padx=10)

# Zone résultats
frame_resultats = ttk.LabelFrame(root, text="Résultats détaillés Eurocode 4")
frame_resultats.pack(fill="both", expand=True, padx=10, pady=10)

scroll = tk.Scrollbar(frame_resultats)
scroll.pack(side="right", fill="y")

txt_resultats = tk.Text(frame_resultats, font=("Consolas", 10), yscrollcommand=scroll.set)
txt_resultats.pack(fill="both", expand=True)

scroll.config(command=txt_resultats.yview)

root.mainloop()
