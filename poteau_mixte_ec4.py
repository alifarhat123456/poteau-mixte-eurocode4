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
# CLASSE PRINCIPALE: POTEAU MIXTE AVEC FLEXION BIAXIALE
# ============================================================================

class PoteauMixteEC4:
    """
    Calcul des contraintes dans un poteau mixte circulaire (tube enrobé)
    avec flexion biaxiale selon l'Eurocode 4
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
        Initialisation du poteau mixte avec flexion biaxiale
        
        Args:
            D_ext: Diamètre extérieur du tube (m)
            t_paroi: Épaisseur de la paroi (m)
            d_barre: Diamètre des barres d'armature (m)
            n_barres: Nombre de barres uniformément réparties
            enrobage: Enrobage du béton (m)
            N: Effort normal en compression (N, positif = compression)
            My: Moment fléchissant autour Y (N.m)
            Mz: Moment fléchissant autour Z (N.m)
            classe_acier: Classe d'acier (S235, S275, S355, S420, S460)
            classe_beton: Classe de béton (C20-C50)
            classe_armature: Classe d'armature (FeE400, FeE500)
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
        self.I_tube = np.pi * (self.R_ext**4 - self.R_int**4) / 4  # inertie
        
        # BÉTON INTÉRIEUR
        self.A_beton = np.pi * self.R_int**2
        self.I_beton = np.pi * self.R_int**4 / 4
        
        # ARMATURE (barres uniformément réparties sur le périmètre)
        # Rayon des barres (distance du centre aux barres)
        self.r_barres = self.R_int - self.enrobage - self.d_barre / 2
        
        # Section d'une barre
        self.A_barre = np.pi * (self.d_barre / 2)**2
        
        # Aire totale d'armature
        self.A_armature = self.n_barres * self.A_barre
        
        # Inertie de l'armature (barres uniformément réparties)
        self.I_armature = self.n_barres * self.A_barre * (self.r_barres**2)
        
        # SECTION TOTALE ET INERTIE
        self.A_total = self.A_tube + self.A_beton + self.A_armature
        self.I_total = self.I_tube + self.I_beton + self.I_armature
        
        # Pour section circulaire: I_y = I_z = I_total
        self.Iy = self.I_total
        self.Iz = self.I_total
        
    def calculer_contraintes(self) -> Dict:
        """
        Calcul des contraintes avec flexion biaxiale (MY et MZ)
        selon Eurocode 4
        
        Méthode: 
        - Hypothèses: sections planes, domaine élastique
        - Flexion biaxiale: superposition des moments autour Y et Z
        - Prise en compte des modules d'élasticité différents (sections transformées)
        
        Returns:
            Dict contenant les contraintes et résultats détaillés
        """
        
        resultats = {}
        
        # 1. COEFFICIENT D'ÉQUIVALENCE (section transformée)
        n_ac = self.mat.E_acier / self.mat.E_beton  
        n_as = self.mat.E_armature / self.mat.E_beton  
        
        # Aire équivalente (rapportée au béton)
        A_eq_tube = self.A_tube * n_ac
        A_eq_armature = self.A_armature * n_as
        A_eq_total = self.A_beton + A_eq_tube + A_eq_armature
        
        # Inertie équivalente (pour section circulaire: Iy = Iz)
        I_eq = (self.I_beton + 
                self.I_tube * n_ac + 
                self.I_armature * n_as)
        
        resultats['n_ac'] = n_ac
        resultats['n_as'] = n_as
        resultats['A_eq_total'] = A_eq_total
        resultats['I_eq'] = I_eq
        
        # 2. DÉFORMATION AXIALE (effort normal seul)
        eps_0 = self.N / (A_eq_total * self.mat.E_beton) if A_eq_total > 0 else 0
        
        # 3. POSITIONS DES POINTS D'OBSERVATION
        # On observe les contraintes à 4 points sur la section transversale:
        # - Top (y=+R, z=0)
        # - Bottom (y=-R, z=0)
        # - Right (y=0, z=+R)
        # - Left (y=0, z=-R)
        
        points_obs = {
            'Top': (0, self.R_ext, 0),      # (x, y, z)
            'Bottom': (0, -self.R_ext, 0),
            'Right': (0, 0, self.R_ext),
            'Left': (0, 0, -self.R_ext),
        }
        
        # 4. CONTRAINTES COMBINÉES (N + My + Mz)
        contraintes = {}
        
        for nom, (x, y, z) in points_obs.items():
            # Contrainte axiale
            sigma_N = (self.N / A_eq_total) if A_eq_total > 0 else 0
            
            # Contraintes de flexion biaxiale
            sigma_My = -(self.My * z / I_eq) if I_eq > 0 else 0  # négatif car convention
            sigma_Mz = (self.Mz * y / I_eq) if I_eq > 0 else 0
            
            # Contrainte totale (superposition)
            sigma_total = sigma_N + sigma_My + sigma_Mz
            
            contraintes[nom] = {
                'sigma_N': sigma_N,
                'sigma_My': sigma_My,
                'sigma_Mz': sigma_Mz,
                'sigma_total': sigma_total,
                'y': y,
                'z': z
            }
        
        # 5. CONTRAINTES MAX/MIN
        sigma_max = max([c['sigma_total'] for c in contraintes.values()])
        sigma_min = min([c['sigma_total'] for c in contraintes.values()])
        
        resultats['deformation_axiale'] = eps_0
        resultats['contraintes_points'] = contraintes
        resultats['sigma_max'] = sigma_max
        resultats['sigma_min'] = sigma_min
        
        # Ratios pour chaque matériau (utiliser sigma_max)
        # TUBE
        f_yd_tube = self.mat.fy_acier / self.mat.gamma_ma
        resultats['sigma_tube_max'] = sigma_max
        resultats['f_yd_tube'] = f_yd_tube
        resultats['ratio_tube'] = abs(sigma_max) / f_yd_tube if f_yd_tube > 0 else 0
        
        # BÉTON (compression seulement)
        sigma_compression = -abs(sigma_min)  # négatif = compression
        f_cd = self.mat.fcd_beton
        resultats['sigma_beton_compression'] = sigma_compression
        resultats['f_cd_beton'] = f_cd
        resultats['ratio_beton'] = abs(sigma_compression) / f_cd if f_cd > 0 else 0
        
        # ARMATURE
        resultats['sigma_armature_max'] = sigma_max
        f_yd_arm = self.mat.fyd_armature
        resultats['f_yd_armature'] = f_yd_arm
        resultats['ratio_armature'] = abs(sigma_max) / f_yd_arm if f_yd_arm > 0 else 0
        
        return resultats
    
    def verifications_ELU(self, resultats: Dict) -> Dict:
        """
        Vérifications des états limites ultimes (ELU) biaxiales
        
        Args:
            resultats: Dictionnaire des contraintes
            
        Returns:
            Dict avec les vérifications
        """
        
        verif = {}
        
        # TUBE MÉTALLIQUE
        sigma_tube_max = abs(resultats['sigma_tube_max'])
        f_yd_tube = resultats['f_yd_tube']
        ratio_tube = resultats['ratio_tube']
        verif['tube_OK'] = ratio_tube <= 1.0
        verif['ratio_tube'] = ratio_tube
        verif['sigma_tube_max_MPa'] = sigma_tube_max / 1e6
        verif['f_yd_tube_MPa'] = f_yd_tube / 1e6
        
        # BÉTON (compression)
        sigma_beton = abs(resultats['sigma_beton_compression'])
        f_cd = resultats['f_cd_beton']
        ratio_beton = resultats['ratio_beton']
        verif['beton_OK'] = ratio_beton <= 1.0
        verif['ratio_beton'] = ratio_beton
        verif['sigma_beton_max_MPa'] = sigma_beton / 1e6
        verif['f_cd_beton_MPa'] = f_cd / 1e6
        
        # ARMATURE
        sigma_armature = abs(resultats['sigma_armature_max'])
        f_yd_arm = resultats['f_yd_armature']
        ratio_armature = resultats['ratio_armature']
        verif['armature_OK'] = ratio_armature <= 1.0
        verif['ratio_armature'] = ratio_armature
        verif['sigma_armature_max_MPa'] = sigma_armature / 1e6
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
        
        print("\n⚙️ CONTRAINTES CALCULÉES (DOMAINE ÉLASTIQUE)")
        print("─" * 88)
        print(f"  Déformation axiale (ε):                {resultats['deformation_axiale']*1e6:.2f} µm/m")
        print(f"  Coefficient d'équivalence (n_ac):      {resultats['n_ac']:.2f}")
        print(f"  Coefficient d'équivalence (n_as):      {resultats['n_as']:.2f}")
        
        print(f"\n  CONTRAINTES AUX POINTS CRITIQUES:")
        print(f"  {'Point':<12} {'σ_N (MPa)':>12} {'σ_My (MPa)':>12} {'σ_Mz (MPa)':>12} {'σ_total (MPa)':>15}")
        print("  " + "─" * 63)
        
        for nom, data in resultats['contraintes_points'].items():
            print(f"  {nom:<12} {data['sigma_N']/1e6:>12.2f} {data['sigma_My']/1e6:>12.2f} "
                  f"{data['sigma_Mz']/1e6:>12.2f} {data['sigma_total']/1e6:>15.2f}")
        
        print(f"\n  CONTRAINTES EXTRÊMES:")
        print(f"    Contrainte max (traction):           {resultats['sigma_max']/1e6:+.2f} MPa")
        print(f"    Contrainte min (compression):        {resultats['sigma_min']/1e6:+.2f} MPa")
        
        # Vérifications ELU
        verif = self.verifications_ELU(resultats)
        
        print("\n✅ VÉRIFICATIONS ÉTATS LIMITES ULTIMES (ELU)")
        print("─" * 88)
        
        print(f"\n  TUBE MÉTALLIQUE ({self.mat.classe_acier}):")
        print(f"    - Contrainte max (abs):              {verif['sigma_tube_max_MPa']:.2f} MPa")
        print(f"    - Résistance de calcul f_yd:         {verif['f_yd_tube_MPa']:.0f} MPa")
        print(f"    - Ratio (σ/f_yd):                    {verif['ratio_tube']:.4f}")
        print(f"    - Marge de sécurité:                 {(1 - verif['ratio_tube'])*100:.1f}%")
        print(f"    - Statut:                            {'✓ OK (ADMIS)' if verif['tube_OK'] else '✗ NON OK (DÉPASSEMENT)'}")
        
        print(f"\n  BÉTON ({self.mat.classe_beton}):")
        print(f"    - Contrainte compression (abs):      {verif['sigma_beton_max_MPa']:.2f} MPa")
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
        
        # 1. Diagramme de contraintes radiales (coupe transversale)
        ax1 = plt.subplot(131)
        
        # Points circulaires autour de la section
        angles = np.linspace(0, 2*np.pi, 200)
        
        # Contraintes à différents rayons
        sigma_points = []
        for angle in angles:
            y = self.R_ext * np.cos(angle)
            z = self.R_ext * np.sin(angle)
            
            sigma_N = self.N / resultats['A_eq_total'] if resultats['A_eq_total'] > 0 else 0
            sigma_My = -(self.My * z / resultats['I_eq']) if resultats['I_eq'] > 0 else 0
            sigma_Mz = (self.Mz * y / resultats['I_eq']) if resultats['I_eq'] > 0 else 0
            
            sigma_total = sigma_N + sigma_My + sigma_Mz
            sigma_points.append(sigma_total / 1e6)
        
        ax1.plot(angles * 180 / np.pi, sigma_points, 'b-', linewidth=2.5, label='σ_total')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.fill_between(angles * 180 / np.pi, 0, sigma_points, alpha=0.3, where=(np.array(sigma_points) >= 0), 
                         color='red', label='Traction')
        ax1.fill_between(angles * 180 / np.pi, 0, sigma_points, alpha=0.3, where=(np.array(sigma_points) < 0), 
                         color='blue', label='Compression')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Angle (°)', fontsize=11)
        ax1.set_ylabel('Contrainte (MPa)', fontsize=11)
        ax1.set_title('Distribution Circumférentielle des Contraintes', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.set_xlim(0, 360)
        
        # 2. Diagramme vectoriel des moments
        ax2 = plt.subplot(132, projection='3d')
        
        # Vecteur N
        ax2.quiver(0, 0, 0, 0, 0, 3*self.N/1e6, color='green', arrow_length_ratio=0.1, 
                   linewidth=3, label=f'N = {self.N/1000:.0f} kN')
        
        # Vecteur My
        ax2.quiver(0, 0, 0, self.My/1e6, 0, 0, color='red', arrow_length_ratio=0.1,
                   linewidth=3, label=f'My = {self.My/1e6:.1f} kN.m')
        
        # Vecteur Mz
        ax2.quiver(0, 0, 0, 0, self.Mz/1e6, 0, color='blue', arrow_length_ratio=0.1,
                   linewidth=3, label=f'Mz = {self.Mz/1e6:.1f} kN.m')
        
        ax2.set_xlabel('My (kN.m)', fontsize=10)
        ax2.set_ylabel('Mz (kN.m)', fontsize=10)
        ax2.set_zlabel('N (kN)', fontsize=10)
        ax2.set_title('Torseur de Sollicitation', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10, loc='upper right')
        
        # 3. Section transversale avec contraintes
        ax3 = plt.subplot(133)
        
        # Positions des points observés
        points_obs = {
            'Top': (0, self.R_ext, 'red'),
            'Bottom': (0, -self.R_ext, 'blue'),
            'Right': (self.R_ext, 0, 'green'),
            'Left': (-self.R_ext, 0, 'orange'),
        }
        
        # Cercle externe (tube)
        circle_ext = plt.Circle((0, 0), self.R_ext*1000, fill=False, edgecolor='black', 
                               linewidth=2, label='Tube')
        ax3.add_patch(circle_ext)
        
        # Cercle interne (béton)
        circle_int = plt.Circle((0, 0), self.R_int*1000, fill=False, edgecolor='blue',
                               linewidth=1.5, linestyle='--', label='Béton')
        ax3.add_patch(circle_int)
        
        # Positions des armatures
        angles_barres = np.linspace(0, 2*np.pi, self.n_barres+1)
        x_barres = self.r_barres * np.cos(angles_barres[:-1]) * 1000
        y_barres = self.r_barres * np.sin(angles_barres[:-1]) * 1000
        ax3.plot(x_barres, y_barres, 'ks', markersize=8, label='Armatures')
        
        # Points critiques avec contraintes
        for nom, (y, z, color) in points_obs.items():
            sigma_N = self.N / resultats['A_eq_total']
            sigma_My = -(self.My * z / resultats['I_eq'])
            sigma_Mz = (self.Mz * y / resultats['I_eq'])
            sigma_tot = sigma_N + sigma_My + sigma_Mz
            
            ax3.plot(y*1000, z*1000, 'o', color=color, markersize=12)
            ax3.text(y*1000*1.15, z*1000*1.15, f'{nom}\n{sigma_tot/1e6:.1f}MPa', 
                    fontsize=9, ha='center', color=color, fontweight='bold')
        
        ax3.set_xlim(-self.R_ext*1200, self.R_ext*1200)
        ax3.set_ylim(-self.R_ext*1200, self.R_ext*1200)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Y (mm)', fontsize=11)
        ax3.set_ylabel('Z (mm)', fontsize=11)
        ax3.set_title('Section Transversale avec Points Critiques', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10, loc='upper right')
        
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
            'geometrie': {
                'A_total_mm2': self.A_total * 1e6,
                'I_total_mm4': self.I_total * 1e12,
                'r_barres_mm': self.r_barres * 1000,
            },
            'proprietes_materiaux': {
                'acier': {
                    'fy_MPa': self.mat.fy_acier / 1e6,
                    'f_yd_MPa': self.mat.fy_acier / self.mat.gamma_ma / 1e6,
                    'E_GPa': self.mat.E_acier / 1e9,
                },
                'beton': {
                    'fck_MPa': self.mat.fck_beton / 1e6,
                    'fcd_MPa': self.mat.fcd_beton / 1e6,
                    'E_GPa': self.mat.E_beton / 1e9,
                },
                'armature': {
                    'fyk_MPa': self.mat.fyk_armature / 1e6,
                    'fyd_MPa': self.mat.fyd_armature / 1e6,
                    'E_GPa': self.mat.E_armature / 1e9,
                }
            },
            'contraintes_MPa': {
                'sigma_max': resultats['sigma_max'] / 1e6,
                'sigma_min': resultats['sigma_min'] / 1e6,
                'points_critiques': {
                    nom: {k: (v/1e6 if 'sigma' in k else v) for k, v in data.items()}
                    for nom, data in resultats['contraintes_points'].items()
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
# EXEMPLES D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "╔" + "═"*86 + "╗")
    print("║" + " EXEMPLE 1: POTEAU MIXTE STANDARD ".center(86) + "║")
    print("╚" + "═"*86 + "╝")
    
    # EXEMPLE 1: Cas standard
    poteau1 = PoteauMixteEC4(
        D_ext=0.323,           # 323 mm
        t_paroi=0.010,         # 10 mm
        d_barre=0.016,         # 16 mm
        n_barres=4,            # 4 barres
        enrobage=0.040,        # 40 mm
        N=1500e3,              # 1500 kN (compression)
        My=50e6,               # 50 kN.m (flexion autour Y)
        Mz=30e6,               # 30 kN.m (flexion autour Z)
        classe_acier='S355',
        classe_beton='C25',
        classe_armature='FeE500'
    )
    
    resultats1, verif1 = poteau1.afficher_resultats()
    poteau1.visualiser_distribution_contraintes(resultats1)
    poteau1.exporter_resultats('resultats_poteau1.json')
    
    
    print("\n" + "╔" + "═"*86 + "╗")
    print("║" + " EXEMPLE 2: POTEAU MIXTE HAUTE RÉSISTANCE ".center(86) + "║")
    print("╚" + "═"*86 + "╝")
    
    # EXEMPLE 2: Acier et béton haute résistance
    poteau2 = PoteauMixteEC4(
        D_ext=0.400,           # 400 mm
        t_paroi=0.012,         # 12 mm
        d_barre=0.020,         # 20 mm
        n_barres=6,            # 6 barres
        enrobage=0.045,        # 45 mm
        N=2500e3,              # 2500 kN (compression)
        My=80e6,               # 80 kN.m
        Mz=60e6,               # 60 kN.m
        classe_acier='S420',
        classe_beton='C40',
        classe_armature='FeE500'
    )
    
    resultats2, verif2 = poteau2.afficher_resultats()
    poteau2.visualiser_distribution_contraintes(resultats2)
    poteau2.exporter_resultats('resultats_poteau2.json')
    
    
    print("\n" + "╔" + "═"*86 + "╗")
    print("║" + " EXEMPLE 3: POTEAU MIXTE GRANDE FLEXION ".center(86) + "║")
    print("╚" + "═"*86 + "╝")
    
    # EXEMPLE 3: Cas de grande flexion biaxiale
    poteau3 = PoteauMixteEC4(
        D_ext=0.350,           # 350 mm
        t_paroi=0.008,         # 8 mm
        d_barre=0.012,         # 12 mm
        n_barres=8,            # 8 barres
        enrobage=0.035,        # 35 mm
        N=800e3,               # 800 kN (compression modérée)
        My=150e6,              # 150 kN.m (forte flexion Y)
        Mz=100e6,              # 100 kN.m (forte flexion Z)
        classe_acier='S355',
        classe_beton='C30',
        classe_armature='FeE500'
    )
    
    resultats3, verif3 = poteau3.afficher_resultats()
    poteau3.visualiser_distribution_contraintes(resultats3)
    poteau3.exporter_resultats('resultats_poteau3.json')
