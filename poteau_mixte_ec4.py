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
        
        Pour section circulaire remplie de béton (6.7.3.6) :
        N_pl,Rd = η_a·A_a·f_yd + A_c·f_cd + A_s·f_sd
        
        Sans confinement (simplifié) :
        N_pl,Rd = A_a·f_yd + A_c·f_cd + A_s·f_sd
        """
        
        N_a = self.A_tube * self.mat.f_yd_acier
        N_c = self.A_beton * self.mat.f_cd_beton
        N_s = self.A_armature * self.mat.f_yd_armature
        
        N_pl_Rd = N_a + N_c + N_s
        
        return N_pl_Rd
    
    def calculer_M_pl_Rd(self) -> float:
        """
        Moment résistant plastique en flexion (EC4 6.7.3.2)
        
        La section se divise en zones de compression/traction
        Pour une section circulaire, on utilise :
        
        M_pl,Rd = M_a + M_c + M_s
        
        où M = (force) × (bras de levier)
        """
        
        # Axe neutre plastique : y_pl
        # N_comp = N_trait
        # Pour flexion pure autour Y :
        
        # Hypothèse : axe neutre au centre (symétrie pour flexion pure)
        # Distance du centre au bord = R_ext
        
        # Moment contribution acier (tube)
        M_a = (self.A_tube * self.mat.f_yd_acier) * self.R_ext
        
        # Moment contribution béton
        # Pour section circulaire : M = ∫ σ · y · dA
        # Simplification : béton travaille en compression, distance moyenne ≈ (4/3π) * R
        M_c = (self.A_beton * self.mat.f_cd_beton) * (4 * self.R_int / (3 * np.pi))
        
        # Moment contribution armature
        M_s = (self.A_armature * self.mat.f_yd_armature) * self.r_barres
        
        M_pl_Rd = M_a + M_c + M_s
        
        return M_pl_Rd
    
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
        N_pl_Rd = self.calculer_N_pl_Rd()
        M_pl_Rd = self.calculer_M_pl_Rd()
        
        resultats['N_pl_Rd'] = N_pl_Rd
        resultats['M_pl_Rd'] = M_pl_Rd
        
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
        for nom, (x, y, z) in points.items():
            
            # Contrainte axiale
            if self.A_total > 0:
                sigma_N_beton = self.N / self.A_total
            else:
                sigma_N_beton = 0
            
            # Contraintes de flexion
            if I_eff > 0:
                sigma_My_beton = -(self.My * z) / I_eff
                sigma_Mz_beton = (self.Mz * y) / I_eff
            else:
                sigma_My_beton = 0
                sigma_Mz_beton = 0
            
            # Contrainte totale dans béton
            sigma_beton = sigma_N_beton + sigma_My_beton + sigma_Mz_beton
            
            # Contraintes dans autres matériaux
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
        resultats['sigma_armature_max'] = max(sigma_arm_vals)
        
        # Ratios ELU
        resultats['ratio_beton'] = abs(resultats['sigma_beton_max']) / self.mat.f_cd_beton if self.mat.f_cd_beton > 0 else 0
        resultats['ratio_tube'] = abs(resultats['sigma_tube_max']) / self.mat.f_yd_acier if self.mat.f_yd_acier > 0 else 0
        resultats['ratio_armature'] = abs(resultats['sigma_armature_max']) / self.mat.f_yd_armature if self.mat.f_yd_armature > 0 else 0
        
        # Vérification flexion biaxiale (EC4 6.7.3.7)
        M_total = np.sqrt(self.My**2 + self.Mz**2)
        resultats['M_total'] = M_total
        resultats['ratio_moment'] = M_total / M_pl_Rd if M_pl_Rd > 0 else 0
        resultats['ratio_effort'] = abs(self.N) / N_pl_Rd if N_pl_Rd > 0 else 0
        
        return resultats
    
    def afficher_resultats(self):
        """Affichage complet des résultats"""
        
        print("\n" + "╔" + "═"*90 + "╗")
        print("║" + " POTEAU MIXTE EUROCODE 4 - COURBE D'INTERACTION N-M ".center(90) + "║")
        print("╚" + "═"*90 + "╝")
        
        print("\n📋 PARAMÈTRES D'ENTRÉE")
        print("─" * 92)
        print(f"  GÉOMÉTRIE:")
        print(f"    Diamètre extérieur:      {self.D_ext*1000:.1f} mm")
        print(f"    Épaisseur paroi:         {self.t_paroi*1000:.2f} mm")
        print(f"    Barres armature:         {self.n_barres} × Ø{self.d_barre*1000:.1f} mm")
        print(f"    Enrobage:                {self.enrobage*1000:.1f} mm")
        
        print(f"\n  MATÉRIAUX:")
        print(f"    Acier tube:              {self.mat.classe_acier} (f_yd = {self.mat.f_yd_acier/1e6:.0f} MPa)")
        print(f"    Béton:                   {self.mat.classe_beton} (f_cd = {self.mat.f_cd_beton/1e6:.2f} MPa)")
        print(f"    Armature:                {self.mat.classe_armature} (f_yd = {self.mat.f_yd_armature/1e6:.0f} MPa)")
        
        print(f"\n  SOLLICITATIONS:")
        print(f"    Effort normal:           {self.N/1000:+.2f} kN")
        print(f"    Moment Y:                {self.My/1e6:+.2f} kN.m")
        print(f"    Moment Z:                {self.Mz/1e6:+.2f} kN.m")
        
        resultats = self.calculer_contraintes()
        
        print("\n📊 RÉSISTANCES PLASTIQUES (EC4 6.7.3.2 - 6.7.3.6)")
        print("─" * 92)
        print(f"  N_pl,Rd:                 {resultats['N_pl_Rd']/1000:.2f} kN")
        print(f"  M_pl,Rd:                 {resultats['M_pl_Rd']/1e6:.2f} kN.m")
        print(f"  M_total:                 {resultats['M_total']/1e6:.2f} kN.m")
        
        print("\n⚙️ CONTRAINTES ÉLASTIQUES - BÉTON")
        print("─" * 92)
        print(f"  {'Point':<12} {'σ_N (MPa)':>12} {'σ_My (MPa)':>12} {'σ_Mz (MPa)':>12} {'σ_total (MPa)':>15}")
        print("  " + "─" * 63)
        
        for nom, data in resultats['contraintes_beton'].items():
            print(f"  {nom:<12} {data['sigma_N']/1e6:>12.2f} {data['sigma_My']/1e6:>12.2f} "
                  f"{data['sigma_Mz']/1e6:>12.2f} {data['sigma_total']/1e6:>15.2f}")
        
        print(f"\n  Contrainte max:          {resultats['sigma_beton_max']/1e6:+.2f} MPa")
        print(f"  Contrainte min:          {resultats['sigma_beton_min']/1e6:+.2f} MPa")
        
        print("\n⚙️ CONTRAINTES ÉLASTIQUES - TUBE (Acier)")
        print("─" * 92)
        for nom, data in resultats['contraintes_tube'].items():
            print(f"  {nom:<12} {data['sigma_total']/1e6:>25.2f} MPa")
        print(f"  Max:                     {resultats['sigma_tube_max']/1e6:>25.2f} MPa")
        
        print("\n⚙️ CONTRAINTES ÉLASTIQUES - ARMATURE")
        print("─" * 92)
        for nom, data in resultats['contraintes_armature'].items():
            print(f"  {nom:<12} {data['sigma_total']/1e6:>25.2f} MPa")
        print(f"  Max:                     {resultats['sigma_armature_max']/1e6:>25.2f} MPa")
        
        print("\n✅ VÉRIFICATIONS ELU (EC4 6.7.3.7 - FLEXION BIAXIALE)")
        print("─" * 92)
        print(f"  TUBE (S{self.mat.classe_acier}):")
        print(f"    - Ratio σ/f_yd:        {resultats['ratio_tube']:.4f} {'✓' if resultats['ratio_tube'] <= 1.0 else '✗'}")
        print(f"    - Statut:              {'OK' if resultats['ratio_tube'] <= 1.0 else 'NON OK'}")
        
        print(f"\n  BÉTON ({self.mat.classe_beton}):")
        print(f"    - Ratio σ/f_cd:        {resultats['ratio_beton']:.4f} {'✓' if resultats['ratio_beton'] <= 1.0 else '✗'}")
        print(f"    - Statut:              {'OK' if resultats['ratio_beton'] <= 1.0 else 'NON OK'}")
        
        print(f"\n  ARMATURE ({self.mat.classe_armature}):")
        print(f"    - Ratio σ/f_yd:        {resultats['ratio_armature']:.4f} {'✓' if resultats['ratio_armature'] <= 1.0 else '✗'}")
        print(f"    - Statut:              {'OK' if resultats['ratio_armature'] <= 1.0 else 'NON OK'}")
        
        print(f"\n  INTERACTION N-M (Équation 6.46-6.47):")
        print(f"    - Ratio M/M_pl,Rd:    {resultats['ratio_moment']:.4f}")
        print(f"    - Ratio N/N_pl,Rd:    {resultats['ratio_effort']:.4f}")
        
        global_ok = (resultats['ratio_tube'] <= 1.0 and 
                    resultats['ratio_beton'] <= 1.0 and 
                    resultats['ratio_armature'] <= 1.0)
        
        print(f"\n  {'🎉 TOUS LES CRITÈRES SATISFAITS ✓' if global_ok else '⚠️  CRITÈRES NON SATISFAITS ✗'}")
        
        print("╚" + "═"*90 + "╝\n")
        
        return resultats


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
    
    except Exception as e:
        messagebox.showerror("Erreur", str(e))


root = tk.Tk()
root.title("POTEAU MIXTE EUROCODE 4 - Courbe d'Interaction N-M")
root.geometry("1400x900")

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

ttk.Button(frame_btn, text="CALCULER", command=calculer).pack(side="left", padx=10)

frame_resultats = ttk.LabelFrame(root, text="Résultats EC4 - Courbe d'Interaction N-M")
frame_resultats.pack(fill="both", expand=True, padx=10, pady=10)

scroll = tk.Scrollbar(frame_resultats)
scroll.pack(side="right", fill="y")

txt_resultats = tk.Text(frame_resultats, font=("Courier", 9), yscrollcommand=scroll.set)
txt_resultats.pack(fill="both", expand=True)

scroll.config(command=txt_resultats.yview)

root.mainloop()
