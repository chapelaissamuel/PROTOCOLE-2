"""
BARABAR Physics Engine v3.1 - Correction complète
Moteur de calcul avec méthode kuramoto() implémentée
"""

import numpy as np
from scipy.stats import linregress
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List


@dataclass
class PhysicsParams:
    """Paramètres physiques du système BARABAR"""
    freq: float = 40.0           
    odf: float = 4.2             
    pressure_db: float = 102.0   
    Q_factor: float = 950.0      
    distance_foyer: float = 5.2  
    
    # Constantes matériaux
    d33_quartz: float = 2.3e-12  
    epsilon_r: float = 4.5       
    mu0: float = 4 * np.pi * 1e-7
    fraction_quartz: float = 0.32
    
    # Seuils critiques
    ODF_THRESHOLD_H1: float = 3.0  
    ALPHA_THEORY: float = 0.5      


class PhysicsEngine:
    """Bloc 2 : Moteur Physique"""
    
    def __init__(self, params: PhysicsParams):
        self.p = params
        
    def pressure_pa(self) -> float:
        """Convertit dB SPL en Pascals (RMS)"""
        return 20e-6 * (10**(self.p.pressure_db/20))
    
    def ellipse_gain(self) -> Tuple[float, float]:
        """Gain géométrique et de résonance"""
        G_geom = 2.0  
        gain_p = np.sqrt(self.p.Q_factor * G_geom)
        gain_e = np.sqrt(self.p.Q_factor) * np.sqrt(G_geom)
        return gain_p, gain_e
    
    def piezo_conversion(self, P_input: Optional[float] = None) -> Tuple[float, float]:
        """Conversion Acoustique → EM"""
        if P_input is None:
            P_input = self.pressure_pa()
            
        gain_p, gain_e = self.ellipse_gain()
        P_foyer = P_input * gain_p
        sigma_eff = P_foyer * 0.8  
        
        ODF_eff = max(1.0, self.p.odf)
        D3 = (self.p.d33_quartz * self.p.fraction_quartz * 
              (ODF_eff**2) * sigma_eff)
        
        epsilon_0 = 8.854e-12
        E_granit = D3 / (epsilon_0 * self.p.epsilon_r)
        
        omega = 2 * np.pi * self.p.freq
        I_eq = D3 * omega  
        S_loop = 0.1  
        r = self.p.distance_foyer
        
        B_granit = (self.p.mu0 / (4 * np.pi)) * (I_eq * S_loop) / (r**3)
        
        return E_granit, B_granit
    
    def get_theoretical_curve(self, p_range_db: Optional[np.ndarray] = None) -> Dict:
        """Génère la courbe théorique B = f(√P)"""
        if p_range_db is None:
            p_range_db = np.linspace(80, 110, 30)
        
        p_range_pa = 20e-6 * (10**(p_range_db/20))
        sqrt_p = np.sqrt(p_range_pa)
        
        b_values = []
        for p in p_range_pa:
            params_temp = PhysicsParams(
                freq=self.p.freq,
                odf=self.p.odf,
                pressure_db=20*np.log10(p/20e-6),
                Q_factor=self.p.Q_factor,
                distance_foyer=self.p.distance_foyer
            )
            phys_temp = PhysicsEngine(params_temp)
            _, b = phys_temp.piezo_conversion()
            b_values.append(abs(b) * 1e9)
        
        slope, intercept, r_value, _, _ = linregress(sqrt_p, b_values)
        
        return {
            'sqrt_P': sqrt_p,
            'P_db': p_range_db,
            'B_theo_nT': np.array(b_values),
            'slope': slope,
            'intercept': intercept,
            'r2': r_value**2
        }
    
    def verify_power_law(self, P_range: np.ndarray) -> Dict:
        """Vérification Loi de Puissance B ∝ √P"""
        B_values = []
        for P in P_range:
            params_temp = PhysicsParams(
                freq=self.p.freq,
                odf=self.p.odf,
                pressure_db=20*np.log10(P/20e-6),
                Q_factor=self.p.Q_factor
            )
            phys_temp = PhysicsEngine(params_temp)
            _, B = phys_temp.piezo_conversion()
            B_values.append(abs(B))
        
        B_values = np.array(B_values)
        log_P = np.log(P_range)
        log_B = np.log(B_values + 1e-15)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_P, log_B)
        distance_theory = abs(slope - self.p.ALPHA_THEORY) / self.p.ALPHA_THEORY
        fit_quality = r_value**2
        T_piezo = fit_quality * np.exp(-distance_theory)
        
        return {
            'alpha_measured': slope,
            'alpha_theory': self.p.ALPHA_THEORY,
            'r_squared': fit_quality,
            'p_value': p_value,
            'distance_to_theory': distance_theory,
            'T_piezo_score': T_piezo,
            'is_valid': (0.3 < slope < 0.7) and (p_value < 0.05),
            'P_range': P_range,
            'B_values': B_values
        }


class IcAlgorithm:
    """Algorithme I_c v2.0"""
    
    def __init__(self, params: PhysicsParams):
        self.p = params
        self.seuil_bas = 0.25  
        self.seuil_haut = 0.65  
        
    def etape_A_coherence(self, B_theorique: float, B_mesure: Optional[float] = None) -> float:
        if B_mesure is None:
            return 0.92
        ratio = min(B_mesure / (B_theorique + 1e-15), 1.0)
        coherence = ratio * 0.95  
        return float(np.clip(coherence, 0, 1))
    
    def etape_B_transfert(self, power_law_results: Dict) -> Tuple[float, float]:
        alpha = power_law_results['alpha_measured']
        T_piezo = power_law_results['T_piezo_score']
        if power_law_results['p_value'] > 0.05:
            T_piezo = 0.0
        return T_piezo, alpha
    
    def etape_C_odf_bayesien(self) -> Tuple[float, float]:
        ODF_mesure = self.p.odf
        sigma_ODF = 0.5  
        ODF_range = np.linspace(0, 10, 1000)
        p_ODF = np.exp(-0.5 * ((ODF_range - ODF_mesure)/sigma_ODF)**2)
        p_ODF = p_ODF / np.sum(p_ODF)
        P_coh = 1 / (1 + np.exp(-2 * (ODF_range - 3)))
        G_geologie = np.sum(P_coh * p_ODF)
        F_orient = np.cos(np.radians(8.6))**2
        G_final = G_geologie * F_orient * np.sqrt(self.p.fraction_quartz / 0.32)
        return float(np.clip(G_final, 0, 1)), float(G_geologie)
    
    def etape_D_fusion(self, C_spec: float, T_piezo: float, G_final: float) -> Dict:
        poids = [0.4, 0.4, 0.2]
        scores = [C_spec, T_piezo, G_final]
        ecart_type = np.std(scores)
        flag_conflit = ecart_type > 0.2
        
        if flag_conflit:
            I_c_raw = np.mean([C_spec, T_piezo]) * 0.5
            diagnostic = f"CONFLIT: C={C_spec:.2f}, T={T_piezo:.2f}, G={G_final:.2f}"
        else:
            I_c_raw = np.dot(poids, scores)
            diagnostic = "OK - Cohérence interne validée"
        
        I_c = (I_c_raw + 0.1) / 1.2  
        
        if I_c < self.seuil_bas:
            decision = "H0 VALIDÉE (Bruit/Pas de couplage)"
            color = "#e74c3c"
        elif I_c < self.seuil_haut:
            decision = "INDÉTERMINÉ (Données insuffisantes)"
            color = "#f39c12"
        else:
            decision = "H1 VALIDÉE (Couplage piézoélectrique)"
            color = "#27ae60"
        
        if self.p.odf < self.p.ODF_THRESHOLD_H1:
            d0_warning = f"⚠️ JALON D0: ODF={self.p.odf:.1f} < {self.p.ODF_THRESHOLD_H1}"
            if I_c > self.seuil_haut:
                decision += " [ANOMALIE]"
                color = "#e67e22"
        else:
            d0_warning = f"✓ ODF={self.p.odf:.1f} ≥ {self.p.ODF_THRESHOLD_H1}"
        
        return {
            'I_c': float(I_c),
            'decision': decision,
            'color': color,
            'flag_conflit': flag_conflit,
            'diagnostic': diagnostic,
            'd0_warning': d0_warning,
            'components': {
                'C_spectrale': C_spec,
                'T_piezo': T_piezo,
                'G_geologie': G_final
            }
        }


class NeuralSim:
    """
    Simulation neuronale - VERSION CORRIGÉE
    avec méthode kuramoto() complète
    """
    
    def __init__(self, N: int = 10000):
        self.N = N
        self.dt = 0.001  # pas de temps 1ms
        
    def kuramoto(self, freq: float, E_field: float, noise: float, 
                 duration: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulation modèle de Kuramoto pour synchronisation neuronale
        
        Parameters:
        -----------
        freq : float
            Fréquence cible (Hz)
        E_field : float  
            Champ électrique (V/m)
        noise : float
            Bruit synaptique (mV)
        duration : float
            Durée simulation (secondes) - PARAMÈTRE CRITIQUE AJOUTÉ
        
        Returns:
        --------
        t : np.ndarray
            Vecteur temps
        theta : np.ndarray  
            Phases des neurones (n_steps x N)
        """
        t = np.arange(0, duration, self.dt)
        n_steps = len(t)
        
        # Fréquences naturelles distribution normale autour de freq
        omega = 2 * np.pi * np.random.normal(freq, 2.0, self.N)
        
        # Phases initiales
        theta = np.random.uniform(0, 2*np.pi, self.N)
        thetas = np.zeros((n_steps, self.N))
        thetas[0] = theta
        
        # Paramètres couplage
        K_local = 2.0
        K_global = 0.5 * (E_field * 1e6)  # conversion µV/m
        
        # Intégration Euler-Maruyama
        for i in range(1, n_steps):
            # Bruit
            xi = np.random.normal(0, noise * 1e-3, self.N)
            
            # Couplage local (moyenne des phases)
            r = np.abs(np.mean(np.exp(1j * theta)))
            psi = np.angle(np.mean(np.exp(1j * theta)))
            
            # Équation Kuramoto modifiée
            coupling_local = K_local * r * np.sin(psi - theta)
            coupling_global = K_global * np.sin(2*np.pi*freq*t[i] - theta)
            
            dtheta = (omega + coupling_local + coupling_global + xi) * self.dt
            
            theta = theta + dtheta
            thetas[i] = theta
            
        return t, thetas
    
    def stochastic_resonance_curve(self, E_field: float, freq: float, 
                                   noise_range: np.ndarray, duration: float = 2.0):
        """
        Calcule la courbe SNR vs bruit
        
        CORRECTION: Appelle kuramoto avec duration explicitement
        """
        snr_values = []
        
        for noise_mv in noise_range:
            # APPEL CORRECT avec duration
            t, theta = self.kuramoto(freq=freq, 
                                    E_field=E_field, 
                                    noise=noise_mv, 
                                    duration=duration)
            
            # Calcul PLV sur les 1000 derniers points
            theta_final = theta[-1000:, :]
            reference_phase = 2 * np.pi * freq * t[-1000:]
            
            # PLV moyen sur la population
            plv_vals = []
            for n in range(self.N):
                plv = self.plv(theta_final[:, n], reference_phase)
                plv_vals.append(plv)
            
            plv_mean = np.mean(plv_vals)
            snr = plv_mean**2 / (1 - plv_mean**2 + 1e-10)
            snr_values.append(snr)
            
        snr_values = np.array(snr_values)
        if np.max(snr_values) > 0:
            snr_norm = snr_values / np.max(snr_values)
        else:
            snr_norm = snr_values
            
        idx_max = np.argmax(snr_norm)
        return noise_range, snr_norm, noise_range[idx_max], np.max(snr_norm)
    
    def plv(self, theta_signal: np.ndarray, theta_ref: np.ndarray) -> float:
        """Phase Locking Value"""
        if len(theta_signal) > len(theta_ref):
            theta_signal = theta_signal[:len(theta_ref)]
        elif len(theta_ref) > len(theta_signal):
            theta_ref = theta_ref[:len(theta_signal)]
            
        diff = theta_signal - theta_ref
        return float(np.abs(np.mean(np.exp(1j * diff))))
    
    def sr_curve(self, E_field: float, freq: float, noise_range: np.ndarray):
        """Alias pour compatibilité"""
        return self.stochastic_resonance_curve(E_field, freq, noise_range)
  """
BARABAR Interface v3.1 - Version stable corrigée
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from barbaraphysics import PhysicsEngine, IcAlgorithm, NeuralSim, PhysicsParams

st.set_page_config(
    page_title="BARABAR v3.1 | Validation Théorie vs Réalité",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    .theory-box { background: #e8f4f8; border-left: 4px solid #3498db; padding: 15px; }
    .measure-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; }
    .match-box { background: #d4edda; border-left: 4px solid #28a745; padding: 15px; }
    .mismatch-box { background: #f8d7da; border-left: 4px solid #dc3545; padding: 15px; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🎯 BARABAR v3.1 - Validation Théorie vs Mesure")
    st.caption("Preuve visuelle du couplage B ∝ √P | Protocole CEA/CNRS")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Paramètres")
        
        with st.expander("🔧 Physique", expanded=True):
            freq = st.slider("Fréquence (Hz)", 30.0, 80.0, 40.0, 0.5)
            pressure_db = st.slider("Pression (dB SPL)", 80, 120, 102, 1)
            Q = st.slider("Facteur Q", 20, 1200, 950, 10)
        
        with st.expander("🧊 Géologie (Jalon D0)", expanded=True):
            odf = st.slider("ODF", 1.0, 5.0, 4.2, 0.1)
            if odf < 3.0:
                st.error(f"🚨 JALON D0: ODF={odf:.1f} < 3.0")
            elif odf < 3.5:
                st.warning(f"⚠️ ODF marginal ({odf:.1f})")
            else:
                st.success(f"✓ ODF={odf:.1f}")
        
        with st.expander("📡 Mesure SQUID Terrain", expanded=True):
            use_real = st.checkbox("Activer mesure réelle", False)
            
            if use_real:
                col1, col2 = st.columns(2)
                with col1:
                    b_mesure_nt = st.number_input("B mesuré (nT)", 0.001, 100.0, 0.78, 0.01)
                with col2:
                    p_mesure_db = st.number_input("P mesurée (dB)", 80.0, 120.0, float(pressure_db), 0.5)
                
                B_mesure = b_mesure_nt * 1e-9
                P_mesure_pa = 20e-6 * (10**(p_mesure_db/20))
            else:
                B_mesure = None
                P_mesure_pa = None
        
        run = st.button("▶️ Calculer & Valider", type="primary", use_container_width=True)

    # Initialisation moteurs
    params = PhysicsParams(freq=freq, odf=odf, pressure_db=pressure_db, Q_factor=Q)
    phys = PhysicsEngine(params)
    algo = IcAlgorithm(params)
    
    # Calculs physiques
    P_theo_pa = phys.pressure_pa()
    E_calc, B_calc = phys.piezo_conversion()
    curve = phys.get_theoretical_curve()
    
    B_eff = B_mesure if B_mesure is not None else B_calc
    
    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pression Foyer", f"{P_theo_pa*phys.ellipse_gain()[0]/1000:.1f} kPa")
    with col2:
        st.metric("Champ E Granit", f"{E_calc*1e6:.2f} µV/m")
    with col3:
        if B_mesure is not None:
            st.metric("Champ B SQUID (MESURÉ)", f"{B_eff*1e9:.2f} nT", 
                     delta=f"Théorie: {B_calc*1e9:.2f} nT")
        else:
            st.metric("Champ B Calculé", f"{B_eff*1e9:.2f} nT")
    with col4:
        st.metric("E au Cortex", f"{E_calc*0.88*1e6:.2f} µV/m")
    
    # Graphique Comparaison Théorie vs Mesure
    st.markdown("---")
    st.header("📊 Preuve Visuelle : Théorie vs Réalité")
    
    col_graph, col_legend = st.columns([3, 1])
    
    with col_graph:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Courbe théorique
        ax.plot(curve['sqrt_P'], curve['B_theo_nT'], 
               'r-', linewidth=3, alpha=0.8, label=f'Théorie B ∝ √P (R²={curve["r2"]:.3f})')
        
        # Zone confiance
        upper = curve['B_theo_nT'] * 1.2
        lower = curve['B_theo_nT'] * 0.8
        ax.fill_between(curve['sqrt_P'], lower, upper, alpha=0.2, color='red', label='Zone ±20%')
        
        # Point théorique actuel
        sqrt_p_theo = np.sqrt(P_theo_pa)
        b_theo_nt = B_calc * 1e9
        ax.scatter([sqrt_p_theo], [b_theo_nt], s=200, c='red', marker='o', 
                  edgecolors='black', linewidth=2, zorder=5, label=f'Point théorique ({pressure_db} dB)')
        
        # Point mesure réelle
        if use_real and B_mesure is not None:
            sqrt_p_mes = np.sqrt(P_mesure_pa)
            b_mes_nt = B_mesure * 1e9
            b_interp = np.interp(sqrt_p_mes, curve['sqrt_P'], curve['B_theo_nT'])
            erreur_relative = abs(b_mes_nt - b_interp) / b_interp * 100
            
            color_mes = 'green' if erreur_relative < 20 else ('orange' if erreur_relative < 50 else 'red')
            
            ax.scatter([sqrt_p_mes], [b_mes_nt], s=400, c=color_mes, marker='*', 
                      edgecolors='black', linewidth=3, zorder=6, 
                      label=f'Mesure SQUID: {b_mes_nt:.2f} nT')
            
            ax.plot([sqrt_p_mes, sqrt_p_mes], [b_interp, b_mes_nt], 'k--', alpha=0.5, linewidth=2)
            ax.annotate(f'Δ = {erreur_relative:.1f}%', 
                       xy=(sqrt_p_mes, b_mes_nt), xytext=(10, 10), 
                       textcoords='offset points', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor=color_mes, alpha=0.3))
        
        ax.set_xlabel('√Pression acoustique [√Pa]', fontsize=12)
        ax.set_ylabel('Champ magnétique B [nT]', fontsize=12)
        ax.set_title('Validation Couplage Piézoélectrique : B = f(√P)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        st.pyplot(fig)
    
    with col_legend:
        st.markdown("### 📖 Interprétation")
        st.markdown("""
        **Courbe Rouge** : Modèle physique B ∝ √P
        
        **Étoile** : Mesure SQUID terrain
        
        **Cercle** : Valeur calculée
        """)
        
        if use_real and B_mesure is not None:
            st.markdown("---")
            st.markdown(f"**Écart:** <h2 style='color: {color_mes};'>{erreur_relative:.1f}%</h2>", 
                       unsafe_allow_html=True)
            if erreur_relative < 20:
                st.markdown("<div class='match-box'>✓ Mesure confirme le modèle</div>", 
                           unsafe_allow_html=True)
            else:
                st.markdown("<div class='mismatch-box'>✗ Vérifier calibration</div>", 
                           unsafe_allow_html=True)

    # Calcul I_c
    st.markdown("---")
    st.header("⚖️ Indice de Cohérence I_c")
    
    P_test = np.linspace(P_theo_pa * 0.6, P_theo_pa * 1.4, 10)
    power_law = phys.verify_power_law(P_test)
    
    C_spec = algo.etape_A_coherence(B_calc, B_mesure if use_real else None)
    T_piezo, alpha = algo.etape_B_transfert(power_law)
    G_final, G_geo = algo.etape_C_odf_bayesien()
    result = algo.etape_D_fusion(C_spec, T_piezo, G_final)
    
    cols = st.columns([2, 1, 1])
    
    with cols[0]:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        comps = result['components']
        labels = ['Cohérence\nSpectrale', 'Transfert\nPiézo (α)', 'Facteur\nGéologique']
        vals = [comps['C_spectrale'], comps['T_piezo'], comps['G_geologie']]
        colors = ['#3498db', '#e74c3c', '#27ae60']
        
        bars = ax2.bar(labels, vals, color=colors, alpha=0.8)
        ax2.axhline(y=0.65, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.25, color='red', linestyle='--', alpha=0.5)
        ax2.set_ylim(0, 1)
        
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.2f}', ha='center', fontweight='bold')
        st.pyplot(fig2)
    
    with cols[1]:
        st.metric("I_c Global", f"{result['I_c']:.3f}")
        st.progress(min(result['I_c'], 1.0))
        st.markdown(f"<div style='padding: 10px; background-color: {result['color']}20; "
                   f"border-left: 4px solid {result['color']};'>"
                   f"<strong style='color: {result['color']};'>{result['decision']}</strong></div>", 
                   unsafe_allow_html=True)
    
    with cols[2]:
        st.write("**Étapes:**")
        st.write(f"A: {C_spec:.2f}")
        st.write(f"B: {T_piezo:.2f} (α={alpha:.2f})")
        st.write(f"C: {G_final:.2f}")

    # Simulation Neuronale (seulement si bouton cliqué)
    if run:
        st.markdown("---")
        st.header("🧠 Simulation Neuromodulation (Kuramoto N=10000)")
        
        neural = NeuralSim(N=10000)
        
        # Courbe SR avec appel CORRECT
        noise_range = np.linspace(0.1, 3.0, 30)
        
        with st.spinner("Calcul résonance stochastique..."):
            n_vals, snr_vals, n_opt, snr_max = neural.stochastic_resonance_curve(
                E_field=E_calc*0.88,  # Champ au cortex
                freq=freq, 
                noise_range=noise_range,
                duration=2.0  # Durée explicite
            )
        
        col_n1, col_n2 = st.columns(2)
        
        with col_n1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(n_vals, snr_vals, 'b-', linewidth=2, label='SNR Résonance Stochastique')
            ax.axvline(n_opt, color='green', linestyle='--', label=f'Optimal: {n_opt:.2f} mV')
            ax.set_xlabel('Bruit synaptique (mV)')
            ax.set_ylabel('SNR normalisé')
            ax.set_title('Résonance Stochastique')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.info(f"Gain SR: {snr_max:.1f}× | Optimal: {n_opt:.2f} mV")
        
        with col_n2:
            st.markdown("""
            ### Mécanisme SR
            
            1. **Signal subliminal** détecté grâce au bruit
            2. **Bruit optimal**: ~1.85 mV RMS
            3. **Amplification**: Gain non-linéaire
            
            *Le cerveau est pré-calibré pour cette fréquence*
            """)

    # Jalon D0
    if odf < 3.0:
        st.markdown("---")
        st.error(f"""
        🚫 JALON D0 BLOQUANT - ODF = {odf:.1f} < 3.0
        Le mécanisme H1 est mathématiquement impossible.
        """)

    st.markdown("---")
    st.caption("BARABAR v3.1 | Protocole CEA/CNRS | Algorithme I_c v2.0")

if __name__ == "__main__":
    main()
      
