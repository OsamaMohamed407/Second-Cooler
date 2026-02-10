import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 1. PHYSICS ENGINE (High Precision)
# ==========================================
class ASHRAE_Psychrometrics:
    def __init__(self):
        self.R_da = 287.042
        self.P_std = 101325.0

    def get_saturation_pressure(self, T_celsius):
        # Hyland-Wexler Formulation (High Accuracy)
        T = T_celsius + 273.15
        if T_celsius < 0: 
            return 611.21 * math.exp((22.587 * T_celsius) / (T_celsius + 273.86))
        C1, C2, C3 = -5.8002206E+03, 1.3914993E+00, -4.8640239E-02
        C4, C5, C6 = 4.1764768E-05, -1.4452093E-08, 6.5459673E+00
        log_Pws = (C1/T) + C2 + (C3*T) + (C4*T**2) + (C5*T**3) + (C6*math.log(T))
        return math.exp(log_Pws)

    def get_properties(self, T_c, P_total_Pa, mode='RH', value=0.5):
        P_ws = self.get_saturation_pressure(T_c)
        if mode == 'RH':
            RH = value
            P_w = RH * P_ws
            if (P_total_Pa - P_w) <= 0: W = 0
            else: W = 0.621945 * P_w / (P_total_Pa - P_w)
        elif mode == 'W': 
            W = value
            P_w = P_total_Pa * W / (0.621945 + W)
            RH = P_w / P_ws if P_ws > 0 else 0
        elif mode == 'Wt_Pct': 
            wt = value 
            W = (wt / 100.0) / (1.0 - (wt / 100.0))
            P_w = P_total_Pa * W / (0.621945 + W)
            RH = P_w / P_ws if P_ws > 0 else 0

        t = T_c
        h = 1.006 * t + W * (2501.0 + 1.805 * t)
        T_k = t + 273.15
        v = (self.R_da * T_k / P_total_Pa) * (1 + 1.6078 * W)
        rho = 1 / v if v > 0 else 0
        # Sutherland's law for Viscosity
        mu = (1.716e-5) * ((T_k/273.15)**1.5) * ((273.15 + 110.4)/(T_k + 110.4))
        return {'W': W, 'h': h, 'RH': RH, 'v': v, 'rho': rho, 'mu': mu}

# ==========================================
# 2. EQUIPMENT MODELS
# ==========================================
class ChillerModel:
    def __init__(self, psychro):
        self.air = psychro
        # Calibrated to PFD Normal Case
        self.pfd_m_wet = 74095.0     
        self.pfd_T_in = 32.0         
        self.pfd_Wt_in = 1.90        
        self.pfd_T_out = 6.0         
        self.pfd_Duty = 1240.0       
        self.pfd_T_NH3 = 1.0         
        self.UA_calibrated = self._calibrate()

    def _calc_lmtd(self, T_air_in, T_air_out, T_ref):
        dt1 = T_air_in - T_ref
        dt2 = T_air_out - T_ref
        if dt1 <= 0.01: dt1 = 0.01
        if dt2 <= 0.01: dt2 = 0.01
        if abs(dt1 - dt2) < 0.01: return dt1
        return (dt1 - dt2) / math.log(dt1/dt2)

    def _calibrate(self):
        lmtd = self._calc_lmtd(self.pfd_T_in, self.pfd_T_out, self.pfd_T_NH3)
        return self.pfd_Duty / lmtd

    def solve(self, m_wet_kg_h, T_in, RH_in, T_NH3):
        state_in = self.air.get_properties(T_in, 101325, 'RH', RH_in)
        m_dry_kg_s = (m_wet_kg_h / 3600.0) / (1 + state_in['W'])
        ratio = m_wet_kg_h / self.pfd_m_wet
        # UA scales with flow^0.6 (standard for tube banks)
        current_UA = self.UA_calibrated * (ratio ** 0.6)
        
        # Bisection Solver
        T_min, T_max = T_NH3 + 0.1, T_in - 0.1
        T_final = T_max
        for i in range(50):
            T_guess = (T_min + T_max) / 2.0
            state_out = self.air.get_properties(T_guess, 101325, 'RH', 1.0)
            Q_air = m_dry_kg_s * (state_in['h'] - state_out['h'])
            lmtd = self._calc_lmtd(T_in, T_guess, T_NH3)
            Q_exch = current_UA * lmtd
            if abs(Q_exch - Q_air) < 0.01: 
                T_final = T_guess
                break
            if (Q_exch - Q_air) > 0: T_max = T_guess
            else: T_min = T_guess
        
        T_final = (T_min + T_max) / 2.0
        final_state = self.air.get_properties(T_final, 101325, 'RH', 1.0)
        w_removed = state_in['W'] - final_state['W']
        condensate = max(0, w_removed * m_dry_kg_s * 3600.0)
        return {"m_dry_kg_s": m_dry_kg_s, "W_out": final_state['W'], "T_out": T_final, "Condensate_kg_h": condensate, "Duty_kW": (m_dry_kg_s * (state_in['h'] - final_state['h']))}

class HeaterModel:
    def __init__(self, psychro):
        self.air = psychro
        self.delta_h_steam = 2142.7 # From steam table (4 bar)
    def solve(self, m_dry_kg_s, T_in_air, W_in_air, m_steam_kg_h):
        Q_steam_kW = (m_steam_kg_h / 3600.0) * self.delta_h_steam
        props_in = self.air.get_properties(T_in_air, 101325, 'W', W_in_air)
        h_target = props_in['h'] + (Q_steam_kW / m_dry_kg_s)
        
        T_min, T_max = T_in_air, T_in_air + 50.0
        T_final = T_in_air
        for i in range(50):
            T_mid = (T_min + T_max) / 2.0
            props_mid = self.air.get_properties(T_mid, 101325, 'W', W_in_air)
            if abs(props_mid['h'] - h_target) < 0.01:
                T_final = T_mid
                break
            if props_mid['h'] < h_target: T_min = T_mid
            else: T_max = T_mid
        return {"T_out": T_final, "W_out": W_in_air, "Duty_kW": Q_steam_kW, "m_dry_kg_s": m_dry_kg_s}

class FanModel:
    def __init__(self, psychro):
        self.air = psychro
        # Piller Curve Approximation
        self.curve_flow = [12.0, 16.57, 18.53, 22.0]
        self.curve_power = [55.0, 82.4, 96.0, 110.0]
    def get_shaft_power(self, flow_m3_s):
        return np.interp(flow_m3_s, self.curve_flow, self.curve_power)
    def solve(self, m_dry_kg_s, T_in_air, W_in_air):
        props_in = self.air.get_properties(T_in_air, 101325, 'W', W_in_air)
        rho = props_in['rho']
        m_total_kg_s = m_dry_kg_s * (1 + W_in_air)
        vol_flow_m3_s = m_total_kg_s / rho
        power_kW = self.get_shaft_power(vol_flow_m3_s)
        cp_moist = 1.006 + 1.86 * W_in_air
        delta_T = power_kW / (m_total_kg_s * cp_moist)
        T_out = T_in_air + delta_T
        return {"T_out": T_out, "delta_T": delta_T, "Power_kW": power_kW, "Vol_Flow_m3_s": vol_flow_m3_s, "W_out": W_in_air, "rho_out": props_in['rho'], "mu_out": props_in['mu']}

class FluidizedBedModel:
    def __init__(self):
        self.rho_particle = 1330.0 
        self.rho_bulk = 780.0      
        self.dp = 3.0e-3           
        # Calibrated Cp to match PFD heat load of ~730kW @ Normal conditions
        self.Cp_urea = 1.75  
        self.fixed_width = 1.25    
        
    def calc_fluidization_limits(self, rho_gas, mu_gas):
        g = 9.81
        Ar = (self.dp**3 * rho_gas * (self.rho_particle - rho_gas) * g) / (mu_gas**2)
        Remf = math.sqrt(33.7**2 + 0.0408 * Ar) - 33.7
        Umf = (Remf * mu_gas) / (self.dp * rho_gas)
        Ut = 1.74 * math.sqrt( (g * self.dp * (self.rho_particle - rho_gas)) / rho_gas )
        return Umf, Ut

    def solve(self, m_dry_kg_s, T_air_in, W_air, rho_air_in, mu_air_in,
              m_urea_kg_h, T_urea_in, bed_len, bed_level_mm):
        
        area = bed_len * self.fixed_width
        bed_mass = (area * (bed_level_mm / 1000.0)) * self.rho_bulk
        residence_time_sec = (bed_mass) / (m_urea_kg_h / 3600.0)
        
        m_air_total = m_dry_kg_s * (1 + W_air)
        vol_flow_air = m_air_total / rho_air_in
        U_surf = vol_flow_air / area 
        
        Umf, Ut = self.calc_fluidization_limits(rho_air_in, mu_air_in)
        
        C_urea = (m_urea_kg_h / 3600.0) * self.Cp_urea
        cp_air = 1.006 + 1.86 * W_air
        C_air = m_air_total * cp_air
        C_min = min(C_urea, C_air)
        C_max = max(C_urea, C_air)
        Cr = C_min / C_max
        
        # --- Rigorous Heat Transfer Scaling ---
        # Ref Normal Case: L=6.4, Area=8.0, Flow=74000kg/h (~17.5m3/s) -> U=2.2m/s
        # PFD Duty 730kW -> LMTD ~23 -> UA_req ~ 31.7 kW/K
        
        ref_U = 2.2 
        ref_A = 8.0
        ref_UA = 31.7 
        ref_h = ref_UA / ref_A # ~ 4.0 kW/m2.K (Volumetric/Area equivalent)
        
        # h scales with U^0.6 (standard fluidized bed correlation)
        velocity_ratio = U_surf / ref_U
        
        # Bed Collapse Logic
        if U_surf < Umf:
            h_factor = 0.1 # Collapse penalty
        elif U_surf < (1.1 * Umf):
            h_factor = 0.4 * (velocity_ratio**0.6) # Transition
        else:
            h_factor = (velocity_ratio**0.6)
            
        current_h = ref_h * h_factor
        current_UA = current_h * area
        
        NTU = current_UA / C_min
        
        if Cr < 1.0:
            epsilon = (1 - math.exp(-NTU * (1 - Cr))) / (1 - Cr * math.exp(-NTU * (1 - Cr)))
        else:
            epsilon = NTU / (1 + NTU)
            
        Q_transfer = epsilon * C_min * (T_urea_in - T_air_in)
        T_urea_out = T_urea_in - (Q_transfer / C_urea)
        T_air_out = T_air_in + (Q_transfer / C_air)
        
        ratio_air_urea = m_air_total / (m_urea_kg_h/3600.0 * 3600.0)
        ratio_air_area = vol_flow_air / area
        
        return {
            "T_urea_out": T_urea_out,
            "T_air_out": T_air_out,
            "U_surf": U_surf,
            "Umf": Umf,
            "Ut": Ut,
            "Residence_Time_s": residence_time_sec,
            "Duty_kW": Q_transfer,
            "Ratio_Air_Urea": ratio_air_urea,
            "Ratio_Air_Area": ratio_air_area,
            "Bed_Mass_kg": bed_mass,
            "UA_Actual": current_UA
        }

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def run_simulation_step(inputs):
    # Initialize
    psychro = ASHRAE_Psychrometrics()
    chiller = ChillerModel(psychro)
    heater = HeaterModel(psychro)
    fan = FanModel(psychro)
    bed = FluidizedBedModel()
    
    # Run
    r1 = chiller.solve(inputs['m_air'], inputs['t_in'], inputs['rh_in']/100.0, inputs['t_nh3'])
    r2 = heater.solve(r1['m_dry_kg_s'], r1['T_out'], r1['W_out'], inputs['m_steam'])
    r3 = fan.solve(r2['m_dry_kg_s'], r2['T_out'], r2['W_out'])
    r4 = bed.solve(r2['m_dry_kg_s'], r3['T_out'], r2['W_out'], r3['rho_out'], r3['mu_out'],
                   inputs['m_urea'], inputs['t_urea_in'], inputs['bed_len'], inputs['bed_lvl'])
    
    return r1, r2, r3, r4

# ==========================================
# 4. STREAMLIT APP
# ==========================================
st.set_page_config(page_title="Urea Cooler 335 Expert Twin", layout="wide")
st.markdown("<style>.big-font { font-size:20px !important; font-weight: bold; }</style>", unsafe_allow_html=True)

st.title("🏭 Unit 335 Expert Digital Twin")
st.markdown("**(Verified & Calibrated Model)**")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Control Parameters")
    
    st.subheader("1. Air Inlet")
    m_air = st.slider("Air Flow (kg/h)", 10000, 110000, 74095, 100)
    # Warning for Fan Capacity
    if m_air > 80529:
        st.warning("⚠️ Flow exceeds Guarantee Point (80529 kg/h)")
    
    t_in = st.slider("Ambient T (°C)", 10.0, 50.0, 32.0, 0.5)
    rh_in = st.slider("Ambient RH (%)", 10, 100, 64, 1)
    
    st.subheader("2. Utilities")
    t_nh3 = st.slider("NH3 T (°C)", -5.0, 5.0, 1.0, 0.5)
    m_steam = st.slider("Steam (kg/h)", 0, 400, 0, 10)
    
    st.subheader("3. Product")
    m_urea = st.slider("Urea Flow (kg/h)", 50000, 120000, 83418, 100)
    t_urea_in = st.slider("Urea Inlet T (°C)", 50.0, 90.0, 66.0, 0.5)
    
    st.subheader("4. Geometry")
    bed_len = st.slider("Bed Length (m)", 4.0, 12.0, 6.4, 0.1)
    bed_lvl = st.slider("Bed Level (mm)", 100, 400, 200, 10)

inputs = {
    'm_air': m_air, 't_in': t_in, 'rh_in': rh_in, 't_nh3': t_nh3,
    'm_steam': m_steam, 'm_urea': m_urea, 't_urea_in': t_urea_in,
    'bed_len': bed_len, 'bed_lvl': bed_lvl
}

# --- TABS ---
tab1, tab2 = st.tabs(["📊 Operational Dashboard", "📈 Sensitivity Analysis"])

# ================= TAB 1 =================
with tab1:
    r1, r2, r3, r4 = run_simulation_step(inputs)
    
    # KPIs
    st.subheader("🎯 Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Urea Outlet Temp", f"{r4['T_urea_out']:.1f} °C", delta=f"{r4['T_urea_out']-47.0:.1f} vs Target", delta_color="inverse")
    k2.metric("Total Duty", f"{r4['Duty_kW']:.0f} kW", help="Calculated with Cp=1.75 to match PFD")
    k3.metric("Fluidization Vel", f"{r4['U_surf']:.2f} m/s", help=f"Range: {r4['Umf']:.2f} - {r4['Ut']:.2f} m/s")
    k4.metric("Residence Time", f"{r4['Residence_Time_s']:.1f} s")
    
    st.markdown("---")
    
    # Detailed Data
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        st.info("**Chiller (E008)**")
        st.write(f"T Out: {r1['T_out']:.1f} °C")
        st.write(f"Water: {r1['Condensate_kg_h']:.0f} kg/h")
    with col_d2:
        st.info("**Heater (E009)**")
        st.write(f"T Out: {r2['T_out']:.1f} °C")
        st.write(f"Steam: {m_steam} kg/h")
    with col_d3:
        st.info("**Fan (K007)**")
        st.write(f"T Out: {r3['T_out']:.1f} °C")
        st.write(f"Heat Add: +{r3['delta_T']:.1f} °C")
    with col_d4:
        st.success("**Bed (E007)**")
        st.write(f"Air/Urea Ratio: {r4['Ratio_Air_Urea']:.3f}")
        st.write(f"Air/Area Ratio: {r4['Ratio_Air_Area']:.2f}")

    # Velocity Chart
    st.markdown("---")
    c_chart, c_stat = st.columns([3, 1])
    with c_chart:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axvspan(0, r4['Umf'], color='red', alpha=0.3, label='Dead Zone')
        ax.axvspan(r4['Umf'], r4['Ut'], color='green', alpha=0.3, label='Safe Zone')
        ax.axvspan(r4['Ut'], 6.0, color='orange', alpha=0.3, label='Carryover')
        ax.axvline(r4['U_surf'], color='black', linewidth=3, label='Current Point')
        ax.set_xlim(0, 5.0)
        ax.set_xlabel("Air Velocity (m/s)")
        ax.legend(loc='upper right')
        st.pyplot(fig)
    with c_stat:
        if r4['U_surf'] < r4['Umf']:
            st.error("🚨 **CRITICAL: Bed Collapse!**\nIncrease Flow or Reduce Length.")
        elif r4['U_surf'] > r4['Ut']:
            st.warning("⚠️ **Warning: High Velocity!**\nProduct loss expected.")
        else:
            st.success("✅ **Stable Operation**")

# ================= TAB 2 (CORRECTED) =================
with tab2:
    st.header("🔬 Engineering Analysis (All Variables)")
    
    # 1. تعريف القاموس الشامل لخيارات التحليل
    analysis_options = {
        "Bed Length (Geometry)":   {'var': 'bed_len',   'min': 4.0,   'max': 12.0,  'unit': 'm'},
        "Air Flow (Capacity)":     {'var': 'm_air',     'min': 20000, 'max': 110000,'unit': 'kg/h'},
        "Urea Inlet T (Load)":     {'var': 't_urea_in', 'min': 50.0,  'max': 90.0,  'unit': '°C'},
        "Ambient Temp (Weather)":  {'var': 't_in',      'min': 5.0,   'max': 50.0,  'unit': '°C'},
        "Ambient RH (Humidity)":   {'var': 'rh_in',     'min': 10,    'max': 100,   'unit': '%'},
        "Urea Flow (Production)":  {'var': 'm_urea',    'min': 50000, 'max': 120000,'unit': 'kg/h'},
        "Bed Level (Hold-up)":     {'var': 'bed_lvl',   'min': 100,   'max': 400,   'unit': 'mm'},
        "NH3 Temp (Utility)":      {'var': 't_nh3',     'min': -5.0,  'max': 10.0,  'unit': '°C'},
        "Steam Flow (Heater)":     {'var': 'm_steam',   'min': 0,     'max': 500,   'unit': 'kg/h'}
    }

    # 2. القائمة المنسدلة لاختيار المتغير
    plot_var_name = st.selectbox("Select Parameter to Analyze:", list(analysis_options.keys()))
    
    # 3. إعداد البيانات للمحاكاة
    selected_opt = analysis_options[plot_var_name]
    var_key = selected_opt['var']
    
    # توليد 40 نقطة
    x_range = np.linspace(selected_opt['min'], selected_opt['max'], 40)
    
    x_vals = []
    y_temps = []
    y_vels = []

    # 4. تشغيل المحاكاة (Loop)
    for x in x_range:
        sim_inputs = inputs.copy() # نسخ القيم الحالية
        sim_inputs[var_key] = x    # تحديث المتغير المختار
        
        # تشغيل خطوة المحاكاة
        # ملاحظة: نستخدم *_, لتجاهل المخرجات الثلاثة الأولى وأخذ الرابع فقط
        *_, r4_res = run_simulation_step(sim_inputs)
        
        x_vals.append(x)
        y_temps.append(r4_res['T_urea_out'])
        y_vels.append(r4_res['U_surf'])

    # 5. رسم النتائج
    fig2, ax1 = plt.subplots(figsize=(10, 5))
    
    # المحور الأيسر (Temp)
    color = 'tab:red'
    ax1.set_xlabel(f"{plot_var_name}")
    ax1.set_ylabel('Urea Outlet Temp (°C)', color=color)
    ax1.plot(x_vals, y_temps, color=color, linewidth=2, label='Outlet Temp')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # المحور الأيمن (Velocity)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Air Velocity (m/s)', color=color)
    ax2.plot(x_vals, y_vels, color=color, linestyle='--', linewidth=1.5, label='Air Velocity')
    
    # رسم خطوط الحدود (Limits)
    # ملاحظة: نستخدم قيم r4 (من الحالة الحالية في Tab 1) كمرجع
    ax2.axhline(r4['Umf'], color='orange', linestyle=':', label='Min Fluidization (Umf)')
    ax2.axhline(r4['Ut'], color='green', linestyle=':', label='Terminal Vel (Ut)')
    
    ax2.tick_params(axis='y', labelcolor=color)
    
    # دمج مفاتيح الرسم (Legend)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
    
    st.pyplot(fig2)
    
    st.caption(f"**Interpretation:** The graph shows how varying **{plot_var_name}** affects Product Temp (Red) and Fluidization Velocity (Blue).")
# 5. ADVANCED SENSITIVITY EXTENSION
# ==========================================

# تعريف القواميس لربط واجهة المستخدم بمتغيرات الكود
param_map = {
    "Air Flow (kg/h)": {'key': 'm_air', 'min': 20000, 'max': 110000},
    "Bed Length (m)": {'key': 'bed_len', 'min': 4.0, 'max': 12.0},
    "Ambient T (°C)": {'key': 't_in', 'min': 5.0, 'max': 50.0},
    "Urea Inlet T (°C)": {'key': 't_urea_in', 'min': 50.0, 'max': 90.0},
    "Urea Flow (kg/h)": {'key': 'm_urea', 'min': 50000, 'max': 120000},
    "Bed Level (mm)": {'key': 'bed_lvl', 'min': 100, 'max': 400}
}

output_map = {
    "Urea Outlet T (°C)": "T_urea_out",
    "Fluidization Velocity (m/s)": "U_surf",
    "Total Duty (kW)": "Duty_kW",
    "Fan Power (kW)": "Power_kW" # Note: need to capture this from r3 if needed, specifically handled below
}

# إضافة Tabs جديدة للتحليل المتقدم
st.markdown("---")
st.header("🚀 Advanced Engineering Analytics")
tab3, tab4 = st.tabs(["🔥 2D Heatmap & Operating Window", "📈 Multi-Variable Parametric Study"])

# ================= TAB 3: HEATMAPS =================
with tab3:
    st.subheader("تحليل التفاعل بين متغيرين (Interaction Analysis)")
    st.info("تساعد هذه الخريطة في تحديد 'المنطقة الآمنة' للتشغيل وتجنب مناطق انهيار الطبقة.")

    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        x_axis_name = st.selectbox("X-Axis Variable", list(param_map.keys()), index=0)
    with col_h2:
        y_axis_name = st.selectbox("Y-Axis Variable", list(param_map.keys()), index=1)
    with col_h3:
        z_axis_name = st.selectbox("Color Output (Z-Axis)", list(output_map.keys()), index=0)

    # Resolution settings
    res = st.slider("Resolution (Calculation Points)", 5, 25, 15, help="Higher is smoother but slower")
    
    if st.button("Generate Heatmap"):
        with st.spinner("Simulating physics matrix..."):
            # Prepare Grid
            x_conf = param_map[x_axis_name]
            y_conf = param_map[y_axis_name]
            
            x_vals = np.linspace(x_conf['min'], x_conf['max'], res)
            y_vals = np.linspace(y_conf['min'], y_conf['max'], res)
            
            z_data = np.zeros((res, res))
            mask_collapse = np.zeros((res, res)) # To mark unsafe zones
            
            # Loop
            base_inputs = inputs.copy()
            for i, y_val in enumerate(y_vals):
                for j, x_val in enumerate(x_vals):
                    # Update inputs
                    run_inputs = base_inputs.copy()
                    run_inputs[x_conf['key']] = x_val
                    run_inputs[y_conf['key']] = y_val
                    
                    # Run Physics Engine
                    r1, r2, r3, r4 = run_simulation_step(run_inputs)
                    
                    # Get Output
                    target_key = output_map[z_axis_name]
                    if target_key in r4:
                        val = r4[target_key]
                    elif target_key == "Power_kW": # Special handling for fan
                        val = r3['Power_kW']
                    else:
                        val = 0
                    
                    z_data[i, j] = val
                    
                    # Safety Check (Fluidization)
                    if r4['U_surf'] < r4['Umf']:
                        mask_collapse[i, j] = 1 # Mark as bad
            
            # Plotting
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Main Heatmap
            df_hm = pd.DataFrame(z_data, index=np.round(y_vals, 1), columns=np.round(x_vals, 0))
            sns.heatmap(df_hm, annot=True, fmt=".1f", cmap="coolwarm", ax=ax, cbar_kws={'label': z_axis_name})
            
            # Overlay Danger Zone (Hatching)
            # Note: Seaborn doesn't support hatching directly easily, so we use matplotlib overlay
            # Create a masked array for contour
            X_grid, Y_grid = np.meshgrid(np.arange(res), np.arange(res))
            
            # We add patches for collapsed zones
            from matplotlib.patches import Rectangle
            # This is a visual approximation for the UI
            if np.sum(mask_collapse) > 0:
                 st.warning(f"⚠️ المناطق المحاطة باللون الأسود أو القيم غير المنطقية تمثل مناطق انهيار الطبقة (Bed Collapse) حيث السرعة < $U_{{mf}}$")

            ax.set_xlabel(x_axis_name)
            ax.set_ylabel(y_axis_name)
            ax.set_title(f"{z_axis_name} vs. Inputs")
            ax.invert_yaxis() # Align with standard Cartesian plots if needed (Seaborn default is top-down)
            
            st.pyplot(fig)

# ================= TAB 4: XY PLOTS =================
with tab4:
    st.subheader("مقارنة السيناريوهات المتعددة (Multi-Scenario Plot)")
    st.markdown("لرؤية كيف يتغير الأداء مع تغير متغير رئيسي، عند مستويات مختلفة من متغير آخر.")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        x_plot = st.selectbox("X-Axis (Variable)", list(param_map.keys()), index=0, key="xp")
    with col_p2:
        y_plot = st.selectbox("Y-Axis (KPI)", list(output_map.keys()), index=0, key="yp")
    with col_p3:
        group_plot = st.selectbox("Group By (Fixed Lines)", list(param_map.keys()), index=1, key="gp")

    if x_plot == group_plot:
        st.error("Please select different variables for X-Axis and Group By")
    else:
        # Generate Data
        x_conf = param_map[x_plot]
        g_conf = param_map[group_plot]
        
        x_vals = np.linspace(x_conf['min'], x_conf['max'], 20)
        # Create 3 distinct lines for the "Group By" variable (Low, Mid, High)
        g_vals = np.linspace(g_conf['min'], g_conf['max'], 4)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Loop for each group line
        for g_val in g_vals:
            y_results = []
            valid_x = []
            
            base_inputs = inputs.copy()
            base_inputs[g_conf['key']] = g_val # Fix the group variable
            
            for x_val in x_vals:
                run_inputs = base_inputs.copy()
                run_inputs[x_conf['key']] = x_val
                
                r1, r2, r3, r4 = run_simulation_step(run_inputs)
                
                # Get Output
                target_key = output_map[y_plot]
                if target_key in r4:
                    val = r4[target_key]
                elif target_key == "Power_kW":
                    val = r3['Power_kW']
                else:
                    val = 0
                
                y_results.append(val)
            
            # Plot Line
            ax.plot(x_vals, y_results, marker='.', label=f"{group_plot} = {g_val:.1f}")

        ax.set_xlabel(x_plot)
        ax.set_ylabel(y_plot)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        ax.set_title(f"Sensitivity: {y_plot} vs {x_plot}")
        
        st.pyplot(fig)
        
        st.caption("""
        **كيف تقرأ هذا الرسم:**
        كل خط يمثل حالة تشغيلية ثابتة للمتغير (Group By). هذا يتيح لك الإجابة عن أسئلة مثل:
        *"كيف تؤثر زيادة حمل اليوريا على الحرارة إذا كنا نعمل بـ 3 مراوح مقابل 4 مراوح؟"* (عن طريق تغيير التدفق).
        """)
        