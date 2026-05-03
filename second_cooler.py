import streamlit as st
import math
import numpy as np
import copy
import sys
import subprocess

# ==========================================
# 0. AUTO-INSTALL FIX
# ==========================================
try:
    import plotly.graph_objects as go
except ImportError:
    st.info("جاري تثبيت مكتبات الرسم المتقدمة...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.graph_objects as go

# ==========================================
# 1. ADVANCED PHYSICS ENGINE (Full Plant Cycle)
# ==========================================
class ASHRAE_Psychrometrics:
    def __init__(self):
        self.R_da = 287.042

    def get_properties(self, T_c, P_total_Pa, mode='RH', value=0.5):
        T_k = T_c + 273.15
        C1, C2, C3 = -5.8002206E+03, 1.3914993E+00, -4.8640239E-02
        C4, C5, C6 = 4.1764768E-05, -1.4452093E-08, 6.5459673E+00
        log_Pws = (C1/T_k) + C2 + (C3*T_k) + (C4*T_k**2) + (C5*T_k**3) + (C6*math.log(T_k))
        P_ws = math.exp(log_Pws)
        if mode == 'RH':
            P_w = value * P_ws
            W = 0.621945 * P_w / (P_total_Pa - P_w) if (P_total_Pa - P_w) > 0 else 0
        else: W = value

        h = 1.006 * T_c + W * (2501.0 + 1.805 * T_c)
        v = (self.R_da * T_k / P_total_Pa) * (1 + 1.6078 * W)
        rho = 1 / v if v > 0 else 0
        mu = (1.716e-5) * ((T_k/273.15)**1.5) * ((273.15 + 110.4)/(T_k + 110.4))
        return {'W': W, 'h': h, 'rho': rho, 'mu': mu}

class ChillerModel:
    def __init__(self, psychro):
        self.air = psychro
        self.pfd_m_wet = 74095.0     
        self.pfd_T_in = 32.0         
        self.pfd_T_out = 6.0         
        self.pfd_Duty = 1240.0       
        self.pfd_T_NH3 = 1.0         
        self.UA_calibrated = self._calibrate()

    def _calc_lmtd(self, T_air_in, T_air_out, T_ref):
        dt1 = max(T_air_in - T_ref, 0.01)
        dt2 = max(T_air_out - T_ref, 0.01)
        if abs(dt1 - dt2) < 0.01: return dt1
        return (dt1 - dt2) / math.log(dt1/dt2)

    def _calibrate(self):
        lmtd = self._calc_lmtd(self.pfd_T_in, self.pfd_T_out, self.pfd_T_NH3)
        return self.pfd_Duty / lmtd

    def solve(self, m_wet_kg_h, T_in, RH_in, T_NH3, area_factor=1.0):
        state_in = self.air.get_properties(T_in, 101325, 'RH', RH_in)
        m_dry_kg_s = (m_wet_kg_h / 3600.0) / (1 + state_in['W'])
        ratio_flow = m_wet_kg_h / self.pfd_m_wet
        current_UA = self.UA_calibrated * (ratio_flow ** 0.6) * area_factor
        
        T_min, T_max = T_NH3 + 0.1, T_in - 0.1
        T_final = T_max
        for i in range(30):
            T_guess = (T_min + T_max) / 2.0
            state_out = self.air.get_properties(T_guess, 101325, 'RH', 1.0)
            Q_air = m_dry_kg_s * (state_in['h'] - state_out['h'])
            lmtd = self._calc_lmtd(T_in, T_guess, T_NH3)
            Q_exch = current_UA * lmtd
            if abs(Q_exch - Q_air) < 0.1: 
                T_final = T_guess
                break
            if (Q_exch - Q_air) > 0: T_max = T_guess
            else: T_min = T_guess
        
        final_state = self.air.get_properties(T_final, 101325, 'RH', 1.0)
        duty_kw = m_dry_kg_s * (state_in['h'] - final_state['h'])
        return {"m_dry_kg_s": m_dry_kg_s, "W_out": final_state['W'], "T_out": T_final, "Duty_kW": duty_kw}

class HeaterModel:
    def __init__(self, psychro):
        self.air = psychro
        self.delta_h_steam = 2142.7 
    def solve(self, m_dry_kg_s, T_in_air, W_in_air, m_steam_kg_h):
        Q_steam_kW = (m_steam_kg_h / 3600.0) * self.delta_h_steam
        props_in = self.air.get_properties(T_in_air, 101325, 'W', W_in_air)
        h_target = props_in['h'] + (Q_steam_kW / m_dry_kg_s)
        T_min, T_max = T_in_air, T_in_air + 50.0
        T_final = T_in_air
        for i in range(30):
            T_mid = (T_min + T_max) / 2.0
            props_mid = self.air.get_properties(T_mid, 101325, 'W', W_in_air)
            if abs(props_mid['h'] - h_target) < 0.1:
                T_final = T_mid
                break
            if props_mid['h'] < h_target: T_min = T_mid
            else: T_max = T_mid
        return {"T_out": T_final, "W_out": W_in_air, "Duty_kW": Q_steam_kW}

class FanModel:
    def solve(self, vol_flow_m3_h, rho, bed_dp_mmH2O):
        total_dp_mmH2O = bed_dp_mmH2O + 30.0
        dp_pa = total_dp_mmH2O * 9.80665
        vol_m3_s = vol_flow_m3_h / 3600.0
        power_kW = (vol_m3_s * dp_pa) / (0.7 * 1000) 
        return power_kW

class FluidizedBedModel:
    def __init__(self):
        self.rho_particle = 780.0 
        self.rho_solid_true = 1330.0 
        self.dp = 3.0e-3 
        self.Cp_urea = 1.75  
        self.fixed_width = 1.30 
        
    def solve(self, m_dry_kg_s, T_air_in, W_air, rho_air_in, mu_air_in,
              m_urea_kg_h, T_urea_in, bed_len, bed_level_mm, air_bias_long=0.0, air_bias_lat=0.0, bypass_frac=0.0):
        
        area = bed_len * self.fixed_width
        m_air_total = m_dry_kg_s * (1 + W_air)
        vol_flow_air = m_air_total / rho_air_in
        U_surf_avg = vol_flow_air / area 
        
        g = 9.81
        Ar = (self.dp**3 * rho_air_in * (self.rho_solid_true - rho_air_in) * g) / (mu_air_in**2)
        Remf = math.sqrt(33.7**2 + 0.0408 * Ar) - 33.7
        Umf = (Remf * mu_air_in) / (self.dp * rho_air_in)
        Ut = 1.74 * math.sqrt((g * self.dp * (self.rho_solid_true - rho_air_in)) / rho_air_in)
        
        epsilon_mf = 1.0 - (self.rho_particle / self.rho_solid_true)
        epsilon = epsilon_mf if U_surf_avg < Umf else max(min(U_surf_avg / Ut, 0.99) ** (1.0 / 2.4), epsilon_mf)
        rho_bulk_dynamic = self.rho_solid_true * (1.0 - epsilon)
        
        bed_dp_mmH2O = rho_bulk_dynamic * (bed_level_mm / 1000.0)

        N_slices_X = 20
        N_slices_Y = 5
        d_area = area / (N_slices_X * N_slices_Y)
        
        m_urea_sec = m_urea_kg_h / 3600.0 
        C_urea_flow_total = m_urea_sec * self.Cp_urea 
        C_urea_flow_lane = C_urea_flow_total / N_slices_Y
        cp_air = 1.006 + 1.86 * W_air
        
        w_x = [max(0.1, 1.0 + (-air_bias_long)*(1.0 - 2.0*i/(N_slices_X-1)) if air_bias_long < 0 else 1.0 + (air_bias_long)*(2.0*i/(N_slices_X-1) - 1.0)) for i in range(N_slices_X)]
        norm_wx = [w / sum(w_x) for w in w_x]
        
        w_y = [max(0.01, 1.0 + air_bias_lat * ((abs(j - 2)/2.0)*2.0 - 1.0)) for j in range(N_slices_Y)]
        norm_wy = [w / sum(w_y) for w in w_y]
        
        T_urea_grid = np.zeros((N_slices_Y, N_slices_X))
        T_air_grid = np.zeros((N_slices_Y, N_slices_X))
        U_local_grid = np.zeros((N_slices_Y, N_slices_X))
        
        T_urea_current = [T_urea_in] * N_slices_Y 
        T_air_out_accum = 0.0 
        profile_x = [(i+0.5) * (bed_len/N_slices_X) for i in range(N_slices_X)]
        profile_y = [(j+0.5) * (self.fixed_width/N_slices_Y) for j in range(N_slices_Y)]

        for i in range(N_slices_X):
            for j in range(N_slices_Y):
                d_m_air = m_air_total * norm_wx[i] * norm_wy[j]
                U_local = (d_m_air / rho_air_in) / d_area
                U_local_grid[j][i] = U_local
                
                d_m_air_eff = d_m_air * (1.0 - bypass_frac)
                d_C_air_eff = d_m_air_eff * cp_air
                d_m_air_bypass = d_m_air * bypass_frac
                
                ref_U = 2.2 
                ref_h = 31.7 / 8.0 
                h_factor = 0.1 if U_local < Umf else (0.4 * ((U_local/ref_U)**0.6) if U_local < (1.1 * Umf) else ((U_local/ref_U)**0.6))
                height_factor = max(bed_level_mm / 150.0, 0.1) 
                d_UA = (ref_h * h_factor) * d_area * height_factor 

                C_min_slice = min(C_urea_flow_lane, d_C_air_eff)
                NTU_slice = d_UA / C_min_slice if C_min_slice > 0 else 0
                effectiveness = 1 - math.exp(-NTU_slice)
                
                Q_slice = effectiveness * C_min_slice * (T_urea_current[j] - T_air_in)
                
                T_urea_current[j] = T_urea_current[j] - (Q_slice / C_urea_flow_lane)
                T_urea_grid[j][i] = T_urea_current[j]
                
                d_T_air_eff_out = T_air_in + (Q_slice / d_C_air_eff) if d_C_air_eff > 0 else T_air_in
                
                if d_m_air > 0:
                    T_air_mixed = (d_m_air_eff * d_T_air_eff_out + d_m_air_bypass * T_air_in) / d_m_air
                else:
                    T_air_mixed = T_air_in
                    
                T_air_grid[j][i] = T_air_mixed
                T_air_out_accum += T_air_mixed * d_m_air

        return {
            "T_urea_out": sum(T_urea_current)/N_slices_Y, 
            "T_air_out": T_air_out_accum / m_air_total if m_air_total > 0 else T_air_in, 
            "U_surf": U_surf_avg, "Umf": Umf, "Ut": Ut, 
            "Bed_DP_mmH2O": bed_dp_mmH2O,
            "Profile_X": profile_x, "Profile_Y": profile_y,
            "T_urea_grid": T_urea_grid.tolist(),
            "U_local_grid": U_local_grid.tolist(),
            "Profile_T_Urea_Avg": np.mean(T_urea_grid, axis=0).tolist(),
            "Profile_T_Air_Avg": np.mean(T_air_grid, axis=0).tolist()
        }

def run_simulation(inputs):
    psychro = ASHRAE_Psychrometrics()
    
    # Ambient State
    props_amb = psychro.get_properties(inputs['t_amb'], 101325, 'RH', inputs['rh_amb']/100.0)
    nominal_vol = inputs['vol_air_nominal']
    actual_vol = nominal_vol * math.sqrt(150.0 / max(inputs['bed_lvl'], 10.0))
    m_air_wet_kgh = actual_vol * props_amb['rho']
    
    # 1. Chiller
    chiller = ChillerModel(psychro)
    res_ch = chiller.solve(m_air_wet_kgh, inputs['t_amb'], inputs['rh_amb']/100.0, inputs['t_nh3'], inputs['chiller_area']/100.0)
    
    # 2. Heater
    heater = HeaterModel(psychro)
    res_ht = heater.solve(res_ch['m_dry_kg_s'], res_ch['T_out'], res_ch['W_out'], inputs['m_steam'])
    
    # 3. Fan Heat & Power
    props_fan_in = psychro.get_properties(res_ht['T_out'], 101325, 'W', res_ht['W_out'])
    fan = FanModel()
    
    # 4. Fluidized Bed
    bed = FluidizedBedModel()
    r_bed = bed.solve(res_ch['m_dry_kg_s'], res_ht['T_out'], res_ht['W_out'], props_fan_in['rho'], props_fan_in['mu'],
                      inputs['m_urea'], inputs['t_urea_in'], inputs['bed_len'], inputs['bed_lvl'], 
                      inputs['air_bias_long'], inputs['air_bias_lat'], inputs['bypass_frac'])
    
    r_bed['Power_kW'] = fan.solve(actual_vol, props_fan_in['rho'], r_bed['Bed_DP_mmH2O'])
    r_bed['Actual_Vol_m3h'] = actual_vol
    r_bed['Actual_Mass_kgh'] = m_air_wet_kgh
    r_bed['Chiller_Duty'] = res_ch['Duty_kW']
    r_bed['Heater_Duty'] = res_ht['Duty_kW']
    r_bed['Air_Temp_To_Bed'] = res_ht['T_out']
    
    return r_bed

# ==========================================
# 4. MODERN SCADA UI (High Contrast & Consistent)
# ==========================================
st.set_page_config(page_title="SCADA: Urea Cooler 335 (Full Plant)", layout="wide", initial_sidebar_state="expanded")

# Clean, native-friendly styling for Streamlit elements
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 15px;
        text-align: right;
        margin-bottom: 15px;
        border-right: 4px solid #00E676;
    }
    .metric-title { font-size: 0.9rem; color: #AAAAAA; margin-bottom: 5px; font-weight: 600; }
    .metric-val { font-size: 1.8rem; color: #FFFFFF; font-weight: bold; }
    .metric-unit { font-size: 0.9rem; color: #888888; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; direction: rtl; color: #00E676;'>🏭 وحدة التبريد المتكاملة - SCADA Pro</h1>", unsafe_allow_html=True)

# --- SIDEBAR (COMPREHENSIVE CONTROLS) ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; direction: rtl; color: #00E676;'>⚙️ لوحة التحكم الشاملة</h2>", unsafe_allow_html=True)
    
    st.subheader("1. البيئة (Ambient)")
    t_amb = st.slider("حرارة الطقس (°C)", 10.0, 50.0, 32.0, 0.5)
    rh_amb = st.slider("رطوبة الطقس (%)", 10, 100, 64, 1)

    st.subheader("2. مبرد ومسخن الهواء (Pre-Conditioning)")
    t_nh3 = st.slider("حرارة أمونيا التبريد (°C)", -5.0, 15.0, 1.0, 0.5)
    chiller_area = st.slider("كفاءة سطح التبريد (%)", 50, 150, 100, 5)
    m_steam = st.slider("كمية بخار التسخين (kg/h)", 0, 5000, 0, 50)

    st.subheader("3. مروحة السحب (Main Fan)")
    vol_air_nominal = st.slider("أمر التشغيل (Nominal m³/h)", 20000, 150000, 85000, 500)
    
    st.subheader("4. مبرد اليوريا (Urea Bed)")
    m_urea = st.slider("معدل دخول المنتج (kg/h)", 50000, 120000, 83418, 100)
    t_urea_in = st.slider("حرارة المنتج الداخل (°C)", 50.0, 100.0, 66.0, 1.0)
    bed_len = st.slider("طول السرير (m)", 2.0, 12.0, 6.4, 0.1)
    bed_lvl = st.slider("مستوى السرير (mm)", 50, 200, 150, 5)

    st.subheader("5. معاملات التشوه (Maldistribution)")
    air_bias_long = st.slider("انحراف طولي (أمام/خلف)", -1.0, 1.0, 0.0, 0.1)
    air_bias_lat = st.slider("انحراف عرضي (أطراف/مركز)", 0.0, 1.0, 0.0, 0.1)
    bypass_frac = st.slider("نسبة التسريب/الهروب (%)", 0.0, 50.0, 0.0, 1.0) / 100.0

inputs = {
    't_amb': t_amb, 'rh_amb': rh_amb, 't_nh3': t_nh3, 'chiller_area': chiller_area, 'm_steam': m_steam,
    'vol_air_nominal': vol_air_nominal, 'm_urea': m_urea, 't_urea_in': t_urea_in,
    'bed_len': bed_len, 'bed_lvl': bed_lvl, 'air_bias_long': air_bias_long, 
    'air_bias_lat': air_bias_lat, 'bypass_frac': bypass_frac
}

res = run_simulation(inputs)

# --- TOP DASHBOARD: METRICS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"<div class='metric-card' style='border-color: {'#00E676' if res['T_urea_out'] <= 48 else '#F44336'};'><div class='metric-title'>متوسط حرارة المنتج الخارج</div><div class='metric-val' style='color: {'#00E676' if res['T_urea_out'] <= 48 else '#F44336'};'>{res['T_urea_out']:.1f} <span class='metric-unit'>°C</span></div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card' style='border-color: #2196F3;'><div class='metric-title'>حرارة الهواء الداخل للسرير</div><div class='metric-val'>{res['Air_Temp_To_Bed']:.1f} <span class='metric-unit'>°C</span></div></div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div class='metric-card' style='border-color: #FF9800;'><div class='metric-title'>حرارة العادم الكلية (Stack)</div><div class='metric-val'>{res['T_air_out']:.1f} <span class='metric-unit'>°C</span></div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card' style='border-color: #9C27B0;'><div class='metric-title'>حمل التبريد (Chiller Duty)</div><div class='metric-val'>{res['Chiller_Duty']:.0f} <span class='metric-unit'>kW</span></div></div>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<div class='metric-card' style='border-color: #03A9F4;'><div class='metric-title'>كتلة الهواء الفعلية</div><div class='metric-val'>{res['Actual_Mass_kgh'] / 1000:.1f}k <span class='metric-unit'>kg/h</span></div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card' style='border-color: #00BCD4;'><div class='metric-title'>حجم الهواء الفعلي</div><div class='metric-val'>{res['Actual_Vol_m3h'] / 1000:.1f}k <span class='metric-unit'>m³/h</span></div></div>", unsafe_allow_html=True)

with col4:
    status_c = "#00E676" if res['Umf'] < res['U_surf'] < res['Ut'] else ("#F44336" if res['U_surf'] < res['Umf'] else "#FF9800")
    st.markdown(f"<div class='metric-card' style='border-color: {status_c};'><div class='metric-title'>سرعة الهواء السطحية</div><div class='metric-val' style='color: {status_c};'>{res['U_surf']:.2f} <span class='metric-unit'>m/s</span></div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card' style='border-color: #607D8B;'><div class='metric-title'>طاقة المروحة / المقاومة</div><div class='metric-val'>{res['Power_kW']:.0f} <span class='metric-unit'>kW</span> / {res['Bed_DP_mmH2O']:.0f} <span class='metric-unit'>mmH2O</span></div></div>", unsafe_allow_html=True)

# Standard Dark Theme settings for Plotly
dark_layout = dict(
    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color="#E0E0E0"), xaxis=dict(gridcolor='#444444'), yaxis=dict(gridcolor='#444444')
)

# --- MIDDLE: HEATMAPS ---
st.markdown("<hr style='border-color: #444;'><h3 style='text-align: right; direction: rtl; color: #00E676;'>🔥 الخرائط الحرارية والديناميكية (2D View)</h3>", unsafe_allow_html=True)
h_col1, h_col2 = st.columns(2)
with h_col1:
    fig_hm_air = go.Figure(data=go.Heatmap(z=res['U_local_grid'], x=res['Profile_X'], y=res['Profile_Y'], colorscale='Teal', hoverongaps=False))
    fig_hm_air.update_layout(**dark_layout, title="سرعة هواء التبريد الموضعية (m/s)", xaxis_title="طول المبرد (m)", yaxis_title="عرض المبرد (m)")
    st.plotly_chart(fig_hm_air, use_container_width=True)
with h_col2:
    fig_hm_urea = go.Figure(data=go.Heatmap(z=res['T_urea_grid'], x=res['Profile_X'], y=res['Profile_Y'], colorscale='Inferno', hoverongaps=False))
    fig_hm_urea.update_layout(**dark_layout, title="درجة حرارة اليوريا الموضعية (°C)", xaxis_title="طول المبرد (m)", yaxis_title="عرض المبرد (m)")
    st.plotly_chart(fig_hm_urea, use_container_width=True)

# --- BOTTOM: ULTIMATE OPTIMIZER ---
st.markdown("<hr style='border-color: #444;'><h3 style='text-align: right; direction: rtl; color: #00E676;'>🎯 المحلل الهندسي الشامل (Full Plant Optimizer)</h3>", unsafe_allow_html=True)

# Comprehensive Variable Dictionary (All Inputs and Outputs)
all_inputs = {
    "معدل المنتج الداخل (kg/h)": ("m_urea", 50000, 120000),
    "حرارة المنتج الداخل (°C)": ("t_urea_in", 50.0, 100.0),
    "أمر تشغيل المروحة (m³/h)": ("vol_air_nominal", 20000, 150000),
    "طول السرير (m)": ("bed_len", 2.0, 12.0),
    "ارتفاع السرير (mm)": ("bed_lvl", 50, 200),
    "حرارة الطقس المحيط (°C)": ("t_amb", 10.0, 50.0),
    "رطوبة الطقس (%)": ("rh_amb", 10, 100),
    "حرارة الأمونيا للمبرد (°C)": ("t_nh3", -5.0, 15.0),
    "بخار التسخين (kg/h)": ("m_steam", 0, 5000),
    "الانحراف الطولي (Bias)": ("air_bias_long", -1.0, 1.0),
    "نسبة هروب الهواء (Bypass)": ("bypass_frac", 0.0, 0.5)
}

all_outputs = {
    "حرارة المنتج الخارج (°C)": lambda r: r['T_urea_out'],
    "حرارة الهواء العادم (°C)": lambda r: r['T_air_out'],
    "حرارة الهواء لسرير اليوريا (°C)": lambda r: r['Air_Temp_To_Bed'],
    "كمية الهواء الفعلي (m³/h)": lambda r: r['Actual_Vol_m3h'],
    "كتلة الهواء الفعلية (kg/h)": lambda r: r['Actual_Mass_kgh'],
    "السرعة السطحية للهواء (m/s)": lambda r: r['U_surf'],
    "قدرة المروحة المستهلكة (kW)": lambda r: r['Power_kW'],
    "حمل التبريد Chiller Duty (kW)": lambda r: r['Chiller_Duty'],
    "حمل التسخين Heater Duty (kW)": lambda r: r['Heater_Duty']
}

opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
with opt_col4:
    x_choice = st.selectbox("المحور الأفقي (Input X):", list(all_inputs.keys()), index=2)
with opt_col3:
    y_choice = st.selectbox("المحور العمودي/العمق (Input Y):", list(all_inputs.keys()), index=4)
with opt_col2:
    z_choice = st.selectbox("النتيجة اللونية/الارتفاع (Output Z):", list(all_outputs.keys()), index=0)
with opt_col1:
    plot_type = st.selectbox("نوع الرسم البياني:", ["خريطة كنتورية (Contour)", "سطح ثلاثي الأبعاد (3D Surface)", "نقاط مبعثرة (3D Scatter)"])

x_key, x_min, x_max = all_inputs[x_choice]
y_key, y_min, y_max = all_inputs[y_choice]

grid_size = 15
x_vals = np.linspace(x_min, x_max, grid_size)
y_vals = np.linspace(y_min, y_max, grid_size)
Z = np.zeros((grid_size, grid_size))

with st.spinner('جاري تشغيل محرك المحاكاة الشامل...'):
    for i in range(grid_size):
        for j in range(grid_size):
            sim_inputs = copy.deepcopy(inputs)
            sim_inputs[y_key] = y_vals[i] # Y-axis maps to rows
            sim_inputs[x_key] = x_vals[j] # X-axis maps to cols
            Z[i, j] = all_outputs[z_choice](run_simulation(sim_inputs))

curr_x = inputs[x_key]
curr_y = inputs[y_key]
curr_z = all_outputs[z_choice](res)

fig_opt = go.Figure()

if plot_type == "خريطة كنتورية (Contour)":
    fig_opt.add_trace(go.Contour(
        z=Z, x=x_vals, y=y_vals, colorscale='Plasma', 
        contours=dict(showlabels=True, labelfont=dict(size=12, color='white')),
        colorbar=dict(title="النتيجة (Z)")
    ))
    fig_opt.add_trace(go.Scatter(x=[curr_x], y=[curr_y], mode='markers', marker=dict(color='#00E676', size=16, symbol='star', line=dict(color='white', width=2)), name='الوضع الحالي'))
    fig_opt.update_layout(**dark_layout, title=f"تحليل الكنتور: أثر {x_choice} و {y_choice} على {z_choice}", xaxis_title=x_choice, yaxis_title=y_choice)

elif plot_type == "سطح ثلاثي الأبعاد (3D Surface)":
    fig_opt.add_trace(go.Surface(z=Z, x=x_vals, y=y_vals, colorscale='Plasma', opacity=0.9))
    fig_opt.add_trace(go.Scatter3d(x=[curr_x], y=[curr_y], z=[curr_z], mode='markers', marker=dict(color='#00E676', size=10, symbol='diamond', line=dict(color='white', width=2)), name='الوضع الحالي'))
    fig_opt.update_layout(template="plotly_dark", title=f"السطح الحراري لـ {z_choice}", scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), margin=dict(l=0, r=0, b=0, t=40))

elif plot_type == "نقاط مبعثرة (3D Scatter)":
    X_flat, Y_flat = np.meshgrid(x_vals, y_vals)
    fig_opt.add_trace(go.Scatter3d(
        x=X_flat.flatten(), y=Y_flat.flatten(), z=Z.flatten(), mode='markers',
        marker=dict(size=5, color=Z.flatten(), colorscale='Plasma', opacity=0.8)
    ))
    fig_opt.add_trace(go.Scatter3d(x=[curr_x], y=[curr_y], z=[curr_z], mode='markers', marker=dict(color='#00E676', size=12, symbol='star', line=dict(color='white', width=2)), name='الوضع الحالي'))
    fig_opt.update_layout(template="plotly_dark", title=f"نقاط التشغيل المحتملة لـ {z_choice}", scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), margin=dict(l=0, r=0, b=0, t=40))

st.plotly_chart(fig_opt, use_container_width=True)