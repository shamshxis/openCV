import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
import os
import pandas as pd
import base64

# --- 1. Page Configuration ---
st.set_page_config(
    layout="wide", 
    page_title="EPIC Dashboard | Empirical Parameter Image-based Colonization", 
    initial_sidebar_state="expanded"
)

# --- 2. INITIALIZATION & DEFAULTS ---
defaults = {
    "stiffness": 800.0, "hydrophobicity": 80.0, 
    "charge": -20.0, "shear": 0.5, "energy": 25.0, "time_slider": 48,
    "hydro_state": True, "bypass_clamp": False, "chem_state": "Control (Physics Only)"
}
for key, val in defaults.items():
    if key not in st.session_state: 
        st.session_state[key] = val

# --- 3. Custom CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    div[data-testid="stToggle"] div[data-checked="true"] > div:first-child { background-color: #ffcc00 !important; }
    [data-testid="stSidebar"] .stSlider { margin-bottom: -15px; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { margin-bottom: -10px; }
    h1 { color: #00bcd4; }
    h2, h3, h4 { color: #f0f2f6; }
    .abstract-box {
        background-color: #161920; padding: 25px; border-radius: 15px;
        border-left: 6px solid #00bcd4; font-family: 'Segoe UI', sans-serif;
        line-height: 1.6; font-size: 1.05em; margin-bottom: 20px;
    }
    .footer { text-align: center; color: #888; font-family: 'Segoe UI', sans-serif; margin-top: 50px; padding: 20px; border-top: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. Dynamic Data Loading & Scaling Engine ---
@st.cache_data
def load_training_data():
    current_dir = os.getcwd()
    csv_path = os.path.join(current_dir, "data", "surface-parameters.csv")
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path), csv_path, True
        except Exception:
            return pd.DataFrame(), csv_path, False
    return pd.DataFrame(), csv_path, False

df_refs, loaded_path, is_loaded = load_training_data()

@st.cache_data
def load_bibliography():
    csv_path = os.path.join(os.getcwd(), "data", "bibliography.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path), True
    return pd.DataFrame(), False

df_bib, bib_loaded = load_bibliography()

def get_thresholds(df, property_name, default_floor, default_ceil):
    if df.empty or 'Property' not in df.columns: return default_floor, default_ceil
    subset = df[df['Property'] == property_name]
    if subset.empty: return default_floor, default_ceil
    return float(subset['Biological_Floor'].mean()), float(subset['Biological_Ceiling'].mean())

t_stiff_floor, t_stiff_ceil = get_thresholds(df_refs, "Stiffness", 10.0, 100.0)
t_rough_floor, t_rough_ceil = get_thresholds(df_refs, "Roughness", 0.0, 800.0)
t_hydro_floor, t_hydro_ceil = get_thresholds(df_refs, "Hydrophobicity", 28.0, 106.0)
t_charge_floor, t_charge_ceil = get_thresholds(df_refs, "Surface Charge", -18.0, 27.0)
t_shear_floor, t_shear_ceil = get_thresholds(df_refs, "Shear Stress", 0.0, 5.4)
t_energy_floor, t_energy_ceil = get_thresholds(df_refs, "Surface Energy", 20.0, 54.0)

# --- 5. Dashboard Headers ---
st.title("EPIC Dashboard")
st.subheader("Empirical Parameter Image-based Colonization for *P. aeruginosa*")
st.markdown("**Author(s):** Hamza Shams & Gemini AI") 
st.markdown("**Affiliation:** University of Oxford")
st.markdown("---")

# --- 6. Sidebar & Image Processing Pipeline ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Oxford-University-Circlet.svg/200px-Oxford-University-Circlet.svg.png", width=100)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.header("📸 1. Image Upload & Calibration")
uploaded_file = st.sidebar.file_uploader("Upload Surface Image (SEM, TIFF, PNG, JPG)", type=["png", "jpg", "jpeg", "tiff", "tif"])

# Standardized computational grid size for performance
GRID_RES = 150 
fov_microns = 100.0
x_surf = np.linspace(0, fov_microns, GRID_RES)
y_surf = np.linspace(0, fov_microns, GRID_RES)

if uploaded_file is not None:
    # 1. Image Processing
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_array

    # 2. Square Truncation & Downsampling to Standard Grid
    min_dim = min(gray_img.shape[0], gray_img.shape[1])
    square_img = gray_img[:min_dim, :min_dim] 
    resized_img = cv2.resize(square_img, (GRID_RES, GRID_RES), interpolation=cv2.INTER_AREA)

    # 3. Z-Height Calibration
    z_max_nm = st.sidebar.slider("Estimated Max Z-Height (nm)", min_value=10.0, max_value=5000.0, value=1000.0, step=10.0)
    Z_substrate = (resized_img / 255.0) * z_max_nm
    Z_substrate = Z_substrate - np.min(Z_substrate) # Normalize floor to 0
    
    # Calculate Global Roughness (Ra) directly from image
    mean_z = np.mean(Z_substrate)
    Ra_empirical = np.mean(np.abs(Z_substrate - mean_z))
    st.session_state.roughness = Ra_empirical

    # 4. ROI Selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 2. Local ROI Selector")
    x_range = st.sidebar.slider("X-Axis ROI (%)", 0, 100, (20, 80))
    y_range = st.sidebar.slider("Y-Axis ROI (%)", 0, 100, (20, 80))
    
    x_start = int((x_range[0] / 100) * GRID_RES)
    x_end = int((x_range[1] / 100) * GRID_RES)
    y_start = int((y_range[0] / 100) * GRID_RES)
    y_end = int((y_range[1] / 100) * GRID_RES)
    
    Z_roi = Z_substrate[y_start:y_end, x_start:x_end]
    if Z_roi.size > 0:
        st.sidebar.info(f"**ROI Metrics:**\n* **Max Peak:** {np.max(Z_roi):.1f} nm\n* **Avg Depth:** {np.mean(Z_roi):.1f} nm\n* **Local $R_a$:** {np.mean(np.abs(Z_roi - np.mean(Z_roi))):.1f} nm")
    
    # 5. Biological Environment Sliders
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 3. Biophysical Environment")
    st.sidebar.selectbox("Chemical State", ["Control (Physics Only)", "Biocidal Coating (Ag+/Cu2+)", "Stealth Coating (PEG)", "Protein Fouling"], key="chem_state")
    current_time = st.sidebar.slider("Incubation Time (Hours)", 0, 96, 48, key='time_slider')
    
    st.sidebar.slider("Stiffness (kPa)", 0.1, 1500.0, key='stiffness')
    st.sidebar.slider("Surface Charge (mV)", -50.0, 50.0, key='charge')
    st.sidebar.slider("Shear Stress (Pa)", 0.0, 15.0, key='shear')
    st.sidebar.slider("Surface Energy (mJ/m²)", 0.1, 100.0, key='energy')
    st.sidebar.toggle("Hydrophobic Regime (>90°)", key='hydro_state')
    st.sidebar.toggle("⚠️ Disable Biological Clamping", key='bypass_clamp')

    # --- 7. Biological Logic Calculations (Euler Engine) ---
    def apply_clamp(val, floor, ceil, bypass):
        if bypass: return val
        return np.clip(val, floor, ceil)

    c_stiff = apply_clamp(st.session_state.stiffness, t_stiff_floor, t_stiff_ceil, st.session_state.bypass_clamp)
    c_rough = apply_clamp(st.session_state.roughness, t_rough_floor, t_rough_ceil, st.session_state.bypass_clamp)
    c_charge = apply_clamp(st.session_state.charge, t_charge_floor, t_charge_ceil, st.session_state.bypass_clamp)
    c_shear = apply_clamp(st.session_state.shear, t_shear_floor, t_shear_ceil, st.session_state.bypass_clamp)
    c_energy = apply_clamp(st.session_state.energy, t_energy_floor, t_energy_ceil, st.session_state.bypass_clamp)
    h_val = apply_clamp(150.0 if st.session_state.hydro_state else 10.0, t_hydro_floor, t_hydro_ceil, st.session_state.bypass_clamp)

    s_n = (c_stiff - t_stiff_floor) / max(1.0, (t_stiff_ceil - t_stiff_floor))
    r_n = (c_rough - t_rough_floor) / max(1.0, (t_rough_ceil - t_rough_floor))
    h_n = (h_val - t_hydro_floor) / max(1.0, (t_hydro_ceil - t_hydro_floor))
    c_n = (c_charge - t_charge_floor) / max(1.0, (t_charge_ceil - t_charge_floor))
    sh_n = c_shear / max(1.0, t_shear_ceil)
    e_n = (c_energy - t_energy_floor) / max(1.0, (t_energy_ceil - t_energy_floor))

    f_veto = -(1.5 * sh_n) - (0.8 * e_n) 
    f_adh = (1.2 * s_n) + (1.2 * r_n) + (0.6 * h_n)
    f_bio = -(0.6 * c_n)
    base_score = f_adh + f_veto + f_bio

    t_arr = np.linspace(0, 96, 100)
    dt = t_arr[1] - t_arr[0]
    
    f_chem_curve = np.zeros_like(t_arr)
    if "Biocidal" in st.session_state.chem_state: f_chem_curve = -3.5 * np.exp(-0.06 * t_arr) 
    elif "Stealth" in st.session_state.chem_state: f_chem_curve = np.full_like(t_arr, -1.5)
    elif "Fouling" in st.session_state.chem_state: f_chem_curve = np.full_like(t_arr, 1.5)

    score_curve = base_score + f_chem_curve

    base_K, alpha, beta, B0 = 500.0, 3.0, 1.0, 10.0
    biomass_curve = np.zeros_like(t_arr)
    biomass_curve[0] = B0

    for i in range(1, len(t_arr)):
        s_i = score_curve[i]
        K_i = base_K / (1 + np.exp(-alpha * (s_i - beta)))
        if K_i > 5.0:
            r_i = 0.15 * (1 + max(0, s_i) * 0.5)
            dB = r_i * biomass_curve[i-1] * (1 - biomass_curve[i-1]/K_i) * dt
            biomass_curve[i] = biomass_curve[i-1] + dB
        else:
            biomass_curve[i] = max(1.0, biomass_curve[i-1] - 0.2 * biomass_curve[i-1] * dt)

    current_idx = np.abs(t_arr - current_time).argmin()
    score = score_curve[current_idx]
    biomass_at_t = biomass_curve[current_idx]

    if biomass_at_t > 200: status, color_status = "MATURE BIOFILM (STATIONARY)", "#00ffcc"
    elif biomass_at_t > 30: status, color_status = "COLONIZING (EXPONENTIAL)", "#ffcc00"
    else: status, color_status = "PLANKTONIC / CLEAN SURFACE", "#ff4444"

    # --- 8. Bacterial Topographical Spawning ---
    X, Y = np.meshgrid(x_surf, y_surf)
    x_bac, y_bac, z_bac = [], [], []
    if biomass_at_t > 0:
        spread = max(2.5, 40.0 * (1 - (biomass_at_t / base_K)))
        total_points = int(100 + (1000 * (biomass_at_t / base_K))) 
        n_clusters = max(5, int(total_points / 30))
        points_per_cluster = max(1, total_points // n_clusters)
        
        flat_indices = np.argsort(Z_substrate.flatten())
        valley_pool = flat_indices[:max(1, int(len(flat_indices) * 0.1))] 
        
        for _ in range(n_clusters):
            chosen_idx = np.random.choice(valley_pool)
            cy_idx, cx_idx = np.unravel_index(chosen_idx, Z_substrate.shape)
            cx, cy = x_surf[cx_idx], y_surf[cy_idx]
            x_bac.extend(np.random.normal(cx, spread, points_per_cluster))
            y_bac.extend(np.random.normal(cy, spread, points_per_cluster))
        
        x_bac = np.clip(x_bac, 0, fov_microns).tolist()
        y_bac = np.clip(y_bac, 0, fov_microns).tolist()
        ix = (np.array(x_bac) / fov_microns * (GRID_RES - 1)).astype(int)
        iy = (np.array(y_bac) / fov_microns * (GRID_RES - 1)).astype(int)
        
        if len(ix) > 0:
            base_z = Z_substrate[iy, ix]
            z_offset = np.max(Z_substrate) * (0.4 if "PLANKTONIC" in status else 0.015)
            z_bac = (base_z + z_offset).tolist()

    # --- 9. Visualizations ---
    col_state, col_score, col_rough = st.columns([2, 1, 1])
    with col_state:
        st.markdown(f"Status at {current_time}h: <span style='color:{color_status}; font-size:1.5em; font-weight:bold'>{status}</span>", unsafe_allow_html=True)
    with col_score: 
        st.metric("Net Risk Score (S)", f"{score:.2f}")
    with col_rough:
        st.metric("Empirical Roughness (Ra)", f"{Ra_empirical:.1f} nm")

    st.markdown("---")
    
    col_img, col_kin = st.columns([1, 1.2])
    with col_img:
        st.markdown("#### 🔬 Processed OpenCV Footprint")
        annotated_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(annotated_img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        st.image(annotated_img, caption="Red Bounding Box = Selected ROI", use_container_width=True)
        
    with col_kin:
        st.markdown("#### 📈 Growth Dynamics (96h)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=t_arr, y=biomass_curve, fill='tozeroy', line=dict(color='#00E5FF', width=3), name='Projected Growth'))
        fig2.add_vline(x=current_time, line_width=2, line_dash="dash", line_color="white")
        fig2.add_trace(go.Scatter(x=[current_time], y=[biomass_at_t], mode='markers', marker=dict(color='white', size=10, line=dict(width=2, color=color_status)), name='Current State'))
        fig2.update_layout(xaxis_title="Time (h)", yaxis_title="Biomass Density (OD600 eq)", template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🏔 3D Empirical Topography & Deposition")
    col_3d, col_2d = st.columns([3, 1.2]) 

    with col_2d:
        st.markdown("##### 🎚️ Profilometer Sweep")
        slice_y_idx = st.slider("Sweep X-Axis Profile", 0, GRID_RES-1, GRID_RES//2, label_visibility="collapsed")
        z_x_profile = Z_substrate[slice_y_idx, :]
        fig_x = go.Figure(go.Scatter(x=x_surf, y=z_x_profile, mode='lines', fill='tozeroy', line=dict(color='cyan', width=2)))
        
        if len(y_bac) > 0:
            bw = 2.0
            mask_x = (np.array(y_bac) >= y_surf[slice_y_idx] - bw) & (np.array(y_bac) <= y_surf[slice_y_idx] + bw)
            if np.any(mask_x):
                fig_x.add_trace(go.Scatter(x=np.array(x_bac)[mask_x], y=np.array(z_bac)[mask_x], mode='markers', marker=dict(color=color_status, size=6, line=dict(width=1, color='black'))))
        
        fig_x.update_layout(title=f"Profile at Y={y_surf[slice_y_idx]:.1f}µm", xaxis_title="X (µm)", yaxis_title="Height (nm)", template="plotly_dark", height=350, margin=dict(l=10, r=10, b=10, t=40), showlegend=False)
        st.plotly_chart(fig_x, use_container_width=True)

    with col_3d:
        fig3 = go.Figure()
        fig3.add_trace(go.Surface(z=Z_substrate, x=x_surf, y=y_surf, colorscale='Greys', showscale=False, opacity=0.9))
        fig3.add_trace(go.Scatter3d(x=x_surf, y=[y_surf[slice_y_idx]]*GRID_RES, z=Z_substrate[slice_y_idx, :], mode='lines', line=dict(color='cyan', width=5)))
        
        if len(x_bac) > 0:
            fig3.add_trace(go.Scatter3d(x=x_bac, y=y_bac, z=np.array(z_bac), mode='markers', marker=dict(size=4, color=color_status, opacity=1.0, line=dict(width=0.5, color='black'))))

        fig3.update_layout(scene=dict(xaxis_title='X (µm)', yaxis_title='Y (µm)', zaxis_title='Height (nm)', aspectratio=dict(x=1, y=1, z=0.25)), template="plotly_dark", height=500, margin=dict(l=0, r=0, b=0, t=0), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("⚠️ **Awaiting Data:** Please upload a surface image (SEM, TIFF, PNG) in the sidebar to initiate the topography mapping and colonization engine.")

# --- 10. Research Framework & References ---
st.markdown("---")
st.header("Research Framework & References")
with st.expander("📊 Mathematical Framework & Euler Integration"):
    st.markdown("""
    Instead of arbitrary weighting, this model treats biofilm formation as a **Hierarchy of Constraints**. 
    """)
    st.latex(r'''S = \underbrace{(1.2 S_n + 1.2 R_n + 0.6 H_n)}_{\text{Adhesion Promoters}} - \underbrace{(1.5 Sh_n + 0.8 E_n)}_{\text{Removal / Exclusion Forces}} - \underbrace{0.6 C_n}_{\text{Biological Inhibition}} + \underbrace{F_{chem}(t)}_{\text{Chemical Modifiers}}''')
    st.markdown("The growth engine utilizes a dynamic Verhulst-Pearl Logistic Equation integrated via Euler methods over 96 hours.")

with st.expander("📚 Bibliography"):
    st.markdown("<strong style='color:#00bcd4;'>Bibliography</strong>", unsafe_allow_html=True)
    if bib_loaded and not df_bib.empty and 'Citation' in df_bib.columns:
        for idx, row in df_bib.iterrows():
            st.markdown(f"<div style='font-size: 0.85em; margin-bottom: 4px;'>• {row['Citation']}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size: 0.85em;'>• Campoccia, D., Montanaro, L., & Arciola, C. R. (2013). A review of the biomaterials technologies for infection-resistant surfaces. <i>Biomaterials</i>, 34(34), 8533–8554.</div>", unsafe_allow_html=True)

# --- 11. Footer ---
footer_html = """
    <div style="display: flex; justify-content: center; align-items: center; position: relative; padding: 20px; border-top: 1px solid #333; margin-top: 50px;">
        <div align="center" style="color: #888; font-family: 'Segoe UI', sans-serif;">
            <strong>EPIC Dashboard</strong><br>
            All rights reserved &copy; University of Oxford, 2026.
        </div>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Oxford-University-Circlet.svg/200px-Oxford-University-Circlet.svg.png" style="height: 50px; position: absolute; right: 20px; opacity: 0.5;">
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
