import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os
import pandas as pd

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
    h2, h3, h4, h5 { color: #f0f2f6; }
    .footer { text-align: center; color: #888; font-family: 'Segoe UI', sans-serif; margin-top: 50px; padding: 20px; border-top: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. Dynamic Data Loading ---
@st.cache_data
def load_training_data():
    csv_path = os.path.join(os.getcwd(), "data", "surface-parameters.csv")
    if os.path.exists(csv_path):
        try: return pd.read_csv(csv_path), csv_path, True
        except Exception: return pd.DataFrame(), csv_path, False
    return pd.DataFrame(), csv_path, False

df_refs, loaded_path, is_loaded = load_training_data()

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

# --- 6. Sidebar Inputs ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Oxford-University-Circlet.svg/200px-Oxford-University-Circlet.svg.png", width=100)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.header("📸 1. Image Upload")
uploaded_file = st.sidebar.file_uploader("Upload Surface Image (SEM, TIFF, PNG)", type=["png", "jpg", "jpeg", "tiff", "tif"])

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 2. Biophysical Environment")
st.sidebar.selectbox("Chemical State", ["Control (Physics Only)", "Biocidal Coating (Ag+/Cu2+)", "Stealth Coating (PEG)", "Protein Fouling"], key="chem_state")
current_time = st.sidebar.slider("Incubation Time (Hours)", 0, 96, 48, key='time_slider')

st.sidebar.slider("Stiffness (kPa)", 0.1, 1500.0, key='stiffness')
st.sidebar.slider("Surface Charge (mV)", -50.0, 50.0, key='charge')
st.sidebar.slider("Shear Stress (Pa)", 0.0, 15.0, key='shear')
st.sidebar.slider("Surface Energy (mJ/m²)", 0.1, 100.0, key='energy')
st.sidebar.toggle("Hydrophobic Regime (>90°)", key='hydro_state')
st.sidebar.toggle("⚠️ Disable Biological Clamping", key='bypass_clamp')

# Standardized computational grid size
GRID_RES = 250 
fov_microns = 100.0
x_surf = np.linspace(0, fov_microns, GRID_RES)
y_surf = np.linspace(0, fov_microns, GRID_RES)

if uploaded_file is not None:
    # --- 7. Image Processing Pipeline (CLAHE & Resize) ---
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_array

    min_dim = min(gray_img.shape[0], gray_img.shape[1])
    square_img = gray_img[:min_dim, :min_dim] 
    
    # Apply CLAHE to rescue SEM contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(square_img)
    
    # Downsample for math matrix
    resized_img = cv2.resize(enhanced_img, (GRID_RES, GRID_RES), interpolation=cv2.INTER_AREA)

    # --- 8. Top Visuals & Topographical UI Controls ---
    col_orig, col_cv = st.columns(2)
    
    with col_orig:
        st.markdown("#### 🖼️ Original Uploaded Image")
        st.image(image, use_container_width=True)
        
    with col_cv:
        st.markdown("#### 🔬 OpenCV Processed Footprint")
        cv_img_placeholder = st.empty() 
        
        # Calibration & ROI Sliders grouped together
        st.markdown("##### 🎛️ Topographical Calibration")
        z_max_nm = st.slider("Estimated Max Z-Height (nm)", min_value=10.0, max_value=5000.0, value=1000.0, step=10.0)
        
        col_sx, col_sy = st.columns(2)
        with col_sx: x_range = st.slider("X-Axis ROI (%)", 0, 100, (20, 80))
        with col_sy: y_range = st.slider("Y-Axis ROI (%)", 0, 100, (20, 80))
        
        # Math Coordinates
        math_x_start, math_x_end = int((x_range[0] / 100) * GRID_RES), int((x_range[1] / 100) * GRID_RES)
        math_y_start, math_y_end = int((y_range[0] / 100) * GRID_RES), int((y_range[1] / 100) * GRID_RES)
        
        # UI Coordinates (drawing on high-res enhanced image)
        orig_dim = enhanced_img.shape[0]
        ui_x_start, ui_x_end = int((x_range[0] / 100) * orig_dim), int((x_range[1] / 100) * orig_dim)
        ui_y_start, ui_y_end = int((y_range[0] / 100) * orig_dim), int((y_range[1] / 100) * orig_dim)
        
        annotated_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(annotated_img, (ui_x_start, ui_y_start), (ui_x_end, ui_y_end), (255, 0, 0), max(1, orig_dim // 75))
        
        cv_img_placeholder.image(annotated_img, caption="Red Bounding Box = Selected ROI (Contrast Enhanced)", use_container_width=True)

    # --- 9. Mathematical Surface Generation ---
    Z_substrate = (resized_img / 255.0) * z_max_nm
    Z_substrate = Z_substrate - np.min(Z_substrate) 
    
    mean_z = np.mean(Z_substrate)
    Ra_empirical = np.mean(np.abs(Z_substrate - mean_z))
    st.session_state.roughness = Ra_empirical

    # ROI local metrics
    with col_cv:
        Z_roi = Z_substrate[math_y_start:math_y_end, math_x_start:math_x_end]
        if Z_roi.size > 0:
            st.info(f"**Local ROI Metrics:** Max Peak: `{np.max(Z_roi):.1f} nm` | Avg Depth: `{np.mean(Z_roi):.1f} nm` | Local $R_a$: `{np.mean(np.abs(Z_roi - np.mean(Z_roi))):.1f} nm`")

    st.markdown("---")

    # --- 10. Biological Logic Calculations (Euler Engine) ---
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

    base_score = (1.2 * s_n) + (1.2 * r_n) + (0.6 * h_n) - (1.5 * sh_n) - (0.8 * e_n) - (0.6 * c_n)

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
            dB = (0.15 * (1 + max(0, s_i) * 0.5)) * biomass_curve[i-1] * (1 - biomass_curve[i-1]/K_i) * dt
            biomass_curve[i] = biomass_curve[i-1] + dB
        else:
            biomass_curve[i] = max(1.0, biomass_curve[i-1] - 0.2 * biomass_curve[i-1] * dt)

    current_idx = np.abs(t_arr - current_time).argmin()
    score = score_curve[current_idx]
    biomass_at_t = biomass_curve[current_idx]

    if biomass_at_t > 200: status, color_status = "MATURE BIOFILM", "#00ffcc"
    elif biomass_at_t > 30: status, color_status = "COLONIZING", "#ffcc00"
    else: status, color_status = "PLANKTONIC", "#ff4444"

    # --- 11. Sidebar Growth Dynamics Plot ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📈 Growth Dynamics")
    fig_side = go.Figure()
    fig_side.add_trace(go.Scatter(x=t_arr, y=biomass_curve, fill='tozeroy', line=dict(color='#00E5FF', width=2)))
    fig_side.add_vline(x=current_time, line_width=2, line_dash="dash", line_color="white")
    fig_side.add_trace(go.Scatter(x=[current_time], y=[biomass_at_t], mode='markers', marker=dict(color='white', size=8, line=dict(width=1, color=color_status))))
    fig_side.update_layout(
        xaxis_title="Time (h)", yaxis_title="OD600 eq", template="plotly_dark", 
        height=250, margin=dict(l=0, r=0, t=10, b=0), showlegend=False
    )
    st.sidebar.plotly_chart(fig_side, use_container_width=True)

    # --- 12. Dashboard Status Metrics ---
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.markdown(f"**Simulation Status (t={current_time}h):** <span style='color:{color_status}; font-size:1.3em;'>{status}</span>", unsafe_allow_html=True)
    col_stat2.metric("Net Risk Score (S)", f"{score:.2f}")
    col_stat3.metric("Global Empirical Roughness (Ra)", f"{Ra_empirical:.1f} nm")
    st.markdown("---")

    # --- 13. Dynamic Topographical Spawning ---
    X, Y = np.meshgrid(x_surf, y_surf)
    x_bac, y_bac, z_bac = [], [], []
    if biomass_at_t > 0:
        spread = max(2.5, 40.0 * (1 - (biomass_at_t / base_K)))
        total_points = int(100 + (1000 * (biomass_at_t / base_K))) 
        n_clusters = max(5, int(total_points / 30))
        points_per_cluster = max(1, total_points // n_clusters)
        
        # Determine settlement pool based on Ra. Low Ra = scatter everywhere. High Ra = stick to valleys.
        valley_percentage = max(0.1, min(1.0, 30.0 / max(0.1, Ra_empirical)))
        
        flat_indices = np.argsort(Z_substrate.flatten())
        pool_size = max(1, int(len(flat_indices) * valley_percentage))
        valley_pool = flat_indices[:pool_size] 
        
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
            z_offset = np.max(Z_substrate) * (0.4 if "PLANKTONIC" in status else 0.015)
            z_bac = (Z_substrate[iy, ix] + z_offset).tolist()

    # --- 14. Bottom Visuals: 3D Topography & Profilometers ---
    st.markdown("#### 🏔 3D Empirical Topography & Profilometry")
    col_3d, col_prof = st.columns([2, 1.2]) 

    with col_prof:
        st.markdown("##### 🎚️ Profilometer Controls")
        
        # X-Axis Sweep
        slice_y_idx = st.slider("Sweep X-Axis Profile (Move Y)", 0, GRID_RES-1, GRID_RES//2, label_visibility="collapsed")
        z_x_profile = Z_substrate[slice_y_idx, :]
        fig_x = go.Figure(go.Scatter(x=x_surf, y=z_x_profile, mode='lines', fill='tozeroy', line=dict(color='cyan', width=2)))
        if len(y_bac) > 0:
            mask_x = (np.array(y_bac) >= y_surf[slice_y_idx] - 2.0) & (np.array(y_bac) <= y_surf[slice_y_idx] + 2.0)
            if np.any(mask_x):
                fig_x.add_trace(go.Scatter(x=np.array(x_bac)[mask_x], y=np.array(z_bac)[mask_x], mode='markers', marker=dict(color=color_status, size=6, line=dict(width=1, color='black'))))
        
        # Dynamic y-axis scaling so changes in z_max_nm are easily visible
        fig_x.update_layout(title=f"X-Profile (at Y={y_surf[slice_y_idx]:.1f}µm)", xaxis_title="X (µm)", yaxis_title="Height (nm)", yaxis=dict(range=[0, max(10, z_max_nm)]), template="plotly_dark", height=250, margin=dict(l=10, r=10, b=10, t=30), showlegend=False)
        st.plotly_chart(fig_x, use_container_width=True)

        # Y-Axis Sweep
        slice_x_idx = st.slider("Sweep Y-Axis Profile (Move X)", 0, GRID_RES-1, GRID_RES//2, label_visibility="collapsed")
        z_y_profile = Z_substrate[:, slice_x_idx]
        fig_y = go.Figure(go.Scatter(x=y_surf, y=z_y_profile, mode='lines', fill='tozeroy', line=dict(color='magenta', width=2)))
        if len(x_bac) > 0:
            mask_y = (np.array(x_bac) >= x_surf[slice_x_idx] - 2.0) & (np.array(x_bac) <= x_surf[slice_x_idx] + 2.0)
            if np.any(mask_y):
                fig_y.add_trace(go.Scatter(x=np.array(y_bac)[mask_y], y=np.array(z_bac)[mask_y], mode='markers', marker=dict(color=color_status, size=6, line=dict(width=1, color='black'))))
        
        fig_y.update_layout(title=f"Y-Profile (at X={x_surf[slice_x_idx]:.1f}µm)", xaxis_title="Y (µm)", yaxis_title="Height (nm)", yaxis=dict(range=[0, max(10, z_max_nm)]), template="plotly_dark", height=250, margin=dict(l=10, r=10, b=10, t=30), showlegend=False)
        st.plotly_chart(fig_y, use_container_width=True)

    with col_3d:
        fig3 = go.Figure()
        fig3.add_trace(go.Surface(z=Z_substrate, x=x_surf, y=y_surf, colorscale='Greys', showscale=False, opacity=0.9))
        
        # Add Slice Indicators
        fig3.add_trace(go.Scatter3d(x=x_surf, y=[y_surf[slice_y_idx]]*GRID_RES, z=Z_substrate[slice_y_idx, :], mode='lines', line=dict(color='cyan', width=4)))
        fig3.add_trace(go.Scatter3d(x=[x_surf[slice_x_idx]]*GRID_RES, y=y_surf, z=Z_substrate[:, slice_x_idx], mode='lines', line=dict(color='magenta', width=4)))
        
        if len(x_bac) > 0:
            fig3.add_trace(go.Scatter3d(x=x_bac, y=y_bac, z=np.array(z_bac), mode='markers', marker=dict(size=4, color=color_status, opacity=1.0, line=dict(width=0.5, color='black'))))

        # Explicitly setting the Z-axis range ensures the visual geometry correctly scales with the user's slider input
        fig3.update_layout(
            scene=dict(
                xaxis_title='X (µm)', 
                yaxis_title='Y (µm)', 
                zaxis_title='Height (nm)', 
                zaxis=dict(range=[0, max(10, z_max_nm)]),
                aspectratio=dict(x=1, y=1, z=0.4)
            ), 
            template="plotly_dark", height=580, margin=dict(l=0, r=0, b=0, t=0), showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("⚠️ **Awaiting Data:** Please upload a surface image in the sidebar to initiate the topography mapping and colonization engine.")

# --- 15. Footer ---
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
