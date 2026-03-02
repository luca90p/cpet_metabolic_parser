import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import xml.etree.ElementTree as ET
from dateutil import parser
from scipy.optimize import curve_fit

st.set_page_config(page_title="CPET Telemetry Dashboard", layout="wide")

# ==========================================
# 1. CORE ALGORITHMS (CPET PARSING)
# ==========================================
@st.cache_data
def process_cpet(df_raw):
    df = df_raw.copy()
    
    # Clean strings
    for c in df.columns:
        if df[c].dtype == 'object' and c not in ['t', 'Fase', 'Marker']:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.', regex=False), errors='coerce')
            
    # Standardize column names based on Metasoft output
    col_map = {'FC': 'HR', "V'O2": 'VO2', "V'CO2": 'VCO2', 'RER': 'RER', 'v': 'Speed', 
               'FAT': 'FAT', 'CHO': 'CHO', 'VT': 'VT', 'BF': 'BF', "V'O2/FC": 'O2Pulse'}
    
    # Rename existing columns
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    # Filter exercise phase
    df = df.dropna(subset=['HR', 'Speed']).copy()
    df = df[(df['HR'] > 90) & (df['Speed'] > 0)].copy()
    
    if df.empty:
        return None

    # Binning (Round to nearest BPM)
    df['_bin'] = df['HR'].round()
    df_binned = df.groupby('_bin', as_index=False).mean().drop(columns=['_bin'])

    # Smoothing (Rolling Mean)
    window = 15
    numeric_cols = [c for c in df_binned.columns if pd.api.types.is_numeric_dtype(df_binned[c])]
    for c in numeric_cols:
        df_binned[c] = df_binned[c].rolling(window=window, min_periods=1, center=True).mean()

    df_binned = df_binned.dropna(subset=['HR']).sort_values('HR').reset_index(drop=True)

    # Resampling (Interpolate 1 BPM steps)
    min_hr = math.ceil(df_binned['HR'].min())
    max_hr = math.floor(df_binned['HR'].max())
    new_hr = np.arange(min_hr, max_hr + 1.0, 1.0)
    
    new_data = {'HR': new_hr}
    for col in numeric_cols:
        if col != 'HR':
            # Interpolate only valid non-NaN data
            mask = ~df_binned[col].isna()
            if mask.sum() > 2:
                new_data[col] = np.interp(new_hr, df_binned.loc[mask, 'HR'], df_binned.loc[mask, col])
            else:
                new_data[col] = np.nan
                
    df_final = pd.DataFrame(new_data)
    
    # Calculate Energy Expenditure (Weir Equation) in kcal/h
    if 'VO2' in df_final.columns and 'VCO2' in df_final.columns:
        df_final['EE_kcal_h'] = (3.941 * df_final['VO2'] + 1.106 * df_final['VCO2']) * 60
        
    return df_final

def extract_metrics(df_final):
    metrics = {}
    
    # VT2 (Crossover dove RER >= 1.0)
    if 'RER' in df_final.columns:
        cross_idx = df_final[df_final['RER'] >= 1.0].index.min()
        metrics['VT2_HR'] = df_final.loc[cross_idx, 'HR'] if not pd.isna(cross_idx) else None

    # FatMax
    if 'FAT' in df_final.columns:
        fatmax_idx = df_final['FAT'].idxmax()
        metrics['FatMax_HR'] = df_final.loc[fatmax_idx, 'HR'] if not pd.isna(fatmax_idx) else None
        metrics['FatMax_g'] = df_final.loc[fatmax_idx, 'FAT'] if not pd.isna(fatmax_idx) else None

    # EE Digital Twin (Quadratic Fit)
    if 'EE_kcal_h' in df_final.columns:
        mask = ~df_final['EE_kcal_h'].isna()
        if mask.sum() > 5:
            p = np.polyfit(df_final.loc[mask, 'HR'], df_final.loc[mask, 'EE_kcal_h'], 2)
            metrics['EE_Poly'] = p

    # FAT/CHO Digital Twin (Polynomial Fits for TCX projection)
    if 'FAT' in df_final.columns and 'CHO' in df_final.columns:
        mask = (~df_final['FAT'].isna()) & (~df_final['CHO'].isna())
        if mask.sum() > 5:
            metrics['FAT_Poly'] = np.polyfit(df_final.loc[mask, 'HR'], df_final.loc[mask, 'FAT'], 3)
            metrics['CHO_Poly'] = np.polyfit(df_final.loc[mask, 'HR'], df_final.loc[mask, 'CHO'], 3)

    return metrics

# ==========================================
# 2. TCX PARSING & DIGITAL TWIN APPL.
# ==========================================
def parse_tcx_apply_twin(file_content, metrics):
    ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
    tree = ET.parse(file_content)
    root = tree.getroot()

    data = []
    for tp in root.findall('.//tcx:Trackpoint', ns):
        time_node = tp.find('tcx:Time', ns)
        hr_node = tp.find('.//tcx:HeartRateBpm/tcx:Value', ns)
        if time_node is not None and hr_node is not None:
            try:
                data.append({'Time': parser.isoparse(time_node.text), 'HR': int(hr_node.text)})
            except:
                pass

    df_tcx = pd.DataFrame(data)
    if df_tcx.empty:
        return None, 0

    df_tcx = df_tcx.sort_values('Time').reset_index(drop=True)
    df_tcx['dt'] = df_tcx['Time'].diff().dt.total_seconds().fillna(0)
    df_tcx['dt'] = np.minimum(df_tcx['dt'], 10) # Cap pauses
    
    # 1. Apply EE Model
    if 'EE_Poly' in metrics:
        p = metrics['EE_Poly']
        df_tcx['EE_kcal_h'] = p[0]*df_tcx['HR']**2 + p[1]*df_tcx['HR'] + p[2]
        df_tcx['EE_kcal_h'] = np.maximum(df_tcx['EE_kcal_h'], 70) # Floor
        df_tcx['Calories'] = df_tcx['EE_kcal_h'] * (df_tcx['dt'] / 3600)
    else:
        df_tcx['Calories'] = 0

    # 2. Apply Substrate Models
    if 'FAT_Poly' in metrics and 'CHO_Poly' in metrics:
        p_fat = metrics['FAT_Poly']
        p_cho = metrics['CHO_Poly']
        df_tcx['FAT_g_h'] = np.polyval(p_fat, df_tcx['HR'])
        df_tcx['CHO_g_h'] = np.polyval(p_cho, df_tcx['HR'])
        
        # Zero out negative consumptions
        df_tcx['FAT_g_h'] = np.maximum(df_tcx['FAT_g_h'], 0)
        df_tcx['CHO_g_h'] = np.maximum(df_tcx['CHO_g_h'], 0)
        
        df_tcx['FAT_g'] = df_tcx['FAT_g_h'] * (df_tcx['dt'] / 3600)
        df_tcx['CHO_g'] = df_tcx['CHO_g_h'] * (df_tcx['dt'] / 3600)

    df_tcx['Time_min'] = df_tcx['dt'].cumsum() / 60
    
    # Garmin reported calories
    tcx_calories = 0
    for lap in root.findall('.//tcx:Lap', ns):
        cal_node = lap.find('tcx:Calories', ns)
        if cal_node is not None:
            tcx_calories += int(cal_node.text)

    return df_tcx, tcx_calories

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.title("🏃‍♂️ Digital Twin Fisiologico (CPET & TCX Analyzer)")

# Sidebar for CPET upload
st.sidebar.header("1. Setup Iniziale (CPET)")
cpet_file = st.sidebar.file_uploader("Carica CSV esportato da Metasoft", type=['csv'])

if cpet_file:
    # Cerchiamo l'header
    try:
        df_temp = pd.read_csv(cpet_file)
        df_final = process_cpet(df_temp)
        metrics = extract_metrics(df_final)
        
        st.sidebar.success("CPET Analizzato con successo!")
        st.sidebar.markdown(f"**VT2 (Crossover):** {metrics.get('VT2_HR', 0):.0f} bpm")
        st.sidebar.markdown(f"**FatMax:** {metrics.get('FatMax_HR', 0):.0f} bpm")
        
    except Exception as e:
        st.error(f"Errore nella lettura del CPET: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📊 Analisi CPET (Motore)", "🔥 Report Termodinamico", "⌚ Analisi Allenamento (TCX)"])

    with tab1:
        st.subheader("Dinamica Stechiometrica (RER vs HR)")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_final['HR'], df_final['RER'], color='red', linewidth=3)
        ax.axhline(y=1.0, color='black', linestyle='--')
        ax.axvline(x=metrics.get('VT2_HR', 170), color='blue', linestyle='--', label='VT2')
        ax.set_ylabel('RER (VCO2 / VO2)')
        ax.set_xlabel('Frequenza Cardiaca (bpm)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with tab2:
        st.subheader("Dispendio Energetico (Weir Eq.) e Digital Twin")
        if 'EE_kcal_h' in df_final.columns:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(df_final['HR'], df_final['EE_kcal_h'], 'b-', linewidth=3, label="Dati Reali (Weir)")
            if 'EE_Poly' in metrics:
                p = metrics['EE_Poly']
                ee_pred = p[0]*df_final['HR']**2 + p[1]*df_final['HR'] + p[2]
                ax2.plot(df_final['HR'], ee_pred, 'r--', linewidth=2, label="Modello Matematico")
            ax2.set_ylabel('Kcal/h')
            ax2.set_xlabel('Frequenza Cardiaca (bpm)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

    with tab3:
        st.subheader("Collaudo su Allenamento Reale")
        tcx_file = st.file_uploader("Carica un file .TCX dal tuo Garmin", type=['tcx'])
        
        if tcx_file:
            df_tcx, garmin_cal = parse_tcx_apply_twin(tcx_file, metrics)
            if df_tcx is not None:
                real_cal = df_tcx['Calories'].sum()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Durata (min)", f"{df_tcx['Time_min'].max():.1f}")
                col2.metric("FC Media (bpm)", f"{df_tcx['HR'].mean():.0f}")
                col3.metric("Calorie Reali (Digital Twin)", f"{real_cal:.0f} kcal")
                col4.metric("Calorie Garmin (Errore)", f"{garmin_cal} kcal", f"Sovrastima: {garmin_cal - real_cal:.0f} kcal", delta_color="inverse")
                
                if 'FAT_g' in df_tcx.columns:
                    st.markdown("### Consumo Substrati")
                    colA, colB = st.columns(2)
                    colA.metric("🔥 Grassi Ossidati", f"{df_tcx['FAT_g'].sum():.1f} g")
                    colB.metric("⚡ Carboidrati Ossidati", f"{df_tcx['CHO_g'].sum():.1f} g")

                st.markdown("### Profilo Energetico Istantaneo")
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.fill_between(df_tcx['Time_min'], df_tcx['EE_kcal_h'], color='firebrick', alpha=0.3)
                ax3.plot(df_tcx['Time_min'], df_tcx['EE_kcal_h'], color='firebrick', linewidth=1.5, label='EE Istantaneo (Kcal/h)')
                ax3.set_ylabel('Kcal/h')
                
                ax4 = ax3.twinx()
                ax4.plot(df_tcx['Time_min'], df_tcx['HR'], color='blue', alpha=0.5, linewidth=1, label='FC (bpm)')
                ax4.set_ylabel('FC (bpm)')
                
                st.pyplot(fig3)
else:
    st.info("👈 Inizia caricando il file CSV del tuo test CPET nella barra laterale per generare il tuo Digital Twin.")
