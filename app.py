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
    
    col_map = {'FC': 'HR', "V'O2": 'VO2', "V'CO2": 'VCO2', 'RER': 'RER', 'v': 'Speed', 
               'FAT': 'FAT', 'CHO': 'CHO', 'VT': 'VT', 'BF': 'BF', "V'O2/FC": 'O2Pulse'}
    
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    keep_cols = [v for k, v in col_map.items() if v in df.columns]
    df = df[keep_cols].copy()
    
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    
    df = df.dropna(subset=['HR', 'Speed']).copy()
    df = df[(df['HR'] > 90) & (df['Speed'] > 0)].copy()
    
    if df.empty:
        return None

    # Binning e Media Mobile
    df['_bin'] = df['HR'].round()
    df_binned = df.groupby('_bin', as_index=False).mean(numeric_only=True)
    if '_bin' in df_binned.columns:
        df_binned = df_binned.drop(columns=['_bin'])

    window = 15
    for c in df_binned.columns:
        df_binned[c] = df_binned[c].rolling(window=window, min_periods=1, center=True).mean()

    df_binned = df_binned.dropna(subset=['HR']).sort_values('HR').reset_index(drop=True)

    # Resampling Unitario
    min_hr = math.ceil(df_binned['HR'].min())
    max_hr = math.floor(df_binned['HR'].max())
    new_hr = np.arange(min_hr, max_hr + 1.0, 1.0)
    
    new_data = {'HR': new_hr}
    for col in df_binned.columns:
        if col != 'HR':
            mask = ~df_binned[col].isna()
            if mask.sum() > 2:
                new_data[col] = np.interp(new_hr, df_binned.loc[mask, 'HR'], df_binned.loc[mask, col])
            else:
                new_data[col] = np.nan
                
    df_final = pd.DataFrame(new_data)
    
    if 'VO2' in df_final.columns and 'VCO2' in df_final.columns:
        df_final['EE_kcal_h'] = (3.941 * df_final['VO2'] + 1.106 * df_final['VCO2']) * 60
        
    return df_final, df

def extract_metrics(df_final):
    metrics = {}
    
    # Valori di Picco registrati
    metrics['VO2_peak_abs'] = df_final['VO2'].max() if 'VO2' in df_final.columns else 0
    metrics['HR_peak_test'] = df_final['HR'].max() if 'HR' in df_final.columns else 0
    
    # VT2 (Crossover dove RER >= 1.0)
    if 'RER' in df_final.columns:
        cross_idx = df_final[df_final['RER'] >= 1.0].index.min()
        metrics['VT2_HR'] = df_final.loc[cross_idx, 'HR'] if not pd.isna(cross_idx) else None

    # FatMax
    if 'FAT' in df_final.columns:
        fatmax_idx = df_final['FAT'].idxmax()
        metrics['FatMax_HR'] = df_final.loc[fatmax_idx, 'HR'] if not pd.isna(fatmax_idx) else None
        metrics['FatMax_g'] = df_final.loc[fatmax_idx, 'FAT'] if not pd.isna(fatmax_idx) else None

    # Plateau O2 Pulse
    if 'O2Pulse' in df_final.columns:
        max_pulse = df_final['O2Pulse'].max()
        metrics['O2Pulse_Plateau_HR'] = df_final[df_final['O2Pulse'] >= max_pulse * 0.98]['HR'].min()

    # EE Digital Twin
    if 'EE_kcal_h' in df_final.columns:
        mask = ~df_final['EE_kcal_h'].isna()
        if mask.sum() > 5:
            p = np.polyfit(df_final.loc[mask, 'HR'], df_final.loc[mask, 'EE_kcal_h'], 2)
            metrics['EE_Poly'] = p

    # FAT/CHO Digital Twin
    if 'FAT' in df_final.columns and 'CHO' in df_final.columns:
        mask = (~df_final['FAT'].isna()) & (~df_final['CHO'].isna())
        if mask.sum() > 5:
            metrics['FAT_Poly'] = np.polyfit(df_final.loc[mask, 'HR'], df_final.loc[mask, 'FAT'], 3)
            metrics['CHO_Poly'] = np.polyfit(df_final.loc[mask, 'HR'], df_final.loc[mask, 'CHO'], 3)

    return metrics

# ==========================================
# 2. TCX PARSING & DIGITAL TWIN APPL.
# ==========================================
# [IL BLOCCO parse_tcx_apply_twin RIMANE IDENTICO A PRIMA]
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
    
    if 'EE_Poly' in metrics:
        p = metrics['EE_Poly']
        df_tcx['EE_kcal_h'] = p[0]*df_tcx['HR']**2 + p[1]*df_tcx['HR'] + p[2]
        df_tcx['EE_kcal_h'] = np.maximum(df_tcx['EE_kcal_h'], 70) # Floor
        df_tcx['Calories'] = df_tcx['EE_kcal_h'] * (df_tcx['dt'] / 3600)
    else:
        df_tcx['Calories'] = 0

    if 'FAT_Poly' in metrics and 'CHO_Poly' in metrics:
        p_fat = metrics['FAT_Poly']
        p_cho = metrics['CHO_Poly']
        df_tcx['FAT_g_h'] = np.polyval(p_fat, df_tcx['HR'])
        df_tcx['CHO_g_h'] = np.polyval(p_cho, df_tcx['HR'])
        
        df_tcx['FAT_g_h'] = np.maximum(df_tcx['FAT_g_h'], 0)
        df_tcx['CHO_g_h'] = np.maximum(df_tcx['CHO_g_h'], 0)
        
        df_tcx['FAT_g'] = df_tcx['FAT_g_h'] * (df_tcx['dt'] / 3600)
        df_tcx['CHO_g'] = df_tcx['CHO_g_h'] * (df_tcx['dt'] / 3600)

    df_tcx['Time_min'] = df_tcx['dt'].cumsum() / 60
    
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

st.sidebar.header("1. Setup Iniziale (CPET)")
cpet_file = st.sidebar.file_uploader("Carica CSV esportato da Metasoft", type=['csv'])

# Inputs aggiuntivi per le estrapolazioni
peso_kg = st.sidebar.number_input("Peso Corporeo (kg) al test", value=75.0, step=0.5)
fc_max_reale = st.sidebar.number_input("FC Max Reale Conosciuta (bpm)", value=182, step=1)

if cpet_file:
    try:
        # 1. Tentiamo di leggere la prima riga per vedere se è il file già pulito
        df_test = pd.read_csv(cpet_file, nrows=2)
        cpet_file.seek(0) # Riportiamo il puntatore del file a zero
        
        # 2. Controllo Intelligente dell'Header
        if 'FC' in df_test.columns or 'HR' in df_test.columns:
            # È il file già pulito o processato
            df_temp = pd.read_csv(cpet_file)
        else:
            # È il file grezzo uscito direttamente da Metasoft (saltiamo l'intestazione clinica)
            df_temp = pd.read_csv(cpet_file, skiprows=116)
            
        df_final, df_raw_filtered = process_cpet(df_temp)
        metrics = extract_metrics(df_final)
        
        # Calcolo Estrapolazione VO2Max
        vo2_peak_ml_kg = (metrics['VO2_peak_abs'] * 1000) / peso_kg
        vo2_max_stimato = vo2_peak_ml_kg * (fc_max_reale / metrics['HR_peak_test'])
        
        st.sidebar.success("CPET Analizzato con successo!")
        st.sidebar.markdown(f"**VO2 Peak Misurato:** {vo2_peak_ml_kg:.1f} ml/kg/min")
        st.sidebar.markdown(f"**VO2 Max Stimato:** {vo2_max_stimato:.1f} ml/kg/min")
        st.sidebar.markdown(f"**VT2 (Soglia):** {metrics.get('VT2_HR', 0):.0f} bpm")
        st.sidebar.markdown(f"**FatMax:** {metrics.get('FatMax_HR', 0):.0f} bpm")
        
    except Exception as e:
        st.error(f"Errore nella lettura del CPET: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📊 Analisi CPET (Motore)", "🔥 Report Termodinamico", "⌚ Analisi Allenamento (TCX)"])

    with tab1:
        st.header("Mappatura Hardware e Centralina")
        
        col1, col2 = st.columns(2)
        
        # 1. Grafico VO2 vs VCO2 (Crossover)
        with col1:
            st.subheader("Cinetica dei Gas (Ricerca VT2)")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(df_final['HR'], df_final['VO2'], color='blue', linewidth=2, label="V'O2")
            ax1.plot(df_final['HR'], df_final['VCO2'], color='red', linewidth=2, label="V'CO2")
            ax1.fill_between(df_final['HR'], df_final['VO2'], df_final['VCO2'], 
                             where=(df_final['VCO2'] > df_final['VO2']), color='red', alpha=0.2)
            ax1.axvline(x=metrics.get('VT2_HR', 0), color='black', linestyle='--', label=f"VT2 ({metrics.get('VT2_HR', 0):.0f} bpm)")
            ax1.set_xlabel('Frequenza Cardiaca (bpm)')
            ax1.set_ylabel('L/min')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)

        # 2. Grafico RER (Stechiometria)
        with col2:
            st.subheader("Rapporto Stechiometrico (RER)")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(df_final['HR'], df_final['RER'], color='darkred', linewidth=2, label='RER')
            ax2.axhline(y=1.0, color='black', linestyle='--')
            ax2.axvline(x=metrics.get('VT2_HR', 0), color='blue', linestyle='--', label='VT2')
            ax2.axhline(y=0.85, color='green', linestyle=':', label='Crociera 0.85')
            ax2.set_xlabel('Frequenza Cardiaca (bpm)')
            ax2.set_ylabel('VCO2 / VO2')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
        col3, col4 = st.columns(2)

        # 3. Grafico Substrati (FatMax)
        with col3:
            st.subheader("Ossidazione Substrati (FatMax)")
            if 'FAT' in df_final.columns and 'CHO' in df_final.columns:
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                ax3.plot(df_final['HR'], df_final['FAT'], color='green', linewidth=2, label='Grassi (g/h)')
                ax3.axvline(x=metrics.get('FatMax_HR', 0), color='green', linestyle='--', label=f"FatMax ({metrics.get('FatMax_HR', 0):.0f} bpm)")
                ax3_twin = ax3.twinx()
                ax3_twin.plot(df_final['HR'], df_final['CHO'], color='orange', linewidth=2, label='Carboidrati (g/h)')
                ax3.set_xlabel('Frequenza Cardiaca (bpm)')
                ax3.set_ylabel('Grassi (g/h)', color='green')
                ax3_twin.set_ylabel('Carboidrati (g/h)', color='orange')
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)
            else:
                st.info("Dati FAT/CHO non disponibili nel file.")

        # 4. Efficienza Idraulica (O2 Pulse)
        with col4:
            st.subheader("Gittata Sistolica (O2 Pulse)")
            if 'O2Pulse' in df_final.columns:
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                ax4.plot(df_final['HR'], df_final['O2Pulse'], color='darkcyan', linewidth=2, label='O2 Pulse')
                ax4.scatter(df_raw_filtered['HR'], df_raw_filtered['O2Pulse'], color='cyan', alpha=0.1, s=10)
                if 'O2Pulse_Plateau_HR' in metrics:
                    ax4.axvline(x=metrics['O2Pulse_Plateau_HR'], color='orange', linestyle='--', label=f"Plateau ({metrics['O2Pulse_Plateau_HR']:.0f} bpm)")
                ax4.set_xlabel('Frequenza Cardiaca (bpm)')
                ax4.set_ylabel('ml O2 / battito')
                ax4.legend(fontsize=8)
                ax4.grid(True, alpha=0.3)
                st.pyplot(fig4)
            else:
                st.info("Dati O2 Pulse non disponibili.")

    with tab2:
        st.subheader("Dispendio Energetico (Weir Eq.) e Digital Twin")
        # [IL BLOCCO DEL TAB 2 RIMANE IDENTICO A PRIMA]
        if 'EE_kcal_h' in df_final.columns:
            fig_ee, ax_ee = plt.subplots(figsize=(10, 5))
            ax_ee.plot(df_final['HR'], df_final['EE_kcal_h'], 'b-', linewidth=3, label="Dati Reali (Weir)")
            if 'EE_Poly' in metrics:
                p = metrics['EE_Poly']
                ee_pred = p[0]*df_final['HR']**2 + p[1]*df_final['HR'] + p[2]
                ax_ee.plot(df_final['HR'], ee_pred, 'r--', linewidth=2, label="Modello Matematico (Digital Twin)")
            ax_ee.set_ylabel('Kcal/h')
            ax_ee.set_xlabel('Frequenza Cardiaca (bpm)')
            ax_ee.legend()
            ax_ee.grid(True, alpha=0.3)
            st.pyplot(fig_ee)

    with tab3:
        # [IL BLOCCO DEL TAB 3 RIMANE IDENTICO A PRIMA]
        st.subheader("Collaudo su Allenamento Reale")
        tcx_file = st.file_uploader("Carica un file .TCX dal tuo Garmin", type=['tcx'])
        
        if tcx_file:
            df_tcx, garmin_cal = parse_tcx_apply_twin(tcx_file, metrics)
            if df_tcx is not None:
                real_cal = df_tcx['Calories'].sum()
                
                colA, colB, colC, colD = st.columns(4)
                colA.metric("Durata (min)", f"{df_tcx['Time_min'].max():.1f}")
                colB.metric("FC Media (bpm)", f"{df_tcx['HR'].mean():.0f}")
                colC.metric("Calorie Reali", f"{real_cal:.0f} kcal")
                colD.metric("Calorie Garmin", f"{garmin_cal} kcal", f"Sovrastima: {garmin_cal - real_cal:.0f} kcal", delta_color="inverse")
                
                if 'FAT_g' in df_tcx.columns:
                    st.markdown("### Consumo Substrati")
                    colE, colF = st.columns(2)
                    colE.metric("🔥 Grassi Ossidati", f"{df_tcx['FAT_g'].sum():.1f} g")
                    colF.metric("⚡ Carboidrati Ossidati", f"{df_tcx['CHO_g'].sum():.1f} g")

                fig_tcx, ax_tcx = plt.subplots(figsize=(10, 4))
                ax_tcx.fill_between(df_tcx['Time_min'], df_tcx['EE_kcal_h'], color='firebrick', alpha=0.3)
                ax_tcx.plot(df_tcx['Time_min'], df_tcx['EE_kcal_h'], color='firebrick', linewidth=1.5, label='EE Istantaneo (Kcal/h)')
                ax_tcx.set_ylabel('Kcal/h')
                
                ax_tcx_hr = ax_tcx.twinx()
                ax_tcx_hr.plot(df_tcx['Time_min'], df_tcx['HR'], color='blue', alpha=0.5, linewidth=1, label='FC (bpm)')
                ax_tcx_hr.set_ylabel('FC (bpm)')
                
                st.pyplot(fig_tcx)
else:
    st.info("👈 Inizia caricando il file CSV del tuo test CPET nella barra laterale per generare il tuo Digital Twin.")
