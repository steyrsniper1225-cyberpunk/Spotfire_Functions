import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# [Spotfire Configuration]
# input_data: Spotfire에서 "Screening Master" 테이블을 Input Parameter로 연결해야 함
# result_data: Spotfire로 반환될 최종 결과 테이블 (Output Parameter로 설정 필요)
# ==========================================

# 1. 설정값 (하드코딩 대신 변수화)
TIME_WINDOWS = {
    '1W': 7,
    '2W': 14,
    '4W': 28,
    'All': 9999
}
MIN_SAMPLE_CNT = 30 

# ==========================================
# 2. 분석 로직 함수 정의 (내용 동일)
# ==========================================

def calculate_z_score(val, mean, std):
    if std == 0: return 0
    return (val - mean) / std

def logic_01_global_line(df_slice, window_name):
    """Logic 01: Process별 Line 간 비교 (Global View)"""
    results = []
    
    # Base Stats
    base_stats = df_slice.groupby(['PROCESS', 'CODE'])['DEF_QTY'].agg(['mean', 'std']).reset_index()
    base_stats.rename(columns={'mean': 'Global_Mean', 'std': 'Global_Std'}, inplace=True)
    
    # Line Stats
    line_stats = df_slice.groupby(['MODEL', 'PROCESS', 'LINE', 'CODE'])['DEF_QTY'].agg(['mean', 'count']).reset_index()
    line_stats.rename(columns={'mean': 'Line_Mean', 'count': 'Sample_Size'}, inplace=True)
    
    # Merge & Calc
    merged = pd.merge(line_stats, base_stats, on=['PROCESS', 'CODE'], how='left')
    merged['Global_Std'] = merged['Global_Std'].replace(0, 1e-9)
    merged['Z_Score'] = (merged['Line_Mean'] - merged['Global_Mean']) / merged['Global_Std']
    
    for _, row in merged.iterrows():
        if row['Sample_Size'] < MIN_SAMPLE_CNT: continue
        
        results.append({
            'Analysis_Type': 'LINE',
            'MODEL': row['MODEL'],
            'PROCESS': row['PROCESS'],
            'LINE': row['LINE'],
            'MACHINE_ID': row['LINE'], 
            'CODE': row['CODE'],
            'Logic_ID': 'Logic01',
            'Time_Window': window_name,
            'Sample_Size': row['Sample_Size'],
            'Risk_Score': round(row['Z_Score'], 2),
            'Risk_Level': 'High' if row['Z_Score'] > 3.0 else ('Med' if row['Z_Score'] > 2.0 else 'Low'),
            'Detail_Msg': f"Line Avg {row['Line_Mean']:.2f} (Global {row['Global_Mean']:.2f})"
        })
    return results

def logic_02_local_unit(df_slice, window_name):
    """Logic 02: Line 내 Unit 간 비교 (Local View)"""
    results = []
    
    # Local Benchmark (Line Mean)
    line_base_stats = df_slice.groupby(['PROCESS', 'LINE', 'MACHINE', 'CODE'])['DEF_QTY'].agg(['mean', 'std']).reset_index()
    line_base_stats.rename(columns={'mean': 'Local_Mean', 'std': 'Local_Std'}, inplace=True)
    
    # Unit Stats
    unit_stats = df_slice.groupby(['MODEL', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_NO', 'MACHINE_ID', 'CODE'])['DEF_QTY'].agg(['mean', 'count']).reset_index()
    unit_stats.rename(columns={'mean': 'Unit_Mean', 'count': 'Sample_Size'}, inplace=True)
    
    # Merge
    merged = pd.merge(unit_stats, line_base_stats, on=['PROCESS', 'LINE', 'MACHINE', 'CODE'], how='left')
    merged['Local_Std'] = merged['Local_Std'].replace(0, 1e-9)
    merged['Z_Score'] = (merged['Unit_Mean'] - merged['Local_Mean']) / merged['Local_Std']
    
    for _, row in merged.iterrows():
        if row['Sample_Size'] < MIN_SAMPLE_CNT: continue
        if row['Z_Score'] < 1.0: continue 

        results.append({
            'Analysis_Type': 'MACHINE',
            'MODEL': row['MODEL'],
            'PROCESS': row['PROCESS'],
            'LINE': row['LINE'],
            'MACHINE_ID': row['MACHINE_ID'],
            'CODE': row['CODE'],
            'Logic_ID': 'Logic02',
            'Time_Window': window_name,
            'Sample_Size': row['Sample_Size'],
            'Risk_Score': round(row['Z_Score'], 2),
            'Risk_Level': 'High' if row['Z_Score'] > 3.0 else ('Med' if row['Z_Score'] > 2.0 else 'Low'),
            'Detail_Msg': f"Unit Avg {row['Unit_Mean']:.2f} (Local {row['Local_Mean']:.2f})"
        })
    return results

def logic_03_short_term_volatility(df_slice, window_name):
    """Logic 03: 변동성 (Stability)"""
    results = []
    
    # Timestamp to Date
    df_slice = df_slice.copy()
    df_slice['Date'] = df_slice['Timestamp'].dt.date
    
    # Daily Stats
    daily_stats = df_slice.groupby(['MODEL', 'PROCESS', 'LINE', 'MACHINE_ID', 'CODE', 'Date'])['DEF_QTY'].mean().reset_index()
    
    # Volatility Stats
    volatility_stats = daily_stats.groupby(['MODEL', 'PROCESS', 'LINE', 'MACHINE_ID', 'CODE'])['DEF_QTY'].agg(['std', 'mean', 'count']).reset_index()
    volatility_stats.rename(columns={'std': 'Daily_Std', 'mean': 'Period_Mean', 'count': 'Days_Count'}, inplace=True)
    
    for _, row in volatility_stats.iterrows():
        if row['Days_Count'] < 3: continue 
        
        if row['Period_Mean'] == 0: cv = 0
        else: cv = row['Daily_Std'] / row['Period_Mean']
        
        if cv > 1.0: 
             results.append({
                'Analysis_Type': 'MACHINE',
                'MODEL': row['MODEL'],
                'PROCESS': row['PROCESS'],
                'LINE': row['LINE'],
                'MACHINE_ID': row['MACHINE_ID'],
                'CODE': row['CODE'],
                'Logic_ID': 'Logic03',
                'Time_Window': window_name,
                'Sample_Size': int(row['Days_Count']), 
                'Risk_Score': round(cv, 2),
                'Risk_Level': 'High' if cv > 2.0 else 'Med',
                'Detail_Msg': f"Unstable (CV {cv:.1f}, Std {row['Daily_Std']:.2f})"
            })
    return results

def logic_04_interaction(df_slice, window_name):
    """Logic 04: 교호작용 (Interaction)"""
    results = []
    target_df = df_slice[df_slice['PROCESS'].astype(str).str.contains('PHT')]
    if target_df.empty: return []

    sub_df = target_df[['Glass_ID', 'MODEL', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_NO', 'DEF_QTY', 'CODE']].drop_duplicates()
    
    try:
        pivot_machines = sub_df.pivot_table(
            index=['Glass_ID', 'MODEL', 'PROCESS', 'LINE', 'CODE', 'DEF_QTY'], 
            columns='MACHINE', 
            values='MACHINE_NO', 
            aggfunc='first'
        ).reset_index()
    except Exception:
        return [] 

    req_cols = ['VCD', 'SHP'] 
    available_cols = [c for c in req_cols if c in pivot_machines.columns]
    if len(available_cols) < 2: return []
    
    group_cols = ['MODEL', 'PROCESS', 'LINE', 'CODE'] + available_cols
    combo_stats = pivot_machines.groupby(group_cols)['DEF_QTY'].agg(['mean', 'count']).reset_index()
    
    for _, row in combo_stats.iterrows():
        if row['count'] < MIN_SAMPLE_CNT: continue
        
        if row['mean'] > 5.0: 
            vcd_val = row['VCD'] if 'VCD' in row else 'NA'
            shp_val = row['SHP'] if 'SHP' in row else 'NA'
            combo_name = f"{row['LINE']}_VCD{vcd_val}+SHP{shp_val}"
            
            results.append({
                'Analysis_Type': 'INTERACTION',
                'MODEL': row['MODEL'],
                'PROCESS': row['PROCESS'],
                'LINE': row['LINE'],
                'MACHINE_ID': combo_name,
                'CODE': row['CODE'],
                'Logic_ID': 'Logic04',
                'Time_Window': window_name,
                'Sample_Size': row['count'],
                'Risk_Score': round(row['mean'], 2),
                'Risk_Level': 'High',
                'Detail_Msg': f"Interaction Mean {row['mean']:.2f}"
            })
    return results

# ==========================================
# 3. Main Logic (Spotfire Entry Point)
# ==========================================
# Spotfire에서 'input_data' 변수가 들어온다고 가정합니다.

# (1) 데이터 전처리
df = input_data.copy()

# Timestamp 컬럼이 문자열로 올 경우를 대비해 변환
# Spotfire DateTime은 보통 Pandas Timestamp로 자동 변환되지만 안전장치 추가
if df['Timestamp'].dtype == 'object':
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 메모리 최적화 (Spotfire에서 String으로 넘겨준 경우 대비)
cat_cols = ['MODEL', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_NO', 'CODE']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

all_results = []
max_date = df['Timestamp'].max()

# (2) 병렬 로직 실행
for window_name, days in TIME_WINDOWS.items():
    start_date = max_date - timedelta(days=days)
    df_slice = df[df['Timestamp'] >= start_date].copy()
    
    if df_slice.empty: continue
    
    all_results.extend(logic_01_global_line(df_slice, window_name))
    all_results.extend(logic_02_local_unit(df_slice, window_name))
    all_results.extend(logic_03_short_term_volatility(df_slice, window_name))
    all_results.extend(logic_04_interaction(df_slice, window_name))

# (3) 결과 반환 (result_data 할당)
if all_results:
    result_data = pd.DataFrame(all_results)
    
    # 컬럼 순서 정리
    cols_order = ['Analysis_Type', 'MODEL', 'PROCESS', 'LINE', 'MACHINE_ID', 'CODE', 
                  'Logic_ID', 'Time_Window', 'Sample_Size', 'Risk_Score', 'Risk_Level', 'Detail_Msg']
    # 혹시 모를 컬럼 누락 방지
    final_cols = [c for c in cols_order if c in result_data.columns]
    result_data = result_data[final_cols]
else:
    # 결과가 없을 경우 빈 테이블 반환 (에러 방지)
    result_data = pd.DataFrame(columns=['Analysis_Type', 'MODEL', 'Risk_Level', 'Detail_Msg'])
