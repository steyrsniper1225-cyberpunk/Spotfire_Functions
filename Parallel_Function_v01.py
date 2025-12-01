import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# [User Configuration] 실행 환경 선택
# ==========================================
# 'LOCAL' : 집/Antigravity에서 CSV 파일로 실행 시
# 'SPOTFIRE' : 회사 Spotfire Data Function으로 실행 시
EXECUTION_MODE = 'LOCAL' 

# 입력/출력 파일명 (LOCAL 모드용)
INPUT_FILE = "dummy_screening_master_v2.csv"
OUTPUT_FILE = "screening_result_v3.csv"

# 분석 설정
TIME_WINDOWS = {'1W': 7, '2W': 14, 'All': 9999}
MIN_SAMPLE_CNT = 30 
CORR_THRESHOLD = 0.3 # Logic 05 상관계수 기준

# ==========================================
# 1. Logic Functions
# ==========================================

def logic_01_global_line(df_slice, window_name):
    """Logic 01: Process별 Line 간 비교 (Global Z-Score)"""
    results = []
    # Base Stats (Process Level)
    base_stats = df_slice.groupby(['PROCESS', 'CODE'], observed=True)['DEF_QTY'].agg(['mean', 'std']).reset_index()
    base_stats.rename(columns={'mean': 'Global_Mean', 'std': 'Global_Std'}, inplace=True)
    
    # Line Stats
    line_stats = df_slice.groupby(['MODEL', 'PROCESS', 'LINE', 'CODE'], observed=True)['DEF_QTY'].agg(['mean', 'count']).reset_index()
    line_stats.rename(columns={'mean': 'Line_Mean', 'count': 'Sample_Size'}, inplace=True)
    
    # Merge & Z-Score
    merged = pd.merge(line_stats, base_stats, on=['PROCESS', 'CODE'], how='left')
    merged['Global_Std'] = merged['Global_Std'].replace(0, 1e-9)
    merged['Z_Score'] = (merged['Line_Mean'] - merged['Global_Mean']) / merged['Global_Std']
    
    for _, row in merged.iterrows():
        if row['Sample_Size'] < MIN_SAMPLE_CNT: continue
        results.append({
            'Analysis_Type': 'LINE', 'MODEL': row['MODEL'], 'PROCESS': row['PROCESS'], 'LINE': row['LINE'], 'MACHINE_ID': row['LINE'], 
            'CODE': row['CODE'], 'Logic_ID': 'Logic01', 'Time_Window': window_name, 
            'Sample_Size': row['Sample_Size'], 'Risk_Score': round(row['Z_Score'], 2),
            'Risk_Level': 'High' if row['Z_Score'] > 3.0 else ('Med' if row['Z_Score'] > 2.0 else 'Low'),
            'Detail_Msg': f"Line Avg {row['Line_Mean']:.2f} (Global {row['Global_Mean']:.2f})"
        })
    return results

def logic_02_local_unit(df_slice, window_name):
    """Logic 02: Line 내 Unit 간 비교 (Local Z-Score)"""
    results = []
    # Local Benchmark (Line Level)
    line_base_stats = df_slice.groupby(['PROCESS', 'LINE', 'MACHINE', 'CODE'], observed=True)['DEF_QTY'].agg(['mean', 'std']).reset_index()
    line_base_stats.rename(columns={'mean': 'Local_Mean', 'std': 'Local_Std'}, inplace=True)
    
    # Unit Stats
    unit_stats = df_slice.groupby(['MODEL', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_NO', 'MACHINE_ID', 'CODE'], observed=True)['DEF_QTY'].agg(['mean', 'count']).reset_index()
    unit_stats.rename(columns={'mean': 'Unit_Mean', 'count': 'Sample_Size'}, inplace=True)
    
    # Merge & Z-Score
    merged = pd.merge(unit_stats, line_base_stats, on=['PROCESS', 'LINE', 'MACHINE', 'CODE'], how='left')
    merged['Local_Std'] = merged['Local_Std'].replace(0, 1e-9)
    merged['Z_Score'] = (merged['Unit_Mean'] - merged['Local_Mean']) / merged['Local_Std']
    
    for _, row in merged.iterrows():
        if row['Sample_Size'] < MIN_SAMPLE_CNT: continue
        if row['Z_Score'] < 1.0: continue 
        results.append({
            'Analysis_Type': 'MACHINE', 'MODEL': row['MODEL'], 'PROCESS': row['PROCESS'], 'LINE': row['LINE'], 'MACHINE_ID': row['MACHINE_ID'],
            'CODE': row['CODE'], 'Logic_ID': 'Logic02', 'Time_Window': window_name,
            'Sample_Size': row['Sample_Size'], 'Risk_Score': round(row['Z_Score'], 2),
            'Risk_Level': 'High' if row['Z_Score'] > 3.0 else ('Med' if row['Z_Score'] > 2.0 else 'Low'),
            'Detail_Msg': f"Unit Avg {row['Unit_Mean']:.2f} (Local {row['Local_Mean']:.2f})"
        })
    return results

def logic_03_short_term_volatility(df_slice, window_name):
    """Logic 03: 변동성 (Stability Check)"""
    results = []
    df_temp = df_slice.copy()
    df_temp['Date'] = df_temp['Timestamp'].dt.date
    
    daily_stats = df_temp.groupby(['MODEL', 'PROCESS', 'LINE', 'MACHINE_ID', 'CODE', 'Date'], observed=True)['DEF_QTY'].mean().reset_index()
    volatility_stats = daily_stats.groupby(['MODEL', 'PROCESS', 'LINE', 'MACHINE_ID', 'CODE'], observed=True)['DEF_QTY'].agg(['std', 'mean', 'count']).reset_index()
    
    for _, row in volatility_stats.iterrows():
        if row['count'] < 3: continue 
        if row['mean'] == 0: cv = 0
        else: cv = row['std'] / row['mean']
        
        if cv > 1.0: 
             results.append({
                'Analysis_Type': 'MACHINE', 'MODEL': row['MODEL'], 'PROCESS': row['PROCESS'], 'LINE': row['LINE'], 'MACHINE_ID': row['MACHINE_ID'],
                'CODE': row['CODE'], 'Logic_ID': 'Logic03', 'Time_Window': window_name, 'Sample_Size': int(row['count']), 
                'Risk_Score': round(cv, 2), 'Risk_Level': 'High' if cv > 2.0 else 'Med',
                'Detail_Msg': f"Unstable (CV {cv:.1f}, Std {row['std']:.2f})"
            })
    return results

def logic_04_interaction_zscore(df_slice, window_name):
    """
    [Logic 04 개선] 교호작용 Z-Score 평가
    - 기준: 절대값(Mean > 5) -> 상대값(Line 평균 대비 Z-Score > 3.0)
    """
    results = []
    target_df = df_slice[df_slice['PROCESS'].astype(str).str.contains('PHT')].copy()
    if target_df.empty: return []

    # 1. Line Level Basic Stats (Benchmark)
    # 교호작용 조합도 결국 그 Line의 퍼포먼스 안에 있으므로, Line 전체의 평균/편차를 기준으로 함
    line_stats = target_df.groupby(['PROCESS', 'LINE', 'CODE'], observed=True)['DEF_QTY'].agg(['mean', 'std']).reset_index()
    line_stats.rename(columns={'mean': 'Line_Mean', 'std': 'Line_Std'}, inplace=True)

    # 2. Combination Stats Calculation (Optimized Merge)
    mini_df = target_df[['Glass_ID', 'MODEL', 'PROCESS', 'LINE', 'CODE', 'MACHINE', 'MACHINE_NO', 'DEF_QTY']]
    vcd_df = mini_df[mini_df['MACHINE'] == 'VCD'][['Glass_ID', 'MACHINE_NO']].rename(columns={'MACHINE_NO': 'VCD_NO'})
    shp_df = mini_df[mini_df['MACHINE'] == 'SHP'][['Glass_ID', 'MACHINE_NO']].rename(columns={'MACHINE_NO': 'SHP_NO'})
    
    if vcd_df.empty or shp_df.empty: return []

    combo_df = pd.merge(vcd_df, shp_df, on='Glass_ID', how='inner')
    meta_df = mini_df[['Glass_ID', 'MODEL', 'PROCESS', 'LINE', 'CODE', 'DEF_QTY']].drop_duplicates()
    final_df = pd.merge(combo_df, meta_df, on='Glass_ID', how='inner')

    combo_stats = final_df.groupby(['MODEL', 'PROCESS', 'LINE', 'CODE', 'VCD_NO', 'SHP_NO'], observed=True)['DEF_QTY'].agg(['mean', 'count']).reset_index()

    # 3. Compare Combo vs Line
    merged = pd.merge(combo_stats, line_stats, on=['PROCESS', 'LINE', 'CODE'], how='left')
    merged['Line_Std'] = merged['Line_Std'].replace(0, 1e-9)
    merged['Z_Score'] = (merged['mean'] - merged['Line_Mean']) / merged['Line_Std']

    for _, row in merged.iterrows():
        if row['count'] < MIN_SAMPLE_CNT: continue
        
        # Z-Score 기준 적용 (3.0 이상 High Risk)
        if row['Z_Score'] > 2.0: 
            combo_name = f"{row['LINE']}_VCD{row['VCD_NO']}+SHP{row['SHP_NO']}"
            results.append({
                'Analysis_Type': 'INTERACTION', 'MODEL': row['MODEL'], 'PROCESS': row['PROCESS'], 'LINE': row['LINE'], 'MACHINE_ID': combo_name,
                'CODE': row['CODE'], 'Logic_ID': 'Logic04', 'Time_Window': window_name,
                'Sample_Size': row['count'], 'Risk_Score': round(row['Z_Score'], 2), 
                'Risk_Level': 'High' if row['Z_Score'] > 3.0 else 'Med',
                'Detail_Msg': f"Inter. Mean {row['mean']:.2f} (Line {row['Line_Mean']:.2f})"
            })
    return results

def logic_05_slot_correlation(df_slice, window_name):
    """
    [New] Logic 05: Slot Trend Analysis
    - 대상: CST_SLOT
    - 기준: Slot 번호(MACHINE_NO)와 DEF_QTY 간의 상관계수 > 0.3
    """
    results = []
    # CST_SLOT 데이터만 필터링
    slot_df = df_slice[df_slice['MACHINE'] == 'CST_SLOT'].copy()
    if slot_df.empty: return []
    
    # MACHINE_NO를 숫자로 변환
    slot_df['Slot_Int'] = pd.to_numeric(slot_df['MACHINE_NO'], errors='coerce')
    slot_df = slot_df.dropna(subset=['Slot_Int'])

    # GroupBy: Line 단위로 Slot Trend를 봄 (개별 CST_SLOT Unit이 아니라, Line 전체의 Slot 경향성)
    # OVN02 사례처럼 "OVN02 라인의 CST_SLOT" 전체 경향을 파악
    grouped = slot_df.groupby(['MODEL', 'PROCESS', 'LINE', 'CODE'], observed=True)

    for name, group in grouped:
        if len(group) < MIN_SAMPLE_CNT * 2: continue # 상관분석은 샘플이 좀 더 필요함
        
        # 불량이 모두 0이거나 값이 같으면 상관계수 계산 불가 (std=0)
        if group['DEF_QTY'].std() == 0 or group['Slot_Int'].std() == 0:
            continue

        # 상관계수 계산 (Pearson)
        corr = group['Slot_Int'].corr(group['DEF_QTY'])
        
        if corr > CORR_THRESHOLD: # 양의 상관관계 (번호 클수록 불량 많음)
            model, process, line, code = name
            results.append({
                'Analysis_Type': 'MACHINE', 'MODEL': model, 'PROCESS': process, 'LINE': line, 
                'MACHINE_ID': f"{line}_CST_SLOT_TREND", # 가상의 ID 부여
                'CODE': code, 'Logic_ID': 'Logic05', 'Time_Window': window_name,
                'Sample_Size': len(group), 'Risk_Score': round(corr, 2),
                'Risk_Level': 'High' if corr > 0.5 else 'Med',
                'Detail_Msg': f"Slot Trend Detected (Corr {corr:.2f})"
            })
    return results

# ==========================================
# 2. Main Execution Controller
# ==========================================
def run_screening(input_df=None):
    
    # [환경별 데이터 로딩]
    if EXECUTION_MODE == 'LOCAL':
        print(f"[Mode: LOCAL] Loading CSV: {INPUT_FILE}...")
        # 메모리 최적화 로딩
        dtype_map = {
            'MODEL': 'category', 'PROCESS': 'category', 'LINE': 'category',
            'MACHINE': 'category', 'MACHINE_NO': 'category', 'CODE': 'category', 'MACHINE_ID': 'category'
        }
        use_cols = ['Glass_ID', 'MODEL', 'Timestamp', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_NO', 'MACHINE_ID', 'CODE', 'DEF_QTY']
        df = pd.read_csv(INPUT_FILE, dtype=dtype_map, usecols=use_cols, parse_dates=['Timestamp'])
    else:
        # SPOTFIRE 모드
        # Spotfire에서는 input_df가 파라미터로 넘어옴
        print("[Mode: SPOTFIRE] Processing Input Data Table...")
        df = input_df.copy()
        if df['Timestamp'].dtype == 'object':
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        # Category 변환 (안전장치)
        for c in ['MODEL', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_NO', 'CODE']:
            if c in df.columns: df[c] = df[c].astype('category')

    # [분석 실행]
    all_results = []
    max_date = df['Timestamp'].max()
    
    print(f"Data Loaded: {len(df)} rows. Running Parallel Logics...")

    for window_name, days in TIME_WINDOWS.items():
        start_date = max_date - timedelta(days=days)
        df_slice = df[df['Timestamp'] >= start_date].copy()
        
        if df_slice.empty: continue
        
        # Parallel Logic Execution
        all_results.extend(logic_01_global_line(df_slice, window_name))
        all_results.extend(logic_02_local_unit(df_slice, window_name))
        all_results.extend(logic_03_short_term_volatility(df_slice, window_name))
        all_results.extend(logic_04_interaction_zscore(df_slice, window_name)) # Z-Score Ver.
        all_results.extend(logic_05_slot_correlation(df_slice, window_name))   # New Logic

    # [결과 반환/저장]
    if all_results:
        result_df = pd.DataFrame(all_results)
        cols_order = ['Analysis_Type', 'MODEL', 'PROCESS', 'LINE', 'MACHINE_ID', 'CODE', 
                      'Logic_ID', 'Time_Window', 'Sample_Size', 'Risk_Score', 'Risk_Level', 'Detail_Msg']
        # 컬럼 순서 보장 (있는 것만)
        final_cols = [c for c in cols_order if c in result_df.columns]
        result_df = result_df[final_cols]
        
        if EXECUTION_MODE == 'LOCAL':
            result_df.to_csv(OUTPUT_FILE, index=False)
            print(f"Analysis Complete. Saved to {OUTPUT_FILE}")
            print(result_df.head())
            return None
        else:
            return result_df
    else:
        print("No risks found.")
        return pd.DataFrame(columns=['Analysis_Type', 'Risk_Level', 'Detail_Msg'])

# ==========================================
# 3. Entry Point
# ==========================================
if __name__ == "__main__":
    if EXECUTION_MODE == 'LOCAL':
        run_screening()
    else:
        # Spotfire 내장 Python에서는 이 블록이 실행되지 않고,
        # 사용자가 직접 run_screening(input_data)를 호출하거나,
        # 아래와 같이 전역 변수 매핑을 통해 실행되도록 구성함.
        pass

# [Spotfire Data Function Script용 Wrapper]
# Spotfire에 붙여넣을 때는 아래 주석을 풀고 맨 아래에 추가하세요.
# result_data = run_screening(input_data)
