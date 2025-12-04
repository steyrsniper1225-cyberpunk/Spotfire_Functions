import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# [User Configuration] 실행 환경 및 파라미터
# ==========================================
# 'LOCAL' : 로컬 테스트용 / 'SPOTFIRE' : 실제 배포용
EXECUTION_MODE = 'LOCAL' 

INPUT_FILE = "dummy_screening_master_v2.csv"
OUTPUT_FILE = "screening_result_vfinal.csv"

# 분석 설정
# '1W': 7일, '2W': 14일 등 분석 기간 설정
TIME_WINDOWS = {'1W': 7, '2W': 14}
MIN_SAMPLE_CNT = 30    # 최소 표본 수 (Z-Score 신뢰성 확보용)
CORR_THRESHOLD = 0.3   # Logic 05 상관계수 기준

# ==========================================
# 1. Logic Functions
# ==========================================

def logic_01_global_line(df_slice, window_name):
    """
    Logic 01: Process별 Line 간 비교 (Global Z-Score)
    [수정 사항] Machine 중복으로 인한 통계 왜곡 방지를 위해 Pre-aggregation 수행
    """
    results = []
    
    # [Step 0] Pre-aggregation (전처리)
    # 목적: 1 Row = 1 Glass 보장 (Machine 정보로 인한 중복 제거)
    # 가정: 동일 Glass 내 중복 행은 동일한 DEF_QTY를 가짐 (단순 복제)
    df_unique_glass = df_slice.groupby(
        ['MODEL', 'PROCESS', 'LINE', 'CODE', 'GLASS_ID'], 
        observed=True
    )['DEF_QTY'].mean().reset_index()
    
    # [Step 1] Base Stats (Global Level)
    # 중복이 제거된 df_unique_glass를 사용하여 정확한 Mean, Std 산출
    base_stats = df_unique_glass.groupby(['PROCESS', 'CODE'], observed=True)['DEF_QTY'].agg(['mean', 'std']).reset_index()
    base_stats.rename(columns={'mean': 'Global_Mean', 'std': 'Global_Std'}, inplace=True)
    
    # [Step 2] Line Stats (Line Level)
    # 이제 1 Row = 1 Glass이므로 count가 곧 정확한 Sample Size임
    line_stats = df_unique_glass.groupby(['MODEL', 'PROCESS', 'LINE', 'CODE'], observed=True)['DEF_QTY'].agg(['mean', 'count']).reset_index()
    line_stats.rename(columns={'mean': 'Line_Mean', 'count': 'Sample_Size'}, inplace=True)
    
    # [Step 3] Merge & Z-Score
    merged = pd.merge(line_stats, base_stats, on=['PROCESS', 'CODE'], how='left')
    merged['Global_Std'] = merged['Global_Std'].replace(0, 1e-9)
    merged['Z_Score'] = (merged['Line_Mean'] - merged['Global_Mean']) / merged['Global_Std']
    
    for _, row in merged.iterrows():
        if row['Sample_Size'] < MIN_SAMPLE_CNT: continue
        
        results.append({
            'Analysis_Type': 'LINE', 
            'MODEL': row['MODEL'], 'PROCESS': row['PROCESS'], 'LINE': row['LINE'], 'MACHINE_ID': row['LINE'], 
            'CODE': row['CODE'], 'Logic_ID': 'Logic01', 'Time_Window': window_name, 
            'Sample_Size': int(row['Sample_Size']), 
            'Risk_Score': round(row['Z_Score'], 2),
            'Risk_Level': 'High' if row['Z_Score'] > 3.0 else ('Med' if row['Z_Score'] > 2.0 else 'Low'),
            'Detail_Msg': f"Line DPU {row['Line_Mean']:.2f} (Global {row['Global_Mean']:.2f})"
        })
    return results

def logic_02_local_unit(df_slice, window_name):
    """
    Logic 02: Line 내 Unit 간 비교 (Local Z-Score)
    [수정 사항] Unit 단위 분석 시에도 Glass 중복 제거 수행
    """
    results = []
    
    # [Step 0] Pre-aggregation (Unit Level)
    # Unit(MACHINE_ID) 별로도 Glass가 중복될 수 있으므로(드물지만), 안전하게 유니크화
    df_unique_unit = df_slice.groupby(
        ['MODEL', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_NO', 'MACHINE_ID', 'CODE', 'GLASS_ID'], 
        observed=True
    )['DEF_QTY'].mean().reset_index()

    # [Step 1] Local Benchmark (Line Level)
    # 해당 라인의 해당 공정(MACHINE Type) 평균 및 편차
    line_base_stats = df_unique_unit.groupby(['PROCESS', 'LINE', 'MACHINE', 'CODE'], observed=True)['DEF_QTY'].agg(['mean', 'std']).reset_index()
    line_base_stats.rename(columns={'mean': 'Local_Mean', 'std': 'Local_Std'}, inplace=True)
    
    # [Step 2] Unit Stats
    unit_stats = df_unique_unit.groupby(['MODEL', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_NO', 'MACHINE_ID', 'CODE'], observed=True)['DEF_QTY'].agg(['mean', 'count']).reset_index()
    unit_stats.rename(columns={'mean': 'Unit_Mean', 'count': 'Sample_Size'}, inplace=True)
    
    # [Step 3] Merge & Z-Score
    merged = pd.merge(unit_stats, line_base_stats, on=['PROCESS', 'LINE', 'MACHINE', 'CODE'], how='left')
    merged['Local_Std'] = merged['Local_Std'].replace(0, 1e-9)
    merged['Z_Score'] = (merged['Unit_Mean'] - merged['Local_Mean']) / merged['Local_Std']
    
    for _, row in merged.iterrows():
        if row['Sample_Size'] < MIN_SAMPLE_CNT: continue
        if row['Z_Score'] < 1.0: continue # 의미 없는 하위 Unit 제외
        
        results.append({
            'Analysis_Type': 'MACHINE', 
            'MODEL': row['MODEL'], 'PROCESS': row['PROCESS'], 'LINE': row['LINE'], 'MACHINE_ID': row['MACHINE_ID'],
            'CODE': row['CODE'], 'Logic_ID': 'Logic02', 'Time_Window': window_name,
            'Sample_Size': int(row['Sample_Size']), 
            'Risk_Score': round(row['Z_Score'], 2),
            'Risk_Level': 'High' if row['Z_Score'] > 3.0 else ('Med' if row['Z_Score'] > 2.0 else 'Low'),
            'Detail_Msg': f"Unit DPU {row['Unit_Mean']:.2f} (Local {row['Local_Mean']:.2f})"
        })
    return results

def logic_03_short_term_volatility(df_slice, window_name):
    """Logic 03: 변동성 (Stability Check)"""
    results = []
    
    # [Step 0] 날짜 기준 Pre-aggregation
    # 일별 통계를 낼 때도 Glass 중복 제거 필요
    df_unique = df_slice.groupby(['MODEL', 'PROCESS', 'LINE', 'MACHINE_ID', 'CODE', 'GLASS_ID', 'Timestamp'], observed=True)['DEF_QTY'].mean().reset_index()
    df_unique['Date'] = df_unique['Timestamp'].dt.date
    
    # 일별 평균 DPU 산출
    daily_stats = df_unique.groupby(['MODEL', 'PROCESS', 'LINE', 'MACHINE_ID', 'CODE', 'Date'], observed=True)['DEF_QTY'].mean().reset_index()
    
    # 변동성(CV) 산출
    volatility_stats = daily_stats.groupby(['MODEL', 'PROCESS', 'LINE', 'MACHINE_ID', 'CODE'], observed=True)['DEF_QTY'].agg(['std', 'mean', 'count']).reset_index()
    
    for _, row in volatility_stats.iterrows():
        if row['count'] < 3: continue  # 최소 3일치 데이터 필요
        if row['mean'] == 0: cv = 0
        else: cv = row['std'] / row['mean']
        
        if cv > 1.0: # 변동계수가 1.0을 넘으면 불안정
             results.append({
                'Analysis_Type': 'MACHINE', 'MODEL': row['MODEL'], 'PROCESS': row['PROCESS'], 'LINE': row['LINE'], 'MACHINE_ID': row['MACHINE_ID'],
                'CODE': row['CODE'], 'Logic_ID': 'Logic03', 'Time_Window': window_name, 
                'Sample_Size': int(row['count']), # 여기서는 일수(Days)를 의미
                'Risk_Score': round(cv, 2), 
                'Risk_Level': 'High' if cv > 2.0 else 'Med',
                'Detail_Msg': f"Unstable (CV {cv:.1f}, Std {row['std']:.2f})"
            })
    return results

def logic_04_interaction_zscore(df_slice, window_name):
    """Logic 04: 교호작용 (VCD+SHP 조합) Z-Score 평가"""
    results = []
    # PHT 공정만 대상
    target_df = df_slice[df_slice['PROCESS'].astype(str).str.contains('PHT')].copy()
    if target_df.empty: return []

    # [Step 1] Line Level Basic Stats (Benchmark)
    # 라인 전체의 DPU 분포 계산 (중복 제거 후)
    clean_line_df = target_df.groupby(['PROCESS', 'LINE', 'CODE', 'GLASS_ID'], observed=True)['DEF_QTY'].mean().reset_index()
    line_stats = clean_line_df.groupby(['PROCESS', 'LINE', 'CODE'], observed=True)['DEF_QTY'].agg(['mean', 'std']).reset_index()
    line_stats.rename(columns={'mean': 'Line_Mean', 'std': 'Line_Std'}, inplace=True)

    # [Step 2] Combination Data Construction
    # VCD와 SHP 정보를 결합 (Glass ID 기준)
    mini_df = target_df[['Glass_ID', 'MODEL', 'PROCESS', 'LINE', 'CODE', 'MACHINE', 'MACHINE_NO', 'DEF_QTY']]
    
    vcd_df = mini_df[mini_df['MACHINE'] == 'VCD'][['Glass_ID', 'MACHINE_NO']].rename(columns={'MACHINE_NO': 'VCD_NO'})
    shp_df = mini_df[mini_df['MACHINE'] == 'SHP'][['Glass_ID', 'MACHINE_NO']].rename(columns={'MACHINE_NO': 'SHP_NO'})
    
    if vcd_df.empty or shp_df.empty: return []

    # Inner Join으로 VCD, SHP를 모두 거친 Glass만 추출
    combo_df = pd.merge(vcd_df, shp_df, on='Glass_ID', how='inner')
    
    # 메타 정보 및 DEF_QTY 결합 (Glass 중복 제거된 메타 사용)
    meta_df = clean_line_df[['Glass_ID', 'MODEL', 'PROCESS', 'LINE', 'CODE', 'DEF_QTY']]
    final_df = pd.merge(combo_df, meta_df, on='Glass_ID', how='inner')

    # 조합별 통계 산출
    combo_stats = final_df.groupby(['MODEL', 'PROCESS', 'LINE', 'CODE', 'VCD_NO', 'SHP_NO'], observed=True)['DEF_QTY'].agg(['mean', 'count']).reset_index()

    # [Step 3] Compare Combo vs Line (Z-Score)
    merged = pd.merge(combo_stats, line_stats, on=['PROCESS', 'LINE', 'CODE'], how='left')
    merged['Line_Std'] = merged['Line_Std'].replace(0, 1e-9)
    merged['Z_Score'] = (merged['mean'] - merged['Line_Mean']) / merged['Line_Std']

    for _, row in merged.iterrows():
        if row['count'] < MIN_SAMPLE_CNT: continue
        
        if row['Z_Score'] > 2.0: 
            combo_name = f"{row['LINE']}_VCD{row['VCD_NO']}+SHP{row['SHP_NO']}"
            results.append({
                'Analysis_Type': 'INTERACTION', 'MODEL': row['MODEL'], 'PROCESS': row['PROCESS'], 'LINE': row['LINE'], 'MACHINE_ID': combo_name,
                'CODE': row['CODE'], 'Logic_ID': 'Logic04', 'Time_Window': window_name,
                'Sample_Size': int(row['count']), 
                'Risk_Score': round(row['Z_Score'], 2), 
                'Risk_Level': 'High' if row['Z_Score'] > 3.0 else 'Med',
                'Detail_Msg': f"Inter. DPU {row['mean']:.2f} (Line {row['Line_Mean']:.2f})"
            })
    return results

def logic_05_slot_correlation(df_slice, window_name):
    """Logic 05: Slot 번호와 Defect 간의 상관관계 분석"""
    results = []
    # CST_SLOT 데이터만 필터링
    slot_df = df_slice[df_slice['MACHINE'] == 'CST_SLOT'].copy()
    if slot_df.empty: return []
    
    # MACHINE_NO를 숫자로 변환 (분석용)
    slot_df['Slot_Int'] = pd.to_numeric(slot_df['MACHINE_NO'], errors='coerce')
    slot_df = slot_df.dropna(subset=['Slot_Int'])
    
    # 중복 제거 (Glass 단위)
    df_unique = slot_df.groupby(['MODEL', 'PROCESS', 'LINE', 'CODE', 'GLASS_ID', 'Slot_Int'], observed=True)['DEF_QTY'].mean().reset_index()

    # Line 단위 Grouping
    grouped = df_unique.groupby(['MODEL', 'PROCESS', 'LINE', 'CODE'], observed=True)

    for name, group in grouped:
        if len(group) < MIN_SAMPLE_CNT * 2: continue 
        
        # 표준편차가 0이면 상관계수 계산 불가
        if group['DEF_QTY'].std() == 0 or group['Slot_Int'].std() == 0:
            continue

        # 상관계수 (Pearson)
        corr = group['Slot_Int'].corr(group['DEF_QTY'])
        
        if corr > CORR_THRESHOLD: # 양의 상관관계 (번호가 클수록 불량 높음)
            model, process, line, code = name
            results.append({
                'Analysis_Type': 'MACHINE', 'MODEL': model, 'PROCESS': process, 'LINE': line, 
                'MACHINE_ID': f"{line}_CST_SLOT_TREND", 
                'CODE': code, 'Logic_ID': 'Logic05', 'Time_Window': window_name,
                'Sample_Size': len(group), 
                'Risk_Score': round(corr, 2),
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
        dtype_map = {
            'MODEL': 'category', 'PROCESS': 'category', 'LINE': 'category',
            'MACHINE': 'category', 'MACHINE_NO': 'category', 'CODE': 'category', 'MACHINE_ID': 'category'
        }
        # 필요한 컬럼만 로딩하여 메모리 절약
        use_cols = ['Glass_ID', 'MODEL', 'Timestamp', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_NO', 'MACHINE_ID', 'CODE', 'DEF_QTY']
        df = pd.read_csv(INPUT_FILE, dtype=dtype_map, usecols=use_cols, parse_dates=['Timestamp'])
    else:
        # SPOTFIRE 모드
        print("[Mode: SPOTFIRE] Processing Input Data Table...")
        df = input_df.copy()
        if df['Timestamp'].dtype == 'object':
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Category 변환 (속도 최적화)
        for c in ['MODEL', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_NO', 'CODE']:
            if c in df.columns: df[c] = df[c].astype('category')

    # [분석 실행]
    all_results = []
    
    # [수정] CrossTable과 일치시키기 위해 시간(Time) 제거 후 날짜(Date)만 사용
    # 가장 최근 데이터의 날짜 00:00:00 기준
    max_date = df['Timestamp'].max().floor('D')
    
    print(f"Data Loaded: {len(df)} rows. Max Date: {max_date}")

    for window_name, days in TIME_WINDOWS.items():
        # 시작일 계산 (날짜 기준)
        start_date = max_date - timedelta(days=days)
        
        # 해당 날짜 00:00:00 포함 이후 모든 데이터
        df_slice = df[df['Timestamp'] >= start_date].copy()
        
        if df_slice.empty: continue
        
        # 각 로직 실행
        all_results.extend(logic_01_global_line(df_slice, window_name))
        all_results.extend(logic_02_local_unit(df_slice, window_name))
        all_results.extend(logic_03_short_term_volatility(df_slice, window_name))
        all_results.extend(logic_04_interaction_zscore(df_slice, window_name)) 
        all_results.extend(logic_05_slot_correlation(df_slice, window_name)) 

    # [결과 반환]
    if all_results:
        result_df = pd.DataFrame(all_results)
        # 컬럼 순서 정렬
        cols_order = ['Analysis_Type', 'MODEL', 'PROCESS', 'LINE', 'MACHINE_ID', 'CODE', 
                      'Logic_ID', 'Time_Window', 'Sample_Size', 'Risk_Score', 'Risk_Level', 'Detail_Msg']
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
        # 결과 없을 시 빈 테이블 리턴 (에러 방지)
        return pd.DataFrame(columns=['Analysis_Type', 'Risk_Level', 'Detail_Msg'])

# ==========================================
# 3. Entry Point (Spotfire Wrapper)
# ==========================================
if __name__ == "__main__":
    if EXECUTION_MODE == 'LOCAL':
        run_screening()
    else:
        # Spotfire 환경에서는 이 부분은 실행되지 않음
        pass

# [Spotfire Input/Output Parameters]
# Input: input_data (Table)
# Output: result_data (Table)

# Spotfire Script 창에는 아래 한 줄이 활성화되어야 합니다.
# result_data = run_screening(input_data)