import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# [User Configuration] 실행 환경 및 파라미터
# ==========================================
# 'LOCAL' : 로컬 테스트용 / 'SPOTFIRE' : 실제 배포용
EXECUTION_MODE = 'LOCAL' 

INPUT_FILE = "dummy_screening_master_v2.xlsx"
OUTPUT_FILE = "screening_result_vfinal.xlsx"

# 분석 설정
# '1W': 7일, '2W': 14일 등 분석 기간 설정
WINDOWS = {
    '01W': 7,
    '02W': 14,
    '03W': 21,
    '04W': 28,
    '05W': 35,
}
MIN_DATACOUNT_CNT = 30

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
    # 가정: 동일 Glass 내 중복 행은 동일한 DEFECT_QTY를 가짐 (단순 복제)
    df_unique_glass = df_slice.groupby(
        ['MODEL', 'PROCESS', 'LINE', 'CODE', 'GLASS_ID'], 
        observed=True
    )['DEFECT_QTY'].mean().reset_index()
    
    # [Step 1] Base Stats (Global Level)
    # 중복이 제거된 df_unique_glass를 사용하여 정확한 Mean, Std 산출
    base_stats = df_unique_glass.groupby(['MODEL', 'PROCESS', 'CODE'], observed=True)['DEFECT_QTY'].agg(['mean', 'std']).reset_index()
    base_stats.rename(columns={'mean': 'Global_Mean', 'std': 'Global_Std'}, inplace=True)
    
    # [Step 2] Line Stats (Line Level)
    # 이제 1 Row = 1 Glass이므로 count가 곧 정확한 Sample Size임
    line_stats = df_unique_glass.groupby(['MODEL', 'PROCESS', 'LINE', 'CODE'], observed=True)['DEFECT_QTY'].agg(['mean', 'count']).reset_index()
    line_stats.rename(columns={'mean': 'Line_Mean', 'count': 'DATACOUNT'}, inplace=True)
    
    # [Step 3] Merge & Z-Score
    merged = pd.merge(line_stats, base_stats, on=['MODEL', 'PROCESS', 'CODE'], how='left')
    merged['Global_Std'] = merged['Global_Std'].replace(0, 1e-9)
    merged['Z_Score'] = (merged['Line_Mean'] - merged['Global_Mean']) / merged['Global_Std']
    
    for _, row in merged.iterrows():
        if row['DATACOUNT'] < MIN_DATACOUNT_CNT: continue
        if (row['Line_Mean'] - row['Global_Mean']) < 0.5: continue
        
        results.append({
            'TYPE': 'LINE', 
            'MODEL': row['MODEL'],
            'PROCESS': row['PROCESS'],
            'LINE': row['LINE'],
            'MACHINE_ID': row['LINE'],
            'CODE': row['CODE'],
            'LOGIC': 'L01',
            'WINDOW': window_name, 
            'DATACOUNT': int(row['DATACOUNT']), 
            'INDEX': round(row['Z_Score'], 2),
            'LEVEL': 'High' if row['Z_Score'] > 2.0 else ('Med' if row['Z_Score'] > 1.0 else 'Low'),
            'NOTE': f"Line DPU {row['Line_Mean']:.2f} (Global {row['Global_Mean']:.2f})"
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
    )['DEFECT_QTY'].mean().reset_index()

    # [Step 1] Local Benchmark (Line Level)
    # 해당 라인의 해당 공정(MACHINE Type) 평균 및 편차
    line_base_stats = df_unique_unit.groupby(['MODEL', 'PROCESS', 'LINE', 'MACHINE', 'CODE'], observed=True)['DEFECT_QTY'].agg(['mean', 'std']).reset_index()
    line_base_stats.rename(columns={'mean': 'Local_Mean', 'std': 'Local_Std'}, inplace=True)
    
    # [Step 2] Unit Stats
    unit_stats = df_unique_unit.groupby(['MODEL', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_NO', 'MACHINE_ID', 'CODE'], observed=True)['DEFECT_QTY'].agg(['mean', 'count']).reset_index()
    unit_stats.rename(columns={'mean': 'Unit_Mean', 'count': 'DATACOUNT'}, inplace=True)
    
    # [Step 3] Merge & Z-Score
    merged = pd.merge(unit_stats, line_base_stats, on=['MODEL', 'PROCESS', 'LINE', 'MACHINE', 'CODE'], how='left')
    merged['Local_Std'] = merged['Local_Std'].replace(0, 1e-9)
    merged['Z_Score'] = (merged['Unit_Mean'] - merged['Local_Mean']) / merged['Local_Std']
    
    for _, row in merged.iterrows():
        if row['DATACOUNT'] < MIN_DATACOUNT_CNT: continue
        if (row['Unit_Mean'] - row['Local_Mean']) < 1.0: continue
            # RPR은 차이 값, I-PTN은 Z_Score로 필터링
        
        results.append({
            'TYPE': 'MACHINE',
            'MODEL': row['MODEL'],
            'PROCESS': row['PROCESS'],
            'LINE': row['LINE'],
            'MACHINE_ID': row['MACHINE_ID'],
            'CODE': row['CODE'],
            'LOGIC': 'L02',
            'WINDOW': window_name,
            'DATACOUNT': int(row['DATACOUNT']), 
            'INDEX': round(row['Z_Score'], 2),
            'LEVEL': 'High' if row['Z_Score'] > 2.0 else ('Med' if row['Z_Score'] > 1.0 else 'Low'),
            'NOTE': f"Unit DPU {row['Unit_Mean']:.2f} (Local {row['Local_Mean']:.2f})"
        })
    return results

def logic_03_short_term_volatility(df_slice, window_name):
    """Logic 03: 변동성 (Max Z-Score per Group)"""
    results = []
    
    key_cols = ['MODEL', 'PROCESS', 'LINE', 'MACHINE_ID', 'CODE']
    
    # [Step 0] 날짜 기준 Pre-aggregation
    df_unique = df_slice.groupby(key_cols + ['GLASS_ID', 'TIMESTAMP'], observed=True)['DEFECT_QTY'].mean().reset_index()
    df_unique['Date'] = df_unique['TIMESTAMP'].dt.date
    
    # [Step 1] 일별 평균 DPU (Daily Stats)
    daily_stats = df_unique.groupby(key_cols + ['Date'], observed=True)['DEFECT_QTY'].agg(["mean", "count"]).reset_index()
    daily_stats = daily_stats[daily_stats["count"] >= 30].copy()
    daily_stats.rename(columns = {"mean": "DEFECT_QTY"}, inplace = True)
    daily_stats.drop(columns = ["count"], inplace = True)
    
    # [Step 2] Window 기간 전체 통계 (Window Stats)
    volatility_stats = daily_stats.groupby(key_cols, observed=True)['DEFECT_QTY'].agg(['std', 'mean', 'count']).reset_index()
    
    # [Step 3] 병합 및 최소 데이터 수 필터링
    merged_df = pd.merge(daily_stats, volatility_stats, on=key_cols, how='inner')
    merged_df = merged_df[merged_df['count'] >= 3].copy()
    
    # [Step 4] Z-Score 계산 (std > 0 인 경우만)
    merged_df = merged_df[merged_df['std'] > 0].copy()
    merged_df['z_score'] = (merged_df['DEFECT_QTY'] - merged_df['mean']) / merged_df['std']
    
    # [Step 5] Z-Score > 1.0 필터링 후, 그룹별 최대값 1개만 선정
    outliers = merged_df[merged_df['z_score'] > 1.0].copy()
    
    if not outliers.empty:
        # Z-score 내림차순 정렬 -> 그룹별 중복 제거(첫번째=최대값 유지)
        max_outliers = outliers.sort_values(by='z_score', ascending=False).drop_duplicates(
            subset=key_cols, 
            keep='first'
        )
        
        for _, row in max_outliers.iterrows():
            z_score = row['z_score']
            
            if z_score > 3.0: level = 'High'
            elif z_score > 2.0: level = 'Med'
            else: level = 'Low'

            results.append({
                'TYPE': 'MACHINE',
                'MODEL': row['MODEL'],
                'PROCESS': row['PROCESS'],
                'LINE': row['LINE'],
                'MACHINE_ID': row['MACHINE_ID'],
                'CODE': row['CODE'],
                'LOGIC': 'L03',
                'WINDOW': window_name,
                'DATACOUNT': int(row['count']),
                'INDEX': round(z_score, 2),
                'LEVEL': level,
                'NOTE': f"Max Outlier: {row['Date']} (Z {z_score:.2f}, Daily {row['DEFECT_QTY']:.2f})"
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
    clean_line_df = target_df.groupby(['MODEL', 'PROCESS', 'LINE', 'CODE', 'GLASS_ID'], observed=True)['DEFECT_QTY'].mean().reset_index()
    
    line_stats = clean_line_df.groupby(['MODEL', 'PROCESS', 'LINE', 'CODE'], observed=True)['DEFECT_QTY'].agg(['mean', 'std']).reset_index()
    line_stats.rename(columns={'mean': 'Line_Mean', 'std': 'Line_Std'}, inplace=True)

    # [Step 2] Combination Data Construction
    # VCD와 SHP 정보를 결합 (Glass ID 기준)
    mini_df = target_df[['Glass_ID', 'MODEL', 'PROCESS', 'LINE', 'CODE', 'MACHINE', 'MACHINE_NO', 'DEFECT_QTY']]
    
    group_cols = ['Glass_ID', 'MODEL', 'PROCESS', 'LINE', 'CODE']
    
    vcd_df = mini_df[mini_df['MACHINE'] == 'VCD'][[group_cols + 'MACHINE_NO']].rename(columns={'MACHINE_NO': 'VCD_NO'})
    vcd_df = vcd_df.drop_duplicates(subset = group_cols, keep = 'first')
    shp_df = mini_df[mini_df['MACHINE'] == 'SHP'][[group_cols + 'MACHINE_NO']].rename(columns={'MACHINE_NO': 'SHP_NO'})
    shp_df = shp_df.drop_duplicates(subset = group_cols, keep = 'first')
    
    if vcd_df.empty or shp_df.empty: return []

    # Inner Join으로 VCD, SHP를 모두 거친 Glass만 추출
    combo_df = pd.merge(vcd_df, shp_df, on=group_cols, how='inner')
    meta_df = clean_line_df[group_cols + 'DEFECT_QTY']]
    final_df = pd.merge(combo_df, meta_df, on=group_cols, how='inner')

    # 조합별 통계 산출
    combo_stats = final_df.groupby(group_cols[1:] + ['VCD_NO', 'SHP_NO'], observed=True)['DEFECT_QTY'].agg(['mean', 'count']).reset_index()

    # [Step 3] Compare Combo vs Line (Z-Score)
    merged = pd.merge(combo_stats, line_stats, on=['MODEL', 'PROCESS', 'LINE', 'CODE'], how='left')
    merged['Line_Std'] = merged['Line_Std'].replace(0, 1e-9)
    merged['Z_Score'] = (merged['mean'] - merged['Line_Mean']) / merged['Line_Std']

    for _, row in merged.iterrows():
        if row['count'] < (MIN_DATACOUNT_CNT - 20): continue
        if (row['mean'] - row['Line_Mean']) < 0.5: continue
        
        if row['Z_Score'] > 0.1: 
            combo_name = f"{row['LINE']}_VCD{row['VCD_NO']}+SHP{row['SHP_NO']}"
            results.append({
                'TYPE': 'INTERACTION',
                'MODEL': row['MODEL'],
                'PROCESS': row['PROCESS'],
                'LINE': row['LINE'],
                'MACHINE_ID': combo_name,
                'CODE': row['CODE'],
                'LOGIC': 'L04',
                'WINDOW': window_name,
                'DATACOUNT': int(row['count']), 
                'INDEX': round(row['Z_Score'], 2), 
                'LEVEL': 'High' if row['Z_Score'] > 3.0 else ('Med' if row['Z_Score'] > 2.0 else 'Low'),
                'NOTE': f"Inter. DPU {row['mean']:.2f} (Line {row['Line_Mean']:.2f})"
            })
    return results

def logic_05_slot_correlation(df_slice, window_name):
    """Logic 05: Slot 번호와 Defect 간의 상관관계 분석"""
    results = []
    # CST_SLOT 데이터만 필터링
    slot_df = df_slice[df_slice['MODEL', 'MACHINE'] == 'CST_SLOT'].copy()
    if slot_df.empty: return []
    
    # MACHINE_NO를 숫자로 변환 (분석용)
    slot_df['Slot_Int'] = pd.to_numeric(slot_df['MACHINE_NO'], errors='coerce')
    slot_df = slot_df.dropna(subset=['Slot_Int'])
    
    # 중복 제거 (Glass 단위)
    df_unique = slot_df.groupby(['MODEL', 'PROCESS', 'LINE', 'CODE', 'GLASS_ID', 'Slot_Int'], observed=True)['DEFECT_QTY'].mean().reset_index()

    # Line 단위 Grouping
    grouped = df_unique.groupby(['MODEL', 'PROCESS', 'LINE', 'CODE'], observed=True)

    for name, group in grouped:
        if len(group) < MIN_DATACOUNT_CNT * 2: continue 
        
        # 표준편차가 0이면 상관계수 계산 불가
        if group['DEFECT_QTY'].std() == 0 or group['Slot_Int'].std() == 0:
            continue

        # 상관계수 (Pearson)
        corr = group['Slot_Int'].corr(group['DEFECT_QTY'])
        
        if corr > 0.20:
            model, process, line, code = name
            results.append({
                'TYPE': 'MACHINE',
                'MODEL': model,
                'PROCESS': process,
                'LINE': line, 
                'MACHINE_ID': f"{line}_CST_SLOT_TREND", 
                'CODE': code,
                'LOGIC': 'L05',
                'WINDOW': window_name,
                'DATACOUNT': len(group), 
                'INDEX': round(corr, 2),
                'LEVEL': 'High' if corr > 0.5 else 'Med',
                'NOTE': f"Slot Trend Detected (Corr {corr:.2f})"
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
        use_cols = ['Glass_ID', 'MODEL', 'TIMESTAMP', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_NO', 'MACHINE_ID', 'CODE', 'DEFECT_QTY']
        df = pd.read_excel(INPUT_FILE, dtype=dtype_map, usecols=use_cols, parse_dates=['TIMESTAMP'])
    else:
        # SPOTFIRE 모드
        print("[Mode: SPOTFIRE] Processing Input Data Table...")
        df = input_df.copy()
        if df['TIMESTAMP'].dtype == 'object':
            df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        
        # Category 변환 (속도 최적화)
        for c in ['MODEL', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_NO', 'CODE']:
            if c in df.columns: df[c] = df[c].astype('category')

    # [분석 실행]
    all_results = []
    
    # [수정] CrossTable과 일치시키기 위해 시간(Time) 제거 후 날짜(Date)만 사용
    # 가장 최근 데이터의 날짜 00:00:00 기준
    max_date = df['TIMESTAMP'].max().replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    
    print(f"Data Loaded: {len(df)} rows. Max Date: {max_date}")

    for window_name, days in WINDOWS.items():
        # 시작일 계산 (날짜 기준)
        start_date = max_date - timedelta(days=(days - 1))
        
        # 해당 날짜 00:00:00 포함 이후 모든 데이터
        df_slice = df[df['TIMESTAMP'] >= start_date].copy()
        
        df_slice_general = df_slice[df_slice["MACHINE"] != "CST_SLOT"]
        if not df_slice_general.empty:
            all_results.extend(logic_01_global_line(df_slice_general, window_name))
            all_results.extend(logic_02_local_unit(df_slice_general, window_name))
            all_results.extend(logic_03_short_term_volatility(df_slice_general, window_name))
            all_results.extend(logic_04_interaction_zscore(df_slice_general, window_name))
        
        df_slice_cst_slot = df_slice[df_slice["MACHINE"] == "CST_SLOT"]
        if not df_slice_cst_slot.empty:
            all_results.extend(logic_05_slot_correlation(df_slice_cst_slot, window_name)) 

    # [결과 반환]
    if all_results:
        result_df = pd.DataFrame(all_results)
        # 컬럼 순서 정렬
        cols_order = ['TYPE', 'MODEL', 'PROCESS', 'LINE', 'MACHINE_ID', 'CODE', 
                      'LOGIC', 'WINDOW', 'DATACOUNT', 'INDEX', 'LEVEL', 'NOTE']
        final_cols = [c for c in cols_order if c in result_df.columns]
        result_df = result_df[final_cols]
        
        if EXECUTION_MODE == 'LOCAL':
            result_df.to_excel(OUTPUT_FILE, index=False)
            print(f"Analysis Complete. Saved to {OUTPUT_FILE}")
            print(result_df.head())
            return None
        else:
            return result_df
    else:
        # 결과 없을 시 빈 테이블 리턴 (에러 방지)
        return pd.DataFrame(columns=['TYPE', 'LEVEL', 'NOTE'])

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
