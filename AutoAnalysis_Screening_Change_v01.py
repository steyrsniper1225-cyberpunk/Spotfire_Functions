import pandas as pd
import numpy as np

# ==========================================
# [Helper] Week Calculation (ISO Calendar)
# ==========================================
def get_spotfire_window(timestamp_series):
    """
    Spotfire 방식의 "Year-Week" 문자열 생성
    Example: 2025-48
    """
    # ISO Calendar 기준 (월요일 시작)
    iso_cal = timestamp_series.dt.isocalendar()
    return iso_cal.year.astype(str) + "-" + iso_cal.week.map(lambda x: f"{x:02d}")

# ==========================================
# Main Logic
# ==========================================
def run_dpu_variation_analysis(df_history):
    """
    Input: FUNC_GLASS_HISTORY (Raw Data)
    Output: Table 4 (Explanation Lines)
    """
    
    # -------------------------------------------------------
    # [Pre-Processing] Window 생성 및 기본 컬럼 정리
    # -------------------------------------------------------
    df = df_history.copy()
    
    # 1. Window Column Setup
    if 'WINDOW(SPOTFIRE)' in df.columns:
        df['WINDOW'] = df['WINDOW(SPOTFIRE)']
    else:
        if 'TIMESTAMP' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['TIMESTAMP']):
            df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        df['WINDOW'] = get_spotfire_window(df['TIMESTAMP'])
    
    # -------------------------------------------------------
    # [Table 1] Trend Aggregation (Corrected for Duplicates)
    # -------------------------------------------------------
    # Data is granular (Machine/Step level), but DEFECT_QTY is repeated per Glass+Code.
    # We must deduplicate by Glass_ID before summing defects.
    
    # Level 1: Glass Aggregation
    glass_level = df.groupby(['MODEL', 'CODE', 'WINDOW', 'Glass_ID'], as_index=False).agg(
        GLASS_DEFECT=('DEFECT_QTY', 'max') # Assuming repeated value, take max
    )
    
    # Level 2: Window Aggregation
    t1_agg = glass_level.groupby(['MODEL', 'CODE', 'WINDOW'], as_index=False).agg(
        TOTAL_DEFECT=('GLASS_DEFECT', 'sum'),
        TOTAL_GLASS=('Glass_ID', 'nunique') # count is also fine since we group by Glass_ID above
    )
    
    # DPU 계산
    t1_agg['DPU'] = t1_agg['TOTAL_DEFECT'] / t1_agg['TOTAL_GLASS']
    
    # Filter 2: Glass Count >= 300
    t1_agg = t1_agg[t1_agg['TOTAL_GLASS'] >= 300]
    
    # Filter 1: 최신 5개 주차만 유지 (Model, Code 별)
    # WINDOW 기준 정렬 후 Tail(5)
    t1_agg = t1_agg.sort_values(['MODEL', 'CODE', 'WINDOW'])
    t1_agg = t1_agg.groupby(['MODEL', 'CODE']).tail(5)
    
    # -------------------------------------------------------
    # [Table 2] Variation Detection (Trigger)
    # -------------------------------------------------------
    table2_rows = []
    
    # Model/Code 별로 루프를 돌며 전주 대비 변동 확인
    for (model, code) in t1_agg.groupby(['MODEL', 'CODE']).groups.keys():
        group = t1_agg[(t1_agg['MODEL'] == model) & (t1_agg['CODE'] == code)].sort_values('WINDOW')
        
        # 1. CODE 종류에 따른 임계값(Threshold) 설정
        if "PK" in str(code):
            threshold = 0.05
        elif "PLN" in str(code):
            threshold = 0.10
        else:
            threshold = 0.10  # 기본값
            
        # Shift를 이용해 이전 주차 데이터 가져오기
        group['PREV_DPU'] = group['DPU'].shift(1)
        group['PREV_WINDOW'] = group['WINDOW'].shift(1)
        
        # 첫 주는 비교 대상 없으므로 제외
        valid_rows = group.dropna(subset=['PREV_DPU'])
        
        for idx, row in valid_rows.iterrows():
            diff = row['DPU'] - row['PREV_DPU']
            abs_diff = abs(diff)
            
            # 2. 동적 임계값 적용
            if abs_diff >= threshold:
                analysis_id = f"{model}_{code}_{row['WINDOW']}_VS_{row['PREV_WINDOW']}"
                
                table2_rows.append({
                    'ANALYSIS_NO': analysis_id,
                    'MODEL': model,
                    'CODE': code,
                    'CURR_WINDOW': row['WINDOW'],
                    'PREV_WINDOW': row['PREV_WINDOW'],
                    'CURR_DPU': row['DPU'],
                    'PREV_DPU': row['PREV_DPU'],
                    'DELTA_DPU': diff,
                    'ABS_DELTA': abs_diff
                })
    
    df_table_02 = pd.DataFrame(table2_rows)
    
    if df_table_02.empty:
        return pd.DataFrame(columns=['ANALYSIS_NO', 'EXPLAIN_LINE', 'LINE_CONTRIBUTION']), t1_agg, df_table_02

    # -------------------------------------------------------
    # [Table 3 & 4] Root Cause Finding (Line Level)
    # -------------------------------------------------------
    table4_results = []
    
    # Table 2의 각 Case(Analysis No)에 대해 상세 분석 수행
    for idx, case in df_table_02.iterrows():
        target_model = case['MODEL']
        target_code = case['CODE']
        windows = [case['PREV_WINDOW'], case['CURR_WINDOW']]
        
        # 1. 해당 Case의 Raw Data 필터링 (Current Week & Prev Week)
        subset = df[
            (df['MODEL'] == target_model) & 
            (df['CODE'] == target_code) & 
            (df['WINDOW'].isin(windows))
        ].copy()
        
        # 2. Line Level 집계 (Process, Line 별)
        # 중요: Line별 집계 시에도 Glass 단위 중복 제거 필요
        # Line을 통과한 Glass들의 Defect 합 (DPU 개념)
        
        # Step A: Per-Line-Glass Aggregation
        line_glass_agg = subset.groupby(['PROCESS', 'LINE', 'WINDOW', 'Glass_ID'], as_index=False).agg(
            GLASS_DEFECT=('DEFECT_QTY', 'max')
        )
        
        # Step B: Per-Line Aggregation
        line_agg = line_glass_agg.groupby(['PROCESS', 'LINE', 'WINDOW'], as_index=False).agg(
            LINE_DEFECT=('GLASS_DEFECT', 'sum'),
            LINE_GLASS=('Glass_ID', 'nunique') # Glass Count
        )
        
        # 3. Pivot: Columns=[PREV, CURR] 형태로 변환하여 비교 용이하게 만듦
        # Index: [PROCESS, LINE], Columns: [WINDOW] -> Values: [DPU, Count]
        line_pivot = line_agg.pivot(index=['PROCESS', 'LINE'], columns='WINDOW', values=['LINE_DEFECT', 'LINE_GLASS'])
        
        # 컬럼 Flatten (예: LINE_DEFECT_2025-48)
        line_pivot.columns = [f'{col[0]}_{col[1]}' for col in line_pivot.columns]
        line_pivot = line_pivot.reset_index()
        
        # 결측치(특정 주에만 가동된 라인) 처리 -> 0
        line_pivot = line_pivot.fillna(0)
        
        # 동적 컬럼명 할당
        curr_def_col = f"LINE_DEFECT_{case['CURR_WINDOW']}"
        prev_def_col = f"LINE_DEFECT_{case['PREV_WINDOW']}"
        curr_glass_col = f"LINE_GLASS_{case['CURR_WINDOW']}"
        prev_glass_col = f"LINE_GLASS_{case['PREV_WINDOW']}"
        
        # 4. [Core Logic] Contribution Analysis
        # 해당 주차의 전체 Glass 수 (Table 1 데이터 활용 가능하나 여기서 재계산이 안전)
        # Note: subset is filtered by Model/Code/Window. 
        # Total Glass per Window should be calculated from subset (deduplicated)
        total_glass_curr = subset[subset['WINDOW'] == case['CURR_WINDOW']]['Glass_ID'].nunique()
        total_glass_prev = subset[subset['WINDOW'] == case['PREV_WINDOW']]['Glass_ID'].nunique()
        
        # 라인별 DPU 계산
        # (Zero Division 방지: Glass가 0이면 DPU도 0)
        line_pivot['DPU_CURR'] = np.where(line_pivot[curr_glass_col] > 0, 
                                          line_pivot[curr_def_col] / line_pivot[curr_glass_col], 0)
        line_pivot['DPU_PREV'] = np.where(line_pivot[prev_glass_col] > 0, 
                                          line_pivot[prev_def_col] / line_pivot[prev_glass_col], 0)
        
        # 라인별 기여도(Contribution) 계산
        # Contribution = Line DPU * (Line Glass / Total Glass)
        # 즉, 전체 DPU 중 이 라인이 깎아먹은 점수
        line_pivot['CONTRI_CURR'] = line_pivot['DPU_CURR'] * (line_pivot[curr_glass_col] / total_glass_curr)
        line_pivot['CONTRI_PREV'] = line_pivot['DPU_PREV'] * (line_pivot[prev_glass_col] / total_glass_prev)
        
        # 기여도 변동폭 (Delta Contribution)
        # 이 값이 전체 DPU 변동(Delta DPU)과 가장 유사한(방향 및 크기) 라인이 범인
        line_pivot['DELTA_CONTRI'] = line_pivot['CONTRI_CURR'] - line_pivot['CONTRI_PREV']
        
        # 5. Ranking
        # 전체 DPU가 증가(+)했다면 Delta Contri가 가장 큰(+) 놈이 원인
        # 전체 DPU가 감소(-)했다면 Delta Contri가 가장 작은(-) 놈이 원인
        # 즉, 전체 변동과 부호가 같으면서 절대값이 가장 큰 것을 찾음 (혹은 단순히 Signed Sort)
        
        is_increase = case['DELTA_DPU'] > 0
        
        if is_increase:
            # 가장 크게 증가시킨 라인 Top 1
            culprit = line_pivot.sort_values('DELTA_CONTRI', ascending=False).iloc[0]
        else:
            # 가장 크게 감소시킨 라인 Top 1
            culprit = line_pivot.sort_values('DELTA_CONTRI', ascending=True).iloc[0]
            
        # 6. 결과 수집 (Table 4)
        table4_results.append({
            'ANALYSIS_NO': case['ANALYSIS_NO'],
            'MODEL': case['MODEL'],
            'CODE': case['CODE'],
            'WINDOW_CHANGE': f"{case['PREV_WINDOW']} -> {case['CURR_WINDOW']}",
            'TOTAL_DELTA_DPU': round(case['DELTA_DPU'], 3),
            'EXPLAIN_PROCESS': culprit['PROCESS'],
            'EXPLAIN_LINE': culprit['LINE'],
            'LINE_DELTA_CONTRI': round(culprit['DELTA_CONTRI'], 3),
            'NOTE': f"Line DPU: {culprit['DPU_PREV']:.2f} -> {culprit['DPU_CURR']:.2f}"
        })

    return pd.DataFrame(table4_results), t1_agg, df_table_02

if __name__ == "__main__":
    input_file = 'dummy_screening_master_v2.xlsx'
    sheet_name = 'Sheet1'
    output_file = 'Result_AutoAnalysis.xlsx'
    output_t1 = 'Result_Table1.xlsx'
    output_t2 = 'Result_Table2.xlsx'

    print(f"Loading {input_file}...")
    try:
        df = pd.read_excel(input_file, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        exit(1)

    # Check columns and rename if necessary
    if 'DEF_QTY' in df.columns and 'DEFECT_QTY' not in df.columns:
        print("Renaming DEF_QTY to DEFECT_QTY")
        df.rename(columns={'DEF_QTY': 'DEFECT_QTY'}, inplace=True)
    
    # Run analysis
    print("Running Analysis...")
    result_df, result_df_t1, result_df_t2 = run_dpu_variation_analysis(df)

    # Save results
    if not result_df.empty:
        print(f"Saving results to {output_file}...")
        result_df.to_excel(output_file, index=False)
    else:
        print("Analysis returned empty result (Table 4).")
    
    # Save Table 1
    if not result_df_t1.empty:
        print(f"Saving Table 1 to {output_t1}...")
        result_df_t1.to_excel(output_t1, index=False)
        
    # Save Table 2
    if not result_df_t2.empty:
        print(f"Saving Table 2 to {output_t2}...")
        result_df_t2.to_excel(output_t2, index=False)
        
    print("Done.")
