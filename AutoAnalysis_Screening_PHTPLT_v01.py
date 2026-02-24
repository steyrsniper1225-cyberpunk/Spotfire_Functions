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
    iso_cal = timestamp_series.dt.isocalendar()
    return iso_cal.year.astype(str) + "-" + iso_cal.week.map(lambda x: f"{x:02d}")

def analyze_photo_unit_anomalies(df):
    """
    PHT Unit (VCD, SHP, SCP) 이상 감지 엔진
    - Case 1: 3주간 지속적으로 높은 불량 (Z-score 0.3 이상 또는 Diff 0.05 이상)
    - Case 2: 3주 연속 증가 (v2 < v1 < v0)
    - Case 3: Z-value shift (14~8 Days vs 7~0 Days) - TIMESTAMP 존재 시 동작
    """
    records = []
    df_pht = df.copy()
    
    # 1. 컬럼명 표준화 (Change_v01 형식 호환)
    col_map = {'GLASS_ID': 'Glass_ID', 'REAL_DEFECT_QTY': 'DEFECT_QTY'}
    df_pht.rename(columns={k: v for k, v in col_map.items() if k in df_pht.columns}, inplace=True)
    
    # 2. Window(주차) 컬럼 셋업
    if 'WINDOW(SPOTFIRE)' in df_pht.columns:
        df_pht['WEEK'] = df_pht['WINDOW(SPOTFIRE)']
    elif 'WINDOW' in df_pht.columns:
        df_pht['WEEK'] = df_pht['WINDOW']
    else:
        if 'TIMESTAMP' in df_pht.columns:
            df_pht['TIMESTAMP'] = pd.to_datetime(df_pht['TIMESTAMP'])
            df_pht['WEEK'] = get_spotfire_window(df_pht['TIMESTAMP'])
        else:
            return pd.DataFrame([{'Target_EQP': '-', 'Pattern': '-', 'Code': '-', 'Story': 'No Date/Window Column Found', 'Contribution_Pct': 0.0, 'Priority': '-'}])

    # 3. 필수 컬럼 체크
    req_cols = ['MODEL', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE', 'Glass_ID', 'DEFECT_QTY', 'WEEK']
    if not all(c in df_pht.columns for c in req_cols):
        return pd.DataFrame([{
            'Target_EQP': '-', 'Pattern': '-', 'Code': '-', 
            'Story': 'Required columns missing (Need DEFECT_QTY, Glass_ID, etc.)', 'Contribution_Pct': 0.0, 'Priority': '-'
        }])

    # 4. PHT 공정 및 대상 Unit(VCD, SHP, SCP) 필터링
    df_pht = df_pht[df_pht['PROCESS'].str.contains('PHT', na=False, case=False)]
    df_pht = df_pht[df_pht['MACHINE'].isin(['VCD', 'SHP', 'SCP'])]
    
    if df_pht.empty:
        return pd.DataFrame([{
            'Target_EQP': '-', 'Pattern': '-', 'Code': '-', 
            'Story': 'No PHT data found', 'Contribution_Pct': 0.0, 'Priority': '-'
        }])

    # 5. DPU 계산 함수 (중복 Row 대응)
    def calc_dpu(group_df):
        # Change_v01과 동일하게 Glass 단위로 Max 값을 취해 중복 제거 후 합산
        unique_glass = group_df.groupby('Glass_ID')['DEFECT_QTY'].max()
        defects = unique_glass.sum()
        glasses = group_df['Glass_ID'].nunique()
        return defects / glasses if glasses > 0 else 0

    unique_weeks = sorted(df_pht['WEEK'].unique())
    if len(unique_weeks) >= 3:
        w_2, w_1, w_0 = unique_weeks[-3:] 
        
        df_3w = df_pht[df_pht['WEEK'].isin([w_2, w_1, w_0])]
        
        # CODE 별로 DPU 집계
        dpu_weekly = df_3w.groupby(['WEEK', 'MODEL', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE']).apply(calc_dpu).reset_index(name='DPU')
        
        # Z-Score도 동일 CODE 내에서 비교
        dpu_weekly['Z_SCORE'] = dpu_weekly.groupby(['WEEK', 'MODEL', 'LINE', 'MACHINE', 'CODE'])['DPU'].transform(
            lambda x: (x - x.mean()) / (x.std() + 0.0001) if len(x) > 1 else 0
        )
        
        pivot_z = dpu_weekly.pivot_table(index=['MODEL', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE'], columns='WEEK', values='Z_SCORE').reset_index()
        pivot_dpu = dpu_weekly.pivot_table(index=['MODEL', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE'], columns='WEEK', values='DPU').reset_index()
        
        if all(w in pivot_z.columns for w in [w_2, w_1, w_0]):
            # Case 1: 3주간 지속적으로 높은 불량
            for (model, line, unit, code), group in pivot_z.groupby(['MODEL', 'LINE', 'MACHINE', 'CODE']):
                for _, row in group.iterrows():
                    m_id = row['MACHINE_ID']
                    z_vals = [row[w_2], row[w_1], row[w_0]]
                    
                    if all(pd.notna(z) for z in z_vals):
                        avg_z = np.mean(z_vals)
                        
                        dpu_row = pivot_dpu[(pivot_dpu['MACHINE_ID'] == m_id) & (pivot_dpu['CODE'] == code)].iloc[0]
                        current_group_dpu = pivot_dpu[(pivot_dpu['MODEL']==model) & (pivot_dpu['LINE']==line) & (pivot_dpu['MACHINE']==unit) & (pivot_dpu['CODE']==code)]
                        
                        avg_group_dpu = current_group_dpu[w_0].mean()
                        machine_dpu_last = dpu_row[w_0]
                        diff_last = machine_dpu_last - avg_group_dpu
                        
                        if avg_z >= 0.3 or diff_last >= 0.05:
                            records.append({
                                'Target_EQP': m_id, 
                                'Pattern': f"PHT_Unit_Case1 ({unit})",
                                'Code': code,
                                'Story': f"[{model}] {line} 라인 {unit} Unit 중 해당 설비가 '{code}' CODE 에 대해 지속적 고불량 (3주 Avg Z: {avg_z:.2f}, 최근 Diff: {diff_last:.4f}).",
                                'Contribution_Pct': np.nan, 'Priority': 'High'
                            })
            
            # Case 2: 3주 연속 증가
            for (model, line, unit, code), group in pivot_dpu.groupby(['MODEL', 'LINE', 'MACHINE', 'CODE']):
                for _, row in group.iterrows():
                    m_id = row['MACHINE_ID']
                    v2, v1, v0 = row[w_2], row[w_1], row[w_0]
                    if pd.notna(v2) and pd.notna(v1) and pd.notna(v0):
                        if v2 < v1 < v0 and (v0 - v2) >= 0.05:
                            records.append({
                                'Target_EQP': m_id, 
                                'Pattern': f"PHT_Unit_Case2 ({unit})",
                                'Code': code,
                                'Story': f"[{model}] {line} 라인 {unit} Unit 중 해당 설비의 '{code}' CODE DPU가 {v2:.4f} -> {v1:.4f} -> {v0:.4f}로 3주 연속 증가 추세임.",
                                'Contribution_Pct': np.nan, 'Priority': 'High'
                            })

    # 6. Case 3: Z-value shift (TIMESTAMP 컬럼이 존재할 때만 실행)
    if 'TIMESTAMP' in df_pht.columns and pd.api.types.is_datetime64_any_dtype(df_pht['TIMESTAMP']):
        max_date = df_pht['TIMESTAMP'].max()
        past_start, past_end = max_date - pd.Timedelta(days=14), max_date - pd.Timedelta(days=8)
        curr_start, curr_end = max_date - pd.Timedelta(days=7), max_date
        
        df_pht['PERIOD'] = 'Ignore'
        df_pht.loc[(df_pht['TIMESTAMP'] >= past_start) & (df_pht['TIMESTAMP'] <= past_end), 'PERIOD'] = 'PAST'
        df_pht.loc[(df_pht['TIMESTAMP'] > curr_start) & (df_pht['TIMESTAMP'] <= curr_end), 'PERIOD'] = 'CURR'
        
        df_period = df_pht[df_pht['PERIOD'].isin(['PAST', 'CURR'])]
        if not df_period.empty:
            dpu_period = df_period.groupby(['PERIOD', 'MODEL', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE']).apply(calc_dpu).reset_index(name='DPU')
            dpu_period['Z_SCORE'] = dpu_period.groupby(['PERIOD', 'MODEL', 'LINE', 'MACHINE', 'CODE'])['DPU'].transform(
                lambda x: (x - x.mean()) / (x.std() + 0.0001) if len(x) > 1 else 0
            )
            
            pivot_shift_z = dpu_period.pivot_table(index=['MODEL', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE'], columns='PERIOD', values='Z_SCORE').reset_index()
            pivot_shift_dpu = dpu_period.pivot_table(index=['MODEL', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE'], columns='PERIOD', values='DPU').reset_index()
            
            if 'PAST' in pivot_shift_z.columns and 'CURR' in pivot_shift_z.columns:
                for i, row in pivot_shift_z.iterrows():
                    z_past, z_curr = row['PAST'], row['CURR']
                    code = row['CODE']
                    if pd.notna(z_past) and pd.notna(z_curr):
                        shift_delta_z = z_curr - z_past
                        
                        dpu_row = pivot_shift_dpu.iloc[i]
                        dpu_past, dpu_curr = dpu_row['PAST'], dpu_row['CURR']
                        shift_delta_dpu = dpu_curr - dpu_past
                        
                        if abs(shift_delta_z) > 0.3 or abs(shift_delta_dpu) >= 0.05:
                            direction = "상승(악화)" if shift_delta_dpu > 0 else "감소(개선)"
                            records.append({
                                'Target_EQP': row['MACHINE_ID'], 
                                'Pattern': f"PHT_Unit_Case3 ({row['MACHINE']})",
                                'Code': code,
                                'Story': f"[{row['MODEL']}] {row['LINE']} 라인 {row['MACHINE']} Unit 중 해당 설비의 '{code}' CODE DPU가 {direction}됨 (Delta: {shift_delta_dpu:.4f}, Z-Shift: {shift_delta_z:.2f}).",
                                'Contribution_Pct': np.nan, 'Priority': 'Medium' if shift_delta_dpu > 0 else 'Low'
                            })

    # 7. 결과 DataFrame 생성
    if not records:
        df_result = pd.DataFrame([{
            'Target_EQP': '-', 'Pattern': '-', 'Code': '-', 
            'Story': 'No Anomalies Detected', 'Contribution_Pct': 0.0, 'Priority': '-'
        }])
    else:
        df_result = pd.DataFrame(records)
        df_result = df_result.drop_duplicates(subset=['Target_EQP', 'Pattern', 'Code'])
    
    return df_result

# ==========================================
# [Spotfire Entry Point]
# ==========================================
# Spotfire Input: df (Table)
# Spotfire Output: result_pht (Table)

if 'df' in locals():
    result_pht = analyze_photo_unit_anomalies(df)