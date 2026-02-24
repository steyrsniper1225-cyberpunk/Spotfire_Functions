import pandas as pd
import numpy as np

def analyze_photo_unit_anomalies(df):
    """
    PHT Unit (VCD, SHP, SCP) 이상 감지 엔진
    - Case 1: 3주간 지속적으로 높은 불량 (Z-score 0.3 이상 또는 Diff 0.05 이상)
    - Case 2: 3주 연속 증가 (v2 < v1 < v0)
    - Case 3: Z-value shift (14~8 Days vs 7~0 Days) 
    """
    records = []
    
    # 필수 컬럼 체크
    req_cols = ['TIMESTAMP', 'MODEL', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE', 'GLASS_ID', 'REAL_DEFECT_QTY']
    if not all(c in df.columns for c in req_cols):
        # 만약 REAL_DEFECT_QTY가 없고 DEF_PNT_X/Y가 있다면 포인트 개수로부터 유도 시도 (선택 사항)
        # 여기서는 원본 로직을 최대한 유지함
        return pd.DataFrame([{
            'Target_EQP': '-', 'Pattern': '-', 'Code': '-', 
            'Story': 'Required columns missing', 'Contribution_Pct': 0.0, 'Priority': '-'
        }])

    # PHT 공정 및 대상 Unit(VCD, SHP, SCP) 필터링
    df_pht = df[df['PROCESS'].str.contains('PHT', na=False, case=False)].copy()
    df_pht = df_pht[df_pht['MACHINE'].isin(['VCD', 'SHP', 'SCP'])]
    
    if df_pht.empty:
        return pd.DataFrame([{
            'Target_EQP': '-', 'Pattern': '-', 'Code': '-', 
            'Story': 'No PHT data found', 'Contribution_Pct': 0.0, 'Priority': '-'
        }])

    df_pht['TIMESTAMP'] = pd.to_datetime(df_pht['TIMESTAMP'])
    max_date = df_pht['TIMESTAMP'].max()
    
    # 주차(ISO Week) 생성
    df_pht['WEEK'] = df_pht['TIMESTAMP'].dt.isocalendar().year.astype(str) + "-" + \
                     df_pht['TIMESTAMP'].dt.isocalendar().week.map(lambda x: f"{x:02d}")
    
    def calc_dpu(group_df):
        # 중복된 Glass_ID 제거 (해당 그룹 내 단일 CODE 가정이므로 GLASS_ID만으로 충분)
        unique_glass = group_df.drop_duplicates(subset=['GLASS_ID'])
        defects = unique_glass['REAL_DEFECT_QTY'].sum()
        glasses = group_df['GLASS_ID'].nunique()
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
            # Case 1: 3주간 지속적으로 높은 불량 (Z-score 0.3 이상 또는 Diff 0.05 이상)
            for (model, line, unit, code), group in pivot_z.groupby(['MODEL', 'LINE', 'MACHINE', 'CODE']):
                for _, row in group.iterrows():
                    m_id = row['MACHINE_ID']
                    z_vals = [row[w_2], row[w_1], row[w_0]]
                    
                    if all(pd.notna(z) for z in z_vals):
                        avg_z = np.mean(z_vals)
                        
                        # 해당 기간 DPU 실값 및 편차 확인
                        dpu_row = pivot_dpu[(pivot_dpu['MACHINE_ID'] == m_id) & (pivot_dpu['CODE'] == code)].iloc[0]
                        dpu_vals = [dpu_row[w_2], dpu_row[w_1], dpu_row[w_0]]
                        
                        # 그룹 평균 DPU (최근 주차 기준)
                        current_group_dpu = pivot_dpu[(pivot_dpu['MODEL']==model) & (pivot_dpu['LINE']==line) & (pivot_dpu['MACHINE']==unit) & (pivot_dpu['CODE']==code)]
                        avg_group_dpu = current_group_dpu[w_0].mean()
                        machine_dpu_last = dpu_row[w_0]
                        diff_last = machine_dpu_last - avg_group_dpu
                        
                        # 현업 요구사항: 0.05ea 수준의 차이가 지속될 때 감지
                        # 3주 평균 Z-score가 0.3 이상이거나, 최근 주차 DPU 차이가 0.05 이상인 경우
                        if avg_z >= 0.3 or diff_last >= 0.05:
                            records.append({
                                'Target_EQP': m_id, 
                                'Pattern': f"PHT_Unit_Case1 ({unit})",
                                'Code': code,
                                'Story': f"[{model}] {line} 라인 {unit} Unit 중 해당 설비가 '{code}' CODE 에 대해 지속적 고불량 (3주 Avg Z: {avg_z:.2f}, 최근 Diff: {diff_last:.4f}).",
                                'Contribution_Pct': np.nan, 'Priority': 'High'
                            })
            
            # Case 2: 3주 연속 증가 (v2 < v1 < v0)
            for (model, line, unit, code), group in pivot_dpu.groupby(['MODEL', 'LINE', 'MACHINE', 'CODE']):
                for _, row in group.iterrows():
                    m_id = row['MACHINE_ID']
                    v2, v1, v0 = row[w_2], row[w_1], row[w_0]
                    if pd.notna(v2) and pd.notna(v1) and pd.notna(v0):
                        # 현업 요구사항: 0.05ea 수준의 증가폭도 감지
                        if v2 < v1 < v0 and (v0 - v2) >= 0.05:
                            records.append({
                                'Target_EQP': m_id, 
                                'Pattern': f"PHT_Unit_Case2 ({unit})",
                                'Code': code,
                                'Story': f"[{model}] {line} 라인 {unit} Unit 중 해당 설비의 '{code}' CODE DPU가 {v2:.4f} -> {v1:.4f} -> {v0:.4f}로 3주 연속 증가 추세임.",
                                'Contribution_Pct': np.nan, 'Priority': 'High'
                            })

    # Case 3: Z-value shift (14~8 Days vs 7~0 Days)
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
                    
                    # dpu_row를 찾기 위해 pivot_shift_dpu에서 일치하는 행을 찾음 (인덱스가 동일할 것이나 안전하게 병합이나 필터링 고려 가능)
                    # 여기서는 i 인덱스 기반으로 접근 (pivot_table 결과가 동일 정렬 기준이므로 가능)
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

    # 결과 DataFrame 생성
    if not records:
        df_result = pd.DataFrame([{
            'Target_EQP': '-',
            'Pattern': '-',
            'Code': '-', 
            'Story': 'No Anomalies Detected',
            'Contribution_Pct': 0.0,
            'Priority': '-'
        }])
    else:
        df_result = pd.DataFrame(records)
        # 중복 제거
        df_result = df_result.drop_duplicates(subset=['Target_EQP', 'Pattern', 'Code'])
    
    return df_result

# ==========================================
# [Spotfire Entry Point]
# ==========================================
# Spotfire Input: df (Table)
# Spotfire Output: result_pht (Table)

if 'df' in locals():
    result_pht = analyze_photo_unit_anomalies(df)
