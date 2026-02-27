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
            'MACHINE_ID': '-',
            'LOGIC': '-',
            'CODE': '-', 
            'NOTE': 'No Date/Window Column found'
        }])

    # 4. PHT 공정 및 대상 Unit(VCD, SHP, SCP) 필터링
    df_pht = df_pht[df_pht['PROCESS'].str.contains('PHT', na=False, case=False)]
    df_pht = df_pht[df_pht['MACHINE'].isin(['VCD', 'SHP', 'SCP'])]
    
    if df_pht.empty:
        return pd.DataFrame([{
            'MACHINE_ID': '-',
            'LOGIC': '-',
            'CODE': '-', 
            'NOTE': 'No PHT data found'
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
        
        if all(w in pivot_dpu.columns for w in [w_2, w_1, w_0]):
            # Case 1: 3주간 지속적으로 높은 불량
            for (model, line, unit, code), group_dpu in pivot_dpu.groupby(['MODEL', 'LINE', 'MACHINE', 'CODE']):
                if len(group_dpu) > 1:
                    max_dpu_w2 = group_dpu[w_2].max()
                    max_dpu_w1 = group_dpu[w_1].max()
                    max_dpu_w0 = group_dpu[w_0].max()
                    
                    for _, row in group_dpu.iterrows():
                        m_id = row['MACHINE_ID']
                        v2, v1, v0 = row[w_2], row[w_1], row[w_0]
                        
                        if pd.notna(v2) and pd.notna(v1) and pd.notna(v0):
                            if v2 == max_dpu_w2 and v1 == max_dpu_w1 and v0 == max_dpu_w0 and v0 > 0:
                                
                                z_row = pivot_z[(pivot_z["MACHINE_ID"] == m_id) & (pivot_z["CODE"] == code)].iloc[0]
                                avg_z = np.mean([z_row[w_2], z_row[w_1], z_row[w_0]])
                                
                                records.append({
                                        'MACHINE_ID': m_id, 
                                        'LOGIC': "L01",
                                        'CODE': code,
                                        'NOTE': f"[{model}] {m_id} '{code}' 지속 Worst (최근 DPU : {v0:.2f}, 최근 Z : {z_row[w_0]:.2f}",
                                })
            
            # Case 2: 3주 연속 증가
            for (model, line, unit, code), group in pivot_dpu.groupby(['MODEL', 'LINE', 'MACHINE', 'CODE']):
                for _, row in group.iterrows():
                    m_id = row['MACHINE_ID']
                    v2, v1, v0 = row[w_2], row[w_1], row[w_0]
                    if pd.notna(v2) and pd.notna(v1) and pd.notna(v0):
                        if v2 < v1 < v0 and (v0 - v2) >= 0.05:
                            records.append({
                                'MACHINE_ID': m_id, 
                                'LOGIC': "L02",
                                'CODE': code,
                                'NOTE': f"[{model}] {m_id} '{code}' {v2:.2f} -> {v1:.2f} -> {v0:.2f}로 연속 증가 추세",
                            })

    # 6. Case 3: Z-value shift (TIMESTAMP 컬럼이 존재할 때만 실행)
    max_date = df_pht['TIMESTAMP'].max()
    past_start, past_end = max_date - pd.Timedelta(days=14), max_date - pd.Timedelta(days=8)
    curr_start, curr_end = max_date - pd.Timedelta(days=7), max_date
        
    df_pht['PERIOD'] = 'Ignore'
    df_pht.loc[(df_pht['TIMESTAMP'] >= past_start) & (df_pht['TIMESTAMP'] <= past_end), 'PERIOD'] = 'PAST'
    df_pht.loc[(df_pht['TIMESTAMP'] > curr_start) & (df_pht['TIMESTAMP'] <= curr_end), 'PERIOD'] = 'CURR'
        
    df_period = df_pht[df_pht['PERIOD'].isin(['PAST', 'CURR'])]
    
    if not df_period.empty:
        dpu_period = df_period.groupby(['PERIOD', 'MODEL', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE']).apply(calc_dpu).reset_index(name='DPU')
        flass_cnt_period = df_period.groupby(['PERIOD', 'MODEL', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE'])["Glass_ID"].nunique().reset_index(name='GLASS_CNT')
        dpu_period = pd.merge(dpu_period, glass_cnt_period, on = ['PERIOD', 'MODEL', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE'])
        
        dpu_period['Z_SCORE'] = dpu_period.groupby(['PERIOD', 'MODEL', 'LINE', 'MACHINE', 'CODE'])['DPU'].transform(
            lambda x: (x - x.mean()) / (x.std() + 0.0001) if len(x) > 1 else 0
        )
            
        pivot_shift_z = dpu_period.pivot_table(index=['MODEL', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE'], columns='PERIOD', values='Z_SCORE').reset_index()
        pivot_shift_dpu = dpu_period.pivot_table(index=['MODEL', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE'], columns='PERIOD', values='DPU').reset_index()
        pivot_shift_glass = dpu_period.pivot_table(index=['MODEL', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE'], columns='PERIOD', values='GLASS_CNT').reset_index()
            
        if 'PAST' in pivot_shift_z.columns and 'CURR' in pivot_shift_z.columns:
            for i, row in pivot_shift_z.iterrows():
                z_past, z_curr = row['PAST'], row['CURR']
                code = row['CODE']
                
                if pd.notna(z_past) and pd.notna(z_curr):
                    shift_delta_z = z_curr - z_past
                        
                    dpu_row = pivot_shift_dpu.iloc[i]
                    dpu_past, dpu_curr = dpu_row['PAST'], dpu_row['CURR']
                    
                    shift_delta_dpu = dpu_curr - dpu_past
                    
                    glass_row = pivot_shift_glass.iloc[i]
                    glass_past, glass_curr = glass_row['PAST'], glass_row['CURR']
                        
                    if abs(shift_delta_z) > 2.0 or abs(shift_delta_dpu) >= 0.05 and (shift_delta_dpu > 0) and (glass_past >=30 and glass_curr >=30):
                        direction = "증가" if shift_delta_dpu > 0 else "감소"
                        records.append({
                            'MACHINE_ID': row['MACHINE_ID'], 
                            'LOGIC': "L03",
                            'CODE': code,
                            'NOTE': f"[{row['MODEL']}] {row['MACHINE_ID']} '{CODE}' {direction} (Delta : {shift_delta_dpu:.2f}, Z-Shift : {shift_delta_z:.2f}, GLS(Before/After) : {glass_past:.0f}, {glass_curr:.0f})"
                        })

    # 7. 결과 DataFrame 생성
    if not records:
        df_result = pd.DataFrame([{
            'MACHINE_ID': '-',
            'LOGIC': '-',
            'CODE': '-', 
            'NOTE': 'No Date/Window Column found'
        }])
    else:
        df_result = pd.DataFrame(records)
        df_result = df_result.drop_duplicates(subset=['MACHINE_ID', 'LOGIC', 'CODE'])
    
    return df_result

# ==========================================
# [Spotfire Entry Point]
# ==========================================
output_table = analyze_photo_unit_anomalies(input_table)