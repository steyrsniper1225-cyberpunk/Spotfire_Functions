import pandas as pd
import numpy as np

# ==========================================
# [Configuration] 설정 및 매핑 정보
# ==========================================
PROCESS_GROUP_MAP = {
    'GATPHT': 'GATSPT',
    'GATDET': 'GATPHT',
    'GATSTR': 'GATDET',
    'TM1SPT': 'GATSTR',
    'TM1PHT': 'TM1SPT',
    'TM1DET': 'TM1PHT',
    'TM1STR': 'TM1DET',
    'XGTSPT': 'TM1STR',
    'XGTPHT': 'XGTSPT',
    'XGTDET': 'XGTPHT',
    'XGTSTR': 'XGTDET',
    'SD1SPT': 'XGTSTR',
    'SD1PHT': 'SD1SPT',
    'SD1DET': 'SD1PHT',
    'SD1STR': 'SD1DET',
    'PLN1PHT': 'SD1STR',
    'PLN1OVN': 'PLN1PHT',
    'PLN1DET': 'PLN1OVN',
    'SD2SPT': 'PLN1DET',
    'SD2PHT': 'SD2SPT',
    'SD2DET': 'SD2PHT',
    'SD2STR': 'SD2DET',
    'PLN2PHT': 'SD2STR',
    'PLN2OVN': 'PLN2PHT',
    'PLN2DET': 'PLN2OVN',
    'ANDSPT': 'PLN2DET',
    'ANDPHT': 'ANDSPT',
    'ANDDET': 'ANDPHT',
    'ANDWET': 'ANDDET',
    'ANDSTR': 'ANDWET'
}

def normalize_coordinates(df):
    if 'DEF_PNT_X' in df.columns and 'DEF_PNT_Y' in df.columns:
        df['X_PCT'] = (df['DEF_PNT_X'] + 900) / 1800 * 100
        df['Y_PCT'] = (df['DEF_PNT_Y'] + 750) / 1500 * 100
    return df

def detect_background_pattern(row):
    x, y = row.get('X_PCT', -1), row.get('Y_PCT', -1)
    if x == -1 or y == -1: return "Unknown"
    if (2 <= x <= 4) and (2 <= y <= 10): return "Case1:좌측하단"
    if (90 <= y <= 100): return "Case2:상단Edge"
    if (0 <= x <= 5): return "Case3:좌측Edge"
    if (95 <= x <= 100): return "Case4:우측Edge"
    return "Random"

# ==========================================
# [Module] PHT Unit 이상 감지 엔진
# ==========================================
def analyze_photo_unit_anomalies(df):
    records = []
    
    # 필수 컬럼 체크
    req_cols = ['TIMESTAMP', 'MODEL', 'PROCESS', 'LINE', 'MACHINE', 'MACHINE_ID', 'CODE']
    if not all(c in df.columns for c in req_cols):
        return records

    # PHT 공정 및 대상 Unit(VCD, SHP, SCP) 필터링
    df_pht = df[df['PROCESS'].str.contains('PHT', na=False, case=False)].copy()
    df_pht = df_pht[df_pht['MACHINE'].isin(['VCD', 'SHP', 'SCP'])]
    
    if df_pht.empty:
        return records

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
                        # Poisson 분포 변동성을 고려하여 추세가 뚜렷(v0-v2 >= 0.05)하면 감지
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
                    
                    dpu_row = pivot_shift_dpu.iloc[i]
                    dpu_past, dpu_curr = dpu_row['PAST'], dpu_row['CURR']
                    shift_delta_dpu = dpu_curr - dpu_past
                    
                    # 현업 요구사항: Z-score 0.3 이상 또는 DPU 0.05 이상 변화 시 감지
                    if abs(shift_delta_z) > 0.3 or abs(shift_delta_dpu) >= 0.05:
                        direction = "상승(악화)" if shift_delta_dpu > 0 else "감소(개선)"
                        records.append({
                            'Target_EQP': row['MACHINE_ID'], 
                            'Pattern': f"PHT_Unit_Case3 ({row['MACHINE']})",
                            'Code': code,
                            'Story': f"[{row['MODEL']}] {row['LINE']} 라인 {row['MACHINE']} Unit 중 해당 설비의 '{code}' CODE DPU가 {direction}됨 (Delta: {shift_delta_dpu:.4f}, Z-Shift: {shift_delta_z:.2f}).",
                            'Contribution_Pct': np.nan, 'Priority': 'Medium' if shift_delta_dpu > 0 else 'Low'
                        })

    return records

    return records

# ==========================================
# [Main DataFunction] 통합 진단 엔진
# ==========================================
def run_integrated_detective(df_glass, df_defect):
    df_glass.columns = df_glass.columns.str.upper()
    df_defect.columns = df_defect.columns.str.upper()
    
    # 1. 실제 불량 수 사전 집계 (Merge 전 증폭 방지)
    # GLASS_ID, CODE 별로 실제 몇 개의 포인트가 있는지 카운트
    df_defect_counts = df_defect.groupby(['GLASS_ID', 'CODE']).size().reset_index(name='REAL_DEFECT_QTY')

    # 2. Master Glass 정보와 병합 (Right Join 대신 Left Join으로 모든 Glass 유지)
    df = pd.merge(df_glass, df_defect, on=['GLASS_ID', 'CODE'], how='left')
    
    # 3. 실제 집계된 수치를 데이터프레임에 결합
    df = pd.merge(df, df_defect_counts, on=['GLASS_ID', 'CODE'], how='left')
    df['REAL_DEFECT_QTY'] = df['REAL_DEFECT_QTY'].fillna(0)
    
    print(f"DEBUG: Joined DF shape: {df.shape}, Defects found: {df[df['REAL_DEFECT_QTY']>0]['GLASS_ID'].nunique()} glasses")
    
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ---------------------------------------------------------
    # 데이터 전처리
    # ---------------------------------------------------------
    df['LOT_ID'] = df['GLASS_ID'].astype(str).str[4:10]
    df = normalize_coordinates(df)
    df['PATTERN_CASE'] = df.apply(detect_background_pattern, axis=1)

    # 누락 방지: Input에 MACHINE 컬럼이 없을 경우 대비 (실제로는 존재해야 함)
    if 'MACHINE' not in df.columns:
        df['MACHINE'] = 'UNKNOWN'

    # 1. Noise Filter (2 Sigma) - CODE별로 분리하여 Threshold 산출
    # Glass/Code 별 실제 불량 수 
    glass_code_stats = df.drop_duplicates(subset=['GLASS_ID', 'CODE'])[['LOT_ID', 'GLASS_ID', 'CODE', 'REAL_DEFECT_QTY']]
    
    outlier_list = []
    outlier_glass_ids = set()
    
    # 각 CODE별로 독립적인 Threshold 산산
    for code in glass_code_stats['CODE'].unique():
        code_data = glass_code_stats[glass_code_stats['CODE'] == code]['REAL_DEFECT_QTY']
        if code_data.empty: continue
        
        mean_v, std_v = code_data.mean(), code_data.std()
        # Outlier Threshold는 최소 5개 이상일 때만 의미를 갖도록 보정 (저불량 모델 대응)
        threshold = max(5.0, mean_v + (2 * std_v))
        
        # 해당 코드에서 임계치를 넘는 자재 찾기
        code_outliers = glass_code_stats[(glass_code_stats['CODE'] == code) & (glass_code_stats['REAL_DEFECT_QTY'] > threshold)].copy()
        if not code_outliers.empty:
            code_outliers['ISSUE_REASON'] = code_outliers.apply(
                lambda r: f"[{code}] {r['REAL_DEFECT_QTY']:.0f}ea (Threshold: {threshold:.1f})", axis=1
            )
            outlier_list.append(code_outliers)
            outlier_glass_ids.update(code_outliers['GLASS_ID'].tolist())

    # Outlier 데이터프레임 구성
    if outlier_list:
        df_outlier_summary = pd.concat(outlier_list)
        # 여러 코드에서 중복으로 걸린 경우 Reason을 합침
        df_outlier_summary = df_outlier_summary.groupby('GLASS_ID').agg({
            'LOT_ID': 'first',
            'ISSUE_REASON': lambda x: " / ".join(x)
        }).reset_index()
    else:
        df_outlier_summary = pd.DataFrame(columns=['GLASS_ID', 'LOT_ID', 'ISSUE_REASON'])

    # 원본 데이터에서 Outlier 제외
    df_clean = df[~df['GLASS_ID'].isin(outlier_glass_ids)].copy()
    
    # Outlier 시트용 데이터 (좌표 컬럼 제외)
    df_outliers_report = df_outlier_summary.copy()

    # 2. Spatial Analysis (기존 패턴 감지)
    summary_records = []
    tech_records = []
    
    # 패턴 분석용 데이터셋 (좌표가 있는 실제 불량 행만 필터링)
    df_actual_defects = df_clean[df_clean['DEF_PNT_X'].notna()].copy()
    
    machine_pattern_stat = df_actual_defects.groupby(['PROCESS', 'LINE', 'MACHINE_ID', 'PATTERN_CASE']).size().reset_index(name='DEFECT_CNT')
    target_patterns = machine_pattern_stat[machine_pattern_stat['PATTERN_CASE'] != 'Random']
    
    for _, row in target_patterns.iterrows():
        process, line, machine_id, pattern, cnt = row['PROCESS'], row['LINE'], row['MACHINE_ID'], row['PATTERN_CASE'], row['DEFECT_CNT']
        line_total = machine_pattern_stat[(machine_pattern_stat['LINE'] == line)]['DEFECT_CNT'].sum()
        contribution = (cnt / line_total) * 100 if line_total > 0 else 0
        
        if contribution < 30: continue
        
        spatial_z_score = (cnt - machine_pattern_stat['DEFECT_CNT'].mean()) / (machine_pattern_stat['DEFECT_CNT'].std() + 0.001)
        lot_dist = df_clean[(df_clean['MACHINE_ID'] == machine_id) & (df_clean['PATTERN_CASE'] == pattern)]
        lot_cv = lot_dist.groupby('LOT_ID').size().std() / (lot_dist.groupby('LOT_ID').size().mean() + 0.001)
        
        prev_process = PROCESS_GROUP_MAP.get(process, 'Unknown')
        prev_history = "발생 안함"
        if prev_process != 'Unknown':
            prev_cnt = len(df_clean[(df_clean['PROCESS'] == prev_process) & (df_clean['PATTERN_CASE'] == pattern)])
            prev_history = "발생 이력 있음" if prev_cnt > 10 else "발생 안함"
            
        story = (f"[{pattern}] 패턴이 {line} 라인 내에서 {machine_id} 설비에 {contribution:.1f}% 집중됨. "
                 f"2 Sigma 이탈 자재를 제외한 후에도 뚜렷하게 나타남. "
                 f"이전 공정({prev_process})에서는 {prev_history}.")
        
        summary_records.append({
            'Target_EQP': machine_id, 
            'Pattern': pattern, 
            'Code': 'Total',
            'Story': story,
            'Contribution_Pct': round(contribution, 1),
            'Priority': 'High' if contribution > 50 and lot_cv <= 1.5 else 'Medium'
        })
        
        tech_records.append({
            'Target_EQP': machine_id, 'Pattern': pattern, 'Spatial_Z_Score': round(spatial_z_score, 2),
            'Lot_CV': round(lot_cv, 2), 'Prev_Process': prev_process
        })

    # 사진 공정 이상 감지 모듈 실행
    photo_anomalies = analyze_photo_unit_anomalies(df_clean)
    summary_records.extend(photo_anomalies)

    if not summary_records:
        df_summary = pd.DataFrame()
    else:
        df_summary = pd.DataFrame(summary_records)
        # 컬럼 순서 보장 (Code를 Pattern과 Story 사이에 배치)
        cols = ['Target_EQP', 'Pattern', 'Code', 'Story', 'Contribution_Pct', 'Priority']
        # 혹시 모를 추가 컬럼 대응
        remaining_cols = [c for c in df_summary.columns if c not in cols]
        df_summary = df_summary[cols + remaining_cols]
        # 중복 제거 (Target_EQP, Pattern, Code 조합)
        df_summary = df_summary.drop_duplicates(subset=['Target_EQP', 'Pattern', 'Code'])

    df_tech = pd.DataFrame(tech_records).drop_duplicates() if tech_records else pd.DataFrame()
    
    return df_summary, df_tech, df_outliers_report

if __name__ == "__main__":
    # Local 테스트용
    glass_file = 'Result_Glass_Data_v3.xlsx' 
    defect_file = 'Result_Defect_Data_v3.xlsx'
    
    try:
        print("1. 데이터 로딩 중...")
        df_glass_input = pd.read_csv(glass_file) if glass_file.endswith('.csv') else pd.read_excel(glass_file)
        df_defect_input = pd.read_csv(defect_file) if defect_file.endswith('.csv') else pd.read_excel(defect_file)
        
        print("2. 분석 로직 실행 중...")
        df_sum, df_tec, df_out = run_integrated_detective(df_glass_input, df_defect_input)
        
        print("\n=== [Summary Report] ===")
        print(df_sum.head())
        
        # 결과 저장 로직
        output_file = 'Result_DPU_Intg_Detec.xlsx'
        print(f"\n3. 결과 저장 중... ({output_file})")
        
        with pd.ExcelWriter(output_file) as writer:
            df_sum.to_excel(writer, sheet_name='Summary', index=False)
            df_tec.to_excel(writer, sheet_name='Technical', index=False)
            df_out.to_excel(writer, sheet_name='Outliers', index=False)
            
        print("작업이 완료되었습니다.")
        
    except Exception as e:
        print(f"Error: 실행 중 오류 발생 - {e}")

# ==========================================
# [Spotfire Entry Point]
# ==========================================
# Spotfire Input: df_glass (Table), df_defect (Table)
# Spotfire Output: df_summary, df_tech, df_outliers (Table)

if 'df_glass' in locals() and 'df_defect' in locals():
    # Spotfire DataFunction으로 실행될 때 입력 테이블 매핑
    df_summary, df_tech, df_outliers = run_integrated_detective(df_glass, df_defect)
