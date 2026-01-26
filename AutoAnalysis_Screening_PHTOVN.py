import pandas as pd
import numpy as np

def analyze_pht_ovn_flow(input_df, N_WEEKS):
    """
    [Spotfire DataFunction] PHT to OVN Flow Analysis
    
    Args:
        input_df (pd.DataFrame): Glass/Defect History Data
        N_WEEKS (int): Number of recent weeks to analyze
    Returns:
        pd.DataFrame: Analyzed DPU and Delta by Line (PHT/OVN)
    """
    
    # ==============================================================================
    # 1. Data Filter
    # ==============================================================================
    target_codes = ["PLN1-SM", "PLN2-SM"]
    target_processes = ["PLN1PHT", "PLN1OVN", "PLN2PHT", "PLN2OVN"]
    
    # 복사본 생성 및 기본 필터링
    df = input_df[
        (input_df['CODE'].isin(target_codes)) & 
        (input_df['PROCESS'].isin(target_processes))
    ].copy()
    
    # 날짜/Window 처리 (Window가 문자열인 경우 정렬을 위해 처리 필요)
    # 여기서는 WINDOW 컬럼이 이미 "YYYY-WW" 형태의 문자열이라 가정하고 정렬하여 상위 N개 추출
    unique_windows = sorted(df['WINDOW'].unique())
    target_windows = unique_windows[-N_WEEKS:] if len(unique_windows) >= N_WEEKS else unique_windows
    
    df = df[df['WINDOW'].isin(target_windows)].copy()
    
    # ==============================================================================
    # 2. Loop & DataFrame Creation (Pre-aggregation per Glass)
    # ==============================================================================
    # 1 Glass = 1 Row가 되도록 먼저 집계합니다. (Join 시 Cartesian Product 방지)
    # Glass_ID, MODEL, WINDOW, CODE를 Key로 LINE 정보를 남깁니다.
    
    data_dict = {}
    
    for proc in target_processes:
        # 해당 Process 데이터 추출
        sub_df = df[df['PROCESS'] == proc].copy()
        
        # Glass 단위 Unique 집계 (LINE은 해당 공정의 설비, DEFECT_QTY는 Sum)
        # 주의: 한 Glass가 동일 공정 내 여러 설비를 타지 않는다고 가정 (Main 설비 기준)
        glass_agg = sub_df.groupby(['MODEL', 'CODE', 'PROCESS', 'LINE', 'WINDOW', 'Glass_ID'], observed=True)['DEFECT_QTY'].sum().reset_index()
        
        # Dictionary에 저장 (변수명 시뮬레이션: df_pln1pht 등)
        key_name = proc.lower()  # pln1pht, pln1ovn ...
        data_dict[key_name] = glass_agg

    # 명시적 변수 할당 (가독성 및 로직 일치)
    df_pln1pht = data_dict['pln1pht']
    df_pln1ovn = data_dict['pln1ovn']
    df_pln2pht = data_dict['pln2pht']
    df_pln2ovn = data_dict['pln2ovn']

    # ==============================================================================
    # 3. Column Rename
    # ==============================================================================
    # PHT DataFrame: LINE -> PHT
    df_pln1pht = df_pln1pht.rename(columns={'LINE': 'PHT'})
    df_pln2pht = df_pln2pht.rename(columns={'LINE': 'PHT'})
    
    # OVN DataFrame: LINE -> OVN
    # Merge를 위해 불필요한 중복 컬럼(MODEL, CODE 등)은 PHT 기준으로 가져갈 것이므로
    # OVN 쪽에서는 Key(Glass_ID)와 Value(OVN, DEFECT_QTY)만 남기거나, 접미사 처리
    df_pln1ovn = df_pln1ovn.rename(columns={'LINE': 'OVN', 'DEFECT_QTY': 'DEFECT_QTY_OVN'})
    df_pln2ovn = df_pln2ovn.rename(columns={'LINE': 'OVN', 'DEFECT_QTY': 'DEFECT_QTY_OVN'})

    # ==============================================================================
    # 4. Merge (Flow Construction)
    # ==============================================================================
    # Key: Glass_ID (같은 Model, Code, Window라고 가정)
    # merge keys: ['Glass_ID']. 나머지는 검증용
    
    # PLN1 Flow
    df_pln1 = pd.merge(
        df_pln1pht, 
        df_pln1ovn[['Glass_ID', 'OVN', 'DEFECT_QTY_OVN']], 
        on='Glass_ID', 
        how='left'
    )
    
    # PLN2 Flow
    df_pln2 = pd.merge(
        df_pln2pht, 
        df_pln2ovn[['Glass_ID', 'OVN', 'DEFECT_QTY_OVN']], 
        on='Glass_ID', 
        how='left'
    )
    
    # 결측치 처리 (OVN 정보가 없는 경우 'Unknown' 처리하거나 제외)
    df_pln1['OVN'] = df_pln1['OVN'].fillna('Unknown')
    df_pln1['DEFECT_QTY_OVN'] = df_pln1['DEFECT_QTY_OVN'].fillna(0)
    
    df_pln2['OVN'] = df_pln2['OVN'].fillna('Unknown')
    df_pln2['DEFECT_QTY_OVN'] = df_pln2['DEFECT_QTY_OVN'].fillna(0)
    
    # Total Defect Qty 합산 (PHT에서 발생한 것 + OVN에서 발생한 것)
    # 분석 목적에 따라 PHT Defect만 볼지, 합쳐서 볼지 결정해야 하나, 
    # "PLN1-SM" 코드는 공정 전체에서 발생하므로 합산 혹은 PHT Defect 기준으로 흐름 분석
    # 여기서는 PHT Row 기준 Left Join 했으므로 PHT Defect를 메인으로 사용하되
    # Prompt의 의도(Flow 분석)에 맞춰, 해당 Pair에서의 DPU를 구함.
    # (일반적으로는 특정 Step의 Defect Qty를 사용하지만, 여기서는 PHT 데이터의 Defect Qty 사용)
    
    # ==============================================================================
    # 5. Delta Calculation Logic (Function)
    # ==============================================================================
    def calculate_flow_stats(merged_df, flow_name):
        if merged_df.empty: return pd.DataFrame()
        
        # 5-1. Group By (PHT, OVN, WINDOW)
        grouped = merged_df.groupby(['MODEL', 'CODE', 'PHT', 'OVN', 'WINDOW'], observed=True).agg(
            TOTAL_DEFECT=('DEFECT_QTY', 'sum'),
            GLASS_COUNT=('Glass_ID', 'nunique')
        ).reset_index()
        
        # DPU Calculation
        grouped['DPU'] = grouped['TOTAL_DEFECT'] / grouped['GLASS_COUNT']
        
        # 5-2. Delta Calculation (Prev Week vs Curr Week)
        # WINDOW 기준 정렬 -> GroupBy(PHT, OVN) -> Shift -> Diff
        grouped = grouped.sort_values(['MODEL', 'CODE', 'PHT', 'OVN', 'WINDOW'])
        
        grouped['PREV_DPU'] = grouped.groupby(['MODEL', 'CODE', 'PHT', 'OVN'])['DPU'].shift(1)
        grouped['DELTA'] = grouped['DPU'] - grouped['PREV_DPU']
        
        # 첫 주차(NaN)는 0 혹은 제외 처리 (여기서는 0으로 처리하여 유지)
        grouped['DELTA'] = grouped['DELTA'].fillna(0)
        
        grouped['FLOW_GROUP'] = flow_name
        return grouped

    df_pln1_merge = calculate_flow_stats(df_pln1, 'PLN1')
    df_pln2_merge = calculate_flow_stats(df_pln2, 'PLN2')
    
    # Combine for unified processing
    df_merge_all = pd.concat([df_pln1_merge, df_pln2_merge], ignore_index=True)
    
    if df_merge_all.empty:
        return pd.DataFrame({'Message': ['No Data Found']})

    # ==============================================================================
    # 6. Result Aggregation (Marginal Means)
    # ==============================================================================
    results = []
    
    # PHT 관점 집계 (해당 PHT 설비가 포함된 모든 Flow의 평균 DPU/Delta)
    pht_group = df_merge_all.groupby(['FLOW_GROUP', 'MODEL', 'CODE', 'WINDOW', 'PHT'], observed=True)[['DPU', 'DELTA']].mean().reset_index()
    pht_group = pht_group.rename(columns={'PHT': 'EQP_ID'})
    pht_group['EQP_TYPE'] = 'PHT'
    results.append(pht_group)
    
    # OVN 관점 집계 (해당 OVN 설비가 포함된 모든 Flow의 평균 DPU/Delta)
    ovn_group = df_merge_all.groupby(['FLOW_GROUP', 'MODEL', 'CODE', 'WINDOW', 'OVN'], observed=True)[['DPU', 'DELTA']].mean().reset_index()
    ovn_group = ovn_group.rename(columns={'OVN': 'EQP_ID'})
    ovn_group['EQP_TYPE'] = 'OVN'
    results.append(ovn_group)

    # ==============================================================================
    # 7. Final Output
    # ==============================================================================
    df_result = pd.concat(results, ignore_index=True)
    
    # 컬럼 정리 (가독성)
    df_result = df_result[['FLOW_GROUP', 'MODEL', 'CODE', 'WINDOW', 'EQP_TYPE', 'EQP_ID', 'DPU', 'DELTA']]
    
    return df_result

# ==========================================
# Spotfire Execution Block
# ==========================================
# Spotfire의 Input Parameters:
# 1. input_df : Table (Columns: MODEL, CODE, PROCESS, LINE, WINDOW, Glass_ID, DEFECT_QTY)
# 2. N_WEEKS : Value (Integer, e.g., 5)

# Output Parameter:
# 1. output_table : Table

if 'input_df' in globals() and 'N_WEEKS' in globals():
    output_table = analyze_pht_ovn_flow(input_df, N_WEEKS)
