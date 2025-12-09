import pandas as pd
import numpy as np
from datetime import timedelta

# --------------------------------------------------------------------------------
# 1. [Settings] Time Window Code Mapping Definition
# --------------------------------------------------------------------------------
# 사용자가 정의한 Time Window 코드와 실제 일(Day) 수 매핑
TIME_WINDOWS_MAP = {
    '01W': 7,
    '02W': 14,
    '03W': 21,
    '01M': 30,
    '03M': 90,
}

# --------------------------------------------------------------------------------
# 2. [Source Mapping] Spotfire Input Data Tables
# --------------------------------------------------------------------------------
source_data_dict = {
    'GAT-PK': dt_GAT_PK,
    'TM1-PK': dt_TM1_PK,
    'XGT-PK': dt_XGT_PK,
    'SD1-PK': dt_SD1_PK,
    'SD2-PK': dt_SD2_PK,
    'PLN1-FM': dt_PLN1_FM,
    'PLN1-SM': dt_PLN1_SM,
    'PLN2-FM': dt_PLN2_FM,
    'PLN2-SM': dt_PLN2_SM
}

# --------------------------------------------------------------------------------
# 3. [Pattern Labeling Logic] (이전과 동일)
# --------------------------------------------------------------------------------
def detect_pattern(df_subset):
    """
    로드된 Defect 데이터군(Group)의 통계적 특징을 분석하여 
    가장 지배적인 Pattern Label을 반환합니다.
    """
    if df_subset.empty:
        return "None"
    
    x = df_subset['DEF_PNT_X']
    y = df_subset['DEF_PNT_Y']
  
    # 1. Repeater 감지 (동일 좌표 중복도)
    # 소수점 좌표일 경우 반올림하여 비교하거나 Binning 후 비교 필요
    # 여기서는 간단히 좌표 분산이 극도로 작은 군집이 있는지 확인
    
    # 2. Corner/Edge 감지
    # 데이터의 80% 이상이 외곽(Edge) 영역에 있는지 확인
    is_edge_x = (x.abs() > 800)
    is_edge_y = (y.abs() > 650)
    if (is_edge_x | is_edge_y).mean() > 0.7:
        return "Corner/Edge"
      
    # 3. Line Mura / Scratch (Linearity 확인)
    std_x = x.std()
    std_y = y.std()
    
    # X분산이 작고 Y분산이 크면 -> Vertical Line
    if std_y > (std_x * 5):
        return "Line Mura (Vertical)"

    # Y분산이 작고 X분산이 크면 -> Horizontal Line
    if std_x > (std_y * 5):
        return "Line Mura (Horizontal)"

    # 4. Spot (집중도 확인)
    # Grid(100mm) 단위로 나누어 Max Density 확인
    try:
        heatmap, _, _ = np.histogram2d(x, y, bins=[18, 15]) # Coarse check
        if (heatmap.max() / len(df_subset)) > 0.3: # 전체 점의 30% 이상이 한 칸에 뭉침
            return "Spot"
    except:
        pass

    return "Random" # Default

# --------------------------------------------------------------------------------
# 4. [Main Logic] Time Code Parsing & History Matching
# --------------------------------------------------------------------------------

stacked_data = []

# (1) Filtering Target Rows (Risk Score 조건)
target_rows = Screening_Master[
    (Screening_Master['Risk_Score'] > 2.0) & 
    (Screening_Master['Risk_Level'] == 'Med')
]

if not target_rows.empty:
    for idx, row in target_rows.iterrows():
        
        # A. Key 정보 및 Time Window 파싱
        target_code = row['CODE']      
        target_machine = row['MACHINE_ID'] 
        window_code = row['Time_Window'] # ex: '01W'
        
        # 기간(Days) 계산 (Dictionary에서 조회, 없으면 기본값 7일)
        # ANDRPR_Timekey -> TIMESTAMP 컬럼을 항상 기준으로 써야될 듯
        days_to_look_back = TIME_WINDOWS_MAP.get(window_code, 7)
        
        # B. History 테이블에서 기준 시간(Anchor Time) 설정
        # 해당 설비의 가장 최근 기록을 기준으로 역산합니다.
        machine_history = History[History['MACHINE_ID'] == target_machine]
        
        if not machine_history.empty:
            # 기준 종료 시간: 해당 설비 이력의 마지막 시간 (최신)
            anchor_end_time = machine_history['TIMESTAMP'].max()
            # 기준 시작 시간: 종료 시간 - 기간
            anchor_start_time = anchor_end_time - timedelta(days=days_to_look_back)
            
            # C. History Filtering (Time Range)
            target_history_subset = machine_history[
                (machine_history['TIMESTAMP'] >= anchor_start_time) &
                (machine_history['TIMESTAMP'] <= anchor_end_time)
            ]
            
            # D. Target Glass ID 확보
            target_glass_ids = target_history_subset['Glass_ID'].unique()
            
            if len(target_glass_ids) > 0:
                
                # E. Defect Data Fetching
                source_df = source_data_dict.get(target_code)
                
                if source_df is not None and not source_df.empty:
                    
                    # F. Glass ID 기반 Filtering (정확한 매핑)
                    mask = source_df['Glass_ID'].isin(target_glass_ids)
                    df_chunk = source_df.loc[mask].copy()
                    
                    if not df_chunk.empty:
                        # G. Narrow Mapping & Labeling
                        df_chunk['TARGET_EQP_ID'] = target_machine
                        df_chunk['PATTERN_LABEL'] = detect_pattern(df_chunk)
                        
                        # [Optional] 분석 기간 정보 추가 (디버깅용)
                        df_chunk['ANALYSIS_WINDOW'] = window_code
                        
                        stacked_data.append(df_chunk)

# (2) Final Concatenation & Grid Generation
if stacked_data:
    Map_Data = pd.concat(stacked_data, ignore_index=True)
    
    # Dual Grid Generation
    bin_size_x = 18.0
    bin_size_y = 15.0
    
    Map_Data['BIN_X'] = np.floor(Map_Data['DEF_PNT_X'] / bin_size_x) * bin_size_x
    Map_Data['BIN_Y'] = np.floor(Map_Data['DEF_PNT_Y'] / bin_size_y) * bin_size_y
    
    # Macro Zone Logic
    conditions_x = [
        (Map_Data['DEF_PNT_X'] < -300),
        (Map_Data['DEF_PNT_X'] >= -300) & (Map_Data['DEF_PNT_X'] <= 300),
        (Map_Data['DEF_PNT_X'] > 300)
    ]
    Map_Data['ZONE_X'] = np.select(conditions_x, ['Left', 'Center', 'Right'], default='Unknown')
    
    conditions_y = [
        (Map_Data['DEF_PNT_Y'] < -250),
        (Map_Data['DEF_PNT_Y'] >= -250) & (Map_Data['DEF_PNT_Y'] <= 250),
        (Map_Data['DEF_PNT_Y'] > 250)
    ]
    Map_Data['ZONE_Y'] = np.select(conditions_y, ['Bottom', 'Middle', 'Top'], default='Unknown')
    
    Map_Data['MACRO_ZONE'] = Map_Data['ZONE_Y'] + '-' + Map_Data['ZONE_X']

else:
    # 빈 데이터테이블 스키마 생성
    Map_Data = pd.DataFrame(columns=[
        'Glass_ID', 'Panel_ID', 'DEF_CODE', 'TIMESTAMP', 
        'DEF_PNT_X', 'DEF_PNT_Y', 'TARGET_EQP_ID', 'PATTERN_LABEL',
        'ANALYSIS_WINDOW', 'BIN_X', 'BIN_Y', 'ZONE_X', 'ZONE_Y', 'MACRO_ZONE'
    ])
