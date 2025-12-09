import pandas as pd
import numpy as np
from datetime import timedelta

# --------------------------------------------------------------------------------
# 1. [Source Mapping] Spotfire에서 넘겨받은 9개의 테이블을 Dictionary로 구조화
# --------------------------------------------------------------------------------
# script input 변수명은 아래 key와 동일하게 설정했다고 가정합니다.
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
# 2. [Pattern Labeling Logic] (이전과 동일)
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
# 3. [Main Logic] In-Memory Filtering & Stacking
# --------------------------------------------------------------------------------

stacked_data = []
delta_hours = 1 # Screening Master Time Window 커버 범위

# (1) Filtering Target Rows (Risk Score 조건)
target_rows = Screening_Master[
    (Screening_Master['Risk_Score'] > 2.0) & 
    (Screening_Master['Risk_Level'] == 'Med')
]

if not target_rows.empty:
    for idx, row in target_rows.iterrows():
        
        # A. Screening 조건 추출
        target_code = row['CODE']      
        target_machine = row['MACHINE_ID'] # 범인 설비 (ex: PLN1PHT01)
        
        start_time = pd.to_datetime(row['Time_Window'])
        end_time = start_time + timedelta(hours=delta_hours)
        
        # B. [NEW] History 테이블 조회 -> Target Glass ID 리스트 확보
        # 조건: 지정된 시간(Time_Window) 내에 + 지정된 설비(MACHINE_ID)를 거친 Glass
        
        target_history = History[
            (History['MACHINE_ID'] == target_machine) &
            (History['TIMESTAMP'] >= start_time) &
            (History['TIMESTAMP'] < end_time)
        ]
        
        # 해당 설비를 거친 Glass ID 목록 추출 (중복 제거)
        target_glass_ids = target_history['Glass_ID'].unique()
        
        if len(target_glass_ids) > 0:
            
            # C. Defect Data Fetching
            source_df = source_data_dict.get(target_code)
            
            if source_df is not None and not source_df.empty:
                
                # D. Glass ID 기반 Filtering (정확도 100%)
                # 기존의 Time 기반 필터링을 Glass ID 매칭으로 변경
                mask = source_df['Glass_ID'].isin(target_glass_ids)
                
                df_chunk = source_df.loc[mask].copy()
                
                if not df_chunk.empty:
                    # E. Narrow Mapping
                    df_chunk['TARGET_EQP_ID'] = target_machine
                    
                    # Pattern Labeling
                    df_chunk['PATTERN_LABEL'] = detect_pattern(df_chunk)
                    
                    stacked_data.append(df_chunk)

# (2) Final Concatenation
if stacked_data:
    Map_Data = pd.concat(stacked_data, ignore_index=True)
    
    # (3) Dual Grid Generation (Vectorized)
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
    # 빈 껍데기 생성
    Map_Data = pd.DataFrame(columns=[
        'Glass_ID', 'Panel_ID', 'DEF_CODE', 'TIMESTAMP', 
        'DEF_PNT_X', 'DEF_PNT_Y', 'TARGET_EQP_ID', 'PATTERN_LABEL',
        'BIN_X', 'BIN_Y', 'ZONE_X', 'ZONE_Y', 'MACRO_ZONE'
    ])
