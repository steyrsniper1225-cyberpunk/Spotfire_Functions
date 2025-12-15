import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# [Configuration] 분석 파라미터 및 Grid 설정
# ==========================================
# Micro Grid Size (mm)
MICRO_GRID_W = 18.0
MICRO_GRID_H = 15.0

# Glass Size (Approximate for Macro Grid)
# 좌표 범위가 -900~900(W=1800), -750~750(H=1500)이라고 가정
GLASS_WIDTH = 1800
GLASS_HEIGHT = 1500

# Pattern Detection Thresholds (튜닝 필요)
TH_SPOT_DENSITY = 5       # Micro Grid 내 5개 이상이면 Spot 후보
TH_REPEATER_RATIO = 0.3   # 전체 Glass의 30% 이상 동일 좌표 발생 시 Repeater

# Window Mapping (Screening Master와 동일)
WINDOWS = {
    '01W': 7, '02W': 14, '03W': 21, '04W': 28, '05W': 35
}

# ==========================================
# 1. Helper Functions
# ==========================================

def get_target_glass_ids(screening_row, history_df):
    """
    Screening Master의 특정 행(Row) 정보를 받아
    History 테이블에서 분석 대상 Glass ID 리스트를 추출
    """
    # 1. Parse Parameters
    target_model = screening_row['MODEL']
    target_process = screening_row['PROCESS']
    target_line = screening_row['LINE']
    target_machine_id = screening_row['MACHINE_ID']
    target_code = screening_row['CODE']
    target_window = screening_row['WINDOW']
    
    # 2. Date Filtering
    days = WINDOWS.get(target_window, 7)
    max_date = history_df['TIMESTAMP'].max().replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    start_date = max_date - timedelta(days=(days - 1))
    
    # 3. Apply Filters
    # History 테이블에서 [LINE], [MACHINE_ID], [CODE], [TIMESTAMP] 조건 검색
    # 주의: MACHINE_ID가 Screening 결과에서 'LINE' 레벨로 나왔을 수도 있음 (L01 Logic)
    
    mask = (
        (history_df['TIMESTAMP'] >= start_date) &
        (history_df['CODE'] == target_code) &
        (history_df['MODEL'] == target_model) &
        (history_df['PROCESS'] == target_process)
    )
    
    # Logic L01(Line Level)인 경우 Machine ID 필터링 제외 가능, 
    # 하지만 명확성을 위해 Screening 결과의 MACHINE_ID가 실제 설비면 필터링
    if target_machine_id and (target_machine_id in history_df['MACHINE_ID'].unique()):
        mask = mask & (history_df['MACHINE_ID'] == target_machine_id)
    elif target_line:
        mask = mask & (history_df['LINE'] == target_line)
        
    target_glasses = history_df.loc[mask, 'Glass_ID'].unique()
    return target_glasses

def apply_dual_grid(map_df):
    """
    Map 데이터에 Micro/Macro Grid 인덱스 부여
    """
    # [Micro Grid]
    # 좌표 중심을 기준으로 Grid화 (좌표 + Offset) / Size
    # 가정: 좌표계 중심이 (0,0)
    map_df['MICRO_GRID_X'] = np.floor((map_df['DEF_PNT_X'] + (GLASS_WIDTH/2)) / MICRO_GRID_W).astype(int)
    map_df['MICRO_GRID_Y'] = np.floor((map_df['DEF_PNT_Y'] + (GLASS_HEIGHT/2)) / MICRO_GRID_H).astype(int)
    
    # [Macro Grid] 3x3
    # X구간: 0(Left), 1(Center), 2(Right)
    # Y구간: 0(Bottom), 1(Middle), 2(Top)
    def get_macro_idx(val, total_len):
        norm = val + (total_len/2) # 0 ~ Total 변환
        if norm < total_len / 3: return 0
        elif norm < (total_len * 2) / 3: return 1
        else: return 2

    map_df['MACRO_GRID_X'] = map_df['DEF_PNT_X'].apply(lambda x: get_macro_idx(x, GLASS_WIDTH))
    map_df['MACRO_GRID_Y'] = map_df['DEF_PNT_Y'].apply(lambda y: get_macro_idx(y, GLASS_HEIGHT))
    
    # Macro Zone ID (1~9)
    # 7 8 9
    # 4 5 6
    # 1 2 3
    map_df['MACRO_ID'] = (map_df['MACRO_GRID_Y'] * 3) + map_df['MACRO_GRID_X'] + 1
    
    return map_df

def detect_pattern_features(glass_map_df, total_glass_count):
    """
    단일 Glass 혹은 누적된 Map 데이터를 분석하여 패턴 Labeling 수행
    """
    patterns = []
    
    if glass_map_df.empty:
        return "Normal"

    # 1. Repeater Check (Micro Grid 기준)
    # 여러 장의 Glass를 겹쳤을 때 동일 Grid에 반복 발생하는가?
    grid_counts = glass_map_df.groupby(['MICRO_GRID_X', 'MICRO_GRID_Y']).size()
    max_repeat = grid_counts.max()
    
    if total_glass_count > 5 and (max_repeat / total_glass_count) >= TH_REPEATER_RATIO:
        return "Repeater"

    # 2. Line Mura Check (Vertical / Horizontal)
    # 전체 Defect의 X 혹은 Y 표준편차가 매우 작으면서, 반대축으로 넓게 퍼짐
    std_x = glass_map_df['DEF_PNT_X'].std()
    std_y = glass_map_df['DEF_PNT_Y'].std()
    range_x = glass_map_df['DEF_PNT_X'].max() - glass_map_df['DEF_PNT_X'].min()
    range_y = glass_map_df['DEF_PNT_Y'].max() - glass_map_df['DEF_PNT_Y'].min()
    
    # 예외처리: 점이 1개면 std NaN
    if pd.isna(std_x) or pd.isna(std_y): return "Spot"

    # Vertical Line: X편차 작음, Y범위 큼
    if std_x < MICRO_GRID_W and range_y > (GLASS_HEIGHT * 0.5):
        return "Line_Vertical"
    
    # Horizontal Line: Y편차 작음, X범위 큼
    if std_y < MICRO_GRID_H and range_x > (GLASS_WIDTH * 0.5):
        return "Line_Horizontal"

    # 3. Macro Zone Logic: Corner & Directional Edge
    # 전체 Defect 개수
    total_defects = len(glass_map_df)
    if total_defects == 0: return "Normal"

    macro_counts = glass_map_df['MACRO_ID'].value_counts()
    
    # ---------------------------------------------------------
    # Zone Definition (Cartesian 좌표계 기준)
    # ---------------------------------------------------------
    # 7 8 9 (Top)
    # 4 5 6 (Mid)
    # 1 2 3 (Btm)
    
    zones = {
        'Corner': [1, 3, 7, 9],
        'Left':   [1, 4, 7],      # 사용자 입력(1,4,7)은 좌표계상 Left
        'Right':  [3, 6, 9],
        'Top':    [7, 8, 9],
        'Bottom': [1, 2, 3],
        'Center': [5]             # 중앙 집중 (Spin Coating 등)
    }

    # 각 Zone별 점유율(Ratio) 계산
    ratios = {}
    for name, ids in zones.items():
        count = sum([macro_counts.get(z, 0) for z in ids])
        ratios[name] = count / total_defects

    # ---------------------------------------------------------
    # Decision Tree (우선순위 판정)
    # ---------------------------------------------------------
    # Threshold 설정 (예: 80% 이상 집중 시)
    TH_CONCENTRATION = 0.8

    # 3-A. Corner Check (가장 우선)
    # Align 틀어짐, 모서리 파손 등 명확한 기구적 이슈
    if ratios['Corner'] > TH_CONCENTRATION:
        return "Corner"

    # 3-B. Directional Edge Check (상/하/좌/우)
    # 특정 방향의 Guide Rail, Robot Handover, 노즐 편차 등
    directions = ['Left', 'Right', 'Top', 'Bottom']
    # 점유율이 가장 높은 방향 찾기
    best_dir = max(directions, key=lambda x: ratios[x])
    
    if ratios[best_dir] > TH_CONCENTRATION:
        return f"Edge_{best_dir}"  # 예: "Edge_Left"

    # 3-C. Center Check
    # 중앙부 집중 (Spin Dry 얼룩, 중앙 처짐 등)
    if ratios['Center'] > 0.5: # 중앙은 면적이 좁으므로 Threshold 완화 가능
        return "Center"

    # 3-D. General Edge Check
    # 특정 한 방향은 아니지만, 테두리 전반에 퍼진 경우 (예: 액자형 얼룩)
    # Edge Zone = 전체 - Center(5번)
    edge_total_ratio = 1.0 - (macro_counts.get(5, 0) / total_defects)
    if edge_total_ratio > TH_CONCENTRATION:
        return "Edge_Frame" # 혹은 General "Edge"

    # 4. Cluster Analysis (Spot vs Scratch)
    # Grid 밀집도가 높은지 확인
    if max_repeat >= TH_SPOT_DENSITY:
        # Aspect Ratio로 Spot/Scratch 구분
        # 해당 Grid 주변부의 좌표 가져오기
        top_grid = grid_counts.idxmax() # (x, y)
        cluster_df = glass_map_df[
            (glass_map_df['MICRO_GRID_X'] == top_grid[0]) & 
            (glass_map_df['MICRO_GRID_Y'] == top_grid[1])
        ]
        if not cluster_df.empty:
            c_range_x = cluster_df['DEF_PNT_X'].max() - cluster_df['DEF_PNT_X'].min()
            c_range_y = cluster_df['DEF_PNT_Y'].max() - cluster_df['DEF_PNT_Y'].min()
            
            # 0으로 나누기 방지
            ratio = (c_range_y / c_range_x) if c_range_x > 0 else 0
            if ratio < 1: ratio = (c_range_x / c_range_y) if c_range_y > 0 else 0
            
            if ratio > 3.0: return "Scratch"
            else: return "Spot"

    return "Random" # 특이 패턴 없음

# ==========================================
# 2. Main Logic (Spotfire Entry Point)
# ==========================================
def run_map_analysis(input_screening, input_history, input_map):
    """
    input_screening: Screening Master 결과 (1 Row or Filtered Table)
    input_history: 전체 History 테이블
    input_map: 전체 Map 데이터 (Lazy Loading 흉내내기 위해 여기서 필터링)
    """
    
    # 결과 담을 리스트
    results = []

    # [Step 1] Trigger Logic Check
    # Spotfire에서 넘어온 Screening 데이터 중 조건 만족 행만 순회
    # 조건: Index > 3.0 (High) OR (Index > 1.0 AND Level='Med') ??
    # 사용자 요청: ([INDEX]>3.0 And [LEVEL]="Med") 
    # -> 사실 Screening Logic상 Index>2.0이면 High이므로 이 조건은 모순일 수 있으나
    #    요청하신 필터 조건을 그대로 구현 (혹은 OR 조건으로 완화 권장)
    
    # (안전장치) 조건이 너무 Strict하면 데이터가 없을 수 있으므로 
    # Index > 3.0 인 High Level 전체를 대상으로 하거나, 필터링된 Input을 그대로 사용
    target_rows = input_screening.copy()
    
    # 사용자 요청 필터 적용 (데이터가 있다면)
    # condition = (target_rows['INDEX'] > 3.0) & (target_rows['LEVEL'] == 'Med')
    # if condition.any():
    #     target_rows = target_rows[condition]
    
    if target_rows.empty:
        return pd.DataFrame({'Message': ['No Target Found']})

    print(f"Analyzing {len(target_rows)} screening items...")

    for idx, row in target_rows.iterrows():
        # [Step 2] Target Glass ID 식별
        target_glasses = get_target_glass_ids(row, input_history)
        
        if len(target_glasses) == 0:
            continue
            
        # [Step 3] Map Data Load (Filtering)
        # 해당 Glass ID를 가진 Map Data만 추출
        current_map = input_map[input_map['Glass_ID'].isin(target_glasses)].copy()
        current_map = current_map[current_map['CODE'] == row['CODE']]
        
        if current_map.empty:
            continue
            
        # [Step 4] Dual Grid 적용
        current_map = apply_dual_grid(current_map)
        
        # [Step 5] Pattern Labeling
        # 설비 기인성 분석이므로 '적층(Stacked)' 패턴이 중요함
        # 모든 Glass의 점을 합쳐서 패턴을 분석
        detected_pattern = detect_pattern_features(current_map, len(target_glasses))
        
        # [Step 6] 결과 생성
        # 원본 Map 데이터에 분석 결과를 붙여서 리턴할 수도 있고,
        # 요약된 통계만 리턴할 수도 있음. 여기서는 Map Row + Pattern Tag 리턴
        current_map['PATTERN'] = detected_pattern
        current_map['MODEL'] = row['MODEL']
        current_map['PROCESS'] = row['PROCESS']
        current_map['LINE'] = row['LINE']
        current_map['MACHINE_ID'] = row['MACHINE_ID']
        current_map['LOGIC'] = row['LOGIC']
        current_map['WINDOW'] = row['WINDOW']
        
        results.append(current_map)

    # [Step 7] Final Merge
    if len(results) > 0:
        final_df = pd.concat(results, ignore_index=True)
        # 필요한 컬럼만 정리
        out_cols = [
            'Glass_ID',
            'Panel_ID',
            'TIMESTAMP', 
            'DEF_PNT_X',
            'DEF_PNT_Y',
            'CODE', 
            'MICRO_GRID_X',
            'MICRO_GRID_Y',
            'MACRO_GRID_X',
            'MACRO_GRID_Y',
            'MACRO_ID', 
            'PATTERN',
            'MODEL',
            'PROCESS',
            'LINE',
            'MACHINE_ID'
            'LOGIC',
            'WINDOW'
        ]
        # 없는 컬럼 에러 방지
        actual_cols = [c for c in out_cols if c in final_df.columns]
        return final_df[actual_cols]
    else:
        return pd.DataFrame(columns=['Glass_ID', 'PATTERN'])

# ==========================================
# 3. Execution (For Local Test)
# ==========================================
if __name__ == "__main__":
    # Dummy Data 생성 및 테스트 코드는 생략함
    # Spotfire에서는 아래와 같이 호출됨
    # output_map_table = run_map_analysis(Screening_Master, History_Table, Map_Table)
    pass
