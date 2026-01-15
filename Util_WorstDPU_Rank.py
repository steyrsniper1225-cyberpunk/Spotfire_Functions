import pandas as pd
import numpy as np

def calculate_dpu_rank(input_df, time_frame, time_range):
    """
    Spotfire DataFunction Script: DPU_RANK Calculation
    
    Logic based on Util_Ranking_Worst_DPU.md:
    1. Filter by WINDOWFRAME (time_frame)
    2. For each group (FACTORY, PRODUCT, PROCESS):
       - Select recent N WINDOWs (time_range)
       - Calculate Mean DPU per CODE
       - Rank CODEs (Worst 1st ~ 5th)
    3. Map results back to original DataFrame
    """
    
    # [Step 01] 데이터 복사 및 초기화
    # 원본 데이터 보호를 위해 copy 수행
    df = input_df.copy()
    
    # DPU_RANK 컬럼 초기화 (기본값 'ignore')
    df['DPU_RANK'] = 'ignore'

    # [Step 02] WINDOWFRAME 필터링
    # 사용자가 선택한 Time Frame (예: 'WEEK', 'MONTH')에 해당하는 행만 추출
    mask_frame = df['WINDOWFRAME'] == time_frame
    target_df = df[mask_frame].copy()
    
    # 데이터가 없을 경우 바로 반환
    if target_df.empty:
        return df

    # -------------------------------------------------------------------------
    # [Loop Logic Implementation]
    # Factory/Product/Process 별로 최신 Window를 찾고 Rank를 매기는 과정
    # 속도 최적화를 위해 Loop 대신 GroupBy -> Apply 방식을 사용합니다.
    # -------------------------------------------------------------------------

    def get_worst_codes(group):
        """
        각 그룹(Factory, Product, Process) 별로 수행되는 함수
        """
        # [Step 04 & 05] 최근 N개의 WINDOW 선정
        # 현재 그룹 내 존재하는 모든 Window를 내림차순 정렬 (최신순)
        unique_windows = group['WINDOW'].drop_duplicates().sort_values(ascending=False)
        
        # 상위 N개 (time_range) Window만 선정
        target_windows = unique_windows.head(int(time_range))
        
        # 해당 Window에 속하는 데이터만 Filtering
        filtered_group = group[group['WINDOW'].isin(target_windows)]
        
        if filtered_group.empty:
            return None

        # [Step 06] CODE 별 평균 DPU 계산
        code_stats = filtered_group.groupby('CODE')['DPU'].mean()
        
        # [Step 07 ~ 11] DPU 내림차순 정렬 (Worst 선정)
        # 값이 큰 순서대로 정렬 (DPU가 높을수록 Worst)
        sorted_codes = code_stats.sort_values(ascending=False)
        
        # 상위 5개 코드에 대해 Rank 부여
        ranks = {}
        labels = ["1st", "2nd", "3rd", "4th", "5th"]
        
        for i, (code, dpu) in enumerate(sorted_codes.items()):
            if i < 5:
                ranks[code] = labels[i]
            else:
                break # 5위 밖은 무시 (어차피 기본값이 ignore)
                
        # 결과를 Series로 반환 (Index: Code, Value: Rank String)
        return pd.Series(ranks)

    # -------------------------------------------------------------------------
    # [Main Execution] GroupBy Apply
    # -------------------------------------------------------------------------
    # [Step 03] Grouping keys
    group_keys = ['FACTORY', 'PRODUCT', 'PROCESS_CODE']
    
    # 그룹별 연산 수행 -> 결과는 [FACTORY, PRODUCT, PROCESS, CODE]를 인덱스로 갖는 Series
    # reset_index를 통해 DataFrame 형태로 변환: columns = [FACTORY, PRODUCT, PROCESS, CODE, Rank_Label]
    rank_mapping = target_df.groupby(group_keys).apply(get_worst_codes).reset_index()
    
    # 컬럼명 정리 (마지막 컬럼이 Rank 값임)
    # apply 결과의 마지막 컬럼 이름은 0 또는 임의의 이름이므로 'Rank_Label'로 변경
    rank_mapping.columns = group_keys + ['CODE', 'Rank_Label']

    # -------------------------------------------------------------------------
    # [Merge] 결과를 원본 데이터에 매핑
    # -------------------------------------------------------------------------
    # 원본 데이터에 Rank 정보를 결합하기 위해 Merge 수행
    # 주의: Merge 시 time_frame 필터링된 데이터 뿐만 아니라 전체 데이터 중
    # 해당 Key(Factory, Product, Process, Code)를 가진 행에 Rank가 부여될 수 있음.
    # 하지만 Logic상 "ignore" 처리는 [Step 13]에 의해 다른 Frame이나 조건 불만족 시 유지되어야 함.
    
    # Merge를 위해 원본 df에 임시 인덱스 생성 등을 하지 않고,
    # mapping 테이블을 기준으로 Left Join을 활용하여 값을 업데이트
    
    # df와 rank_mapping을 Merge (Left Join)
    # 키: FACTORY, PRODUCT, PROCESS, CODE
    # rank_mapping에는 Top 5인 항목만 존재함 (apply 함수에서 Top 5만 리턴했으므로)
    
    merged_df = pd.merge(
        df, 
        rank_mapping, 
        on=['FACTORY', 'PRODUCT', 'PROCESS_CODE', 'CODE'], 
        how='left'
    )
    
    # [Step 12 & 13] Rank 부여 및 ignore 처리
    # Merge 결과 Rank_Label이 있는 경우 해당 값을 사용, 없으면 기존 'ignore' 유지
    # 단, 사용자가 선택한 'WINDOWFRAME'이 아닌 행들은 이미 'ignore' 상태여야 함.
    # 하지만 Merge를 하면 FACTORY/PRODUCT/PROCESS/CODE가 같으면 WINDOWFRAME이 달라도 Rank가 붙을 수 있음.
    # Logic 요구사항: "Step_13 : 선택되지 않은 나머지... ignore 부여"
    # -> 즉, Rank는 선택된 Time Frame 내에서만 유효해야 함.
    
    # 조건: WINDOWFRAME이 time_frame과 일치하고, Rank_Label이 NaN이 아닌 경우에만 값 업데이트
    condition = (merged_df['WINDOWFRAME'] == time_frame) & (merged_df['Rank_Label'].notna())
    
    # numpy where를 사용하여 조건부 업데이트
    merged_df['DPU_RANK'] = np.where(
        condition,
        merged_df['Rank_Label'],
        'ignore'
    )
    
    # 불필요한 Rank_Label 컬럼 제거
    final_df = merged_df.drop(columns=['Rank_Label'])
    
    return final_df

# ==========================================
# Spotfire Entry Point
# ==========================================
# Spotfire에서는 아래 변수들이 Input Parameter로 주입됨
# input_df: DataFrame
# time_frame: String (e.g., "WEEK")
# time_range: Integer (e.g., 5)

if __name__ == "__main__":
    # Local Test를 위한 Dummy Code (Spotfire 실행 시 무시됨)
    try:
        # result_df = calculate_dpu_rank(input_df, time_frame, time_range)
        pass
    except NameError:
        pass
else:
    # Spotfire DataFunction 실행 영역
    # input_df, time_frame, time_range는 Spotfire에서 전달됨
    output_df = calculate_dpu_rank(input_df, 'DATE', 2)
