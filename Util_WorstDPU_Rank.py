import pandas as pd
import numpy as np

def calculate_dpu_rank(input_df, time_frame, time_range):
    """
    Spotfire DataFunction Script: DPU_RANK Calculation
    
    [Update Logic]
    1. Filter by WINDOWFRAME
    2. Calculate 'Window Rank' (1=Recent, 2=Previous...) per Group
    3. Select Data where Window Rank <= time_range
    4. Calculate Worst 5 Codes based on selected data
    5. Merge Rank back to original DF, but ONLY update rows where Window Rank <= time_range
    """
    
    # [Step 01] 데이터 복사 및 초기화
    df = input_df.copy()
    df['DPU_RANK'] = 'ignore'

    # [Step 02] WINDOWFRAME 필터링 및 타겟 데이터 설정
    # 해당 Frame이 아닌 데이터는 계산에서 제외
    mask_frame = df['WINDOWFRAME'] == time_frame
    
    # 데이터가 없을 경우 바로 반환
    if not mask_frame.any():
        return df

    # -------------------------------------------------------------------------
    # [New Step] Window의 최신 순위(Rank) 계산
    # -------------------------------------------------------------------------
    # 동일 Group(Factory/Product/Process) 내에서 Window가 최신일수록 1에 가까운 값을 가짐
    # method='dense' : 1등, 2등, 3등... 식으로 순차적 정수 부여
    
    group_keys = ['FACTORY', 'PRODUCT', 'PROCESS_DESC']
    
    # 1. 대상 데이터에 대해 Window Rank 계산 (원본 인덱스 유지)
    # ascending=False : 문자열 기준 내림차순 (예: "26-05W" > "26-04W") -> 최신이 1순위
    # 주의: WINDOW 값이 문자열이므로 포맷이 일정하다는 전제하에 정렬됨
    df.loc[mask_frame, 'WIN_RANK'] = df[mask_frame].groupby(group_keys)['WINDOW'] \
                                                   .rank(method='dense', ascending=False)
    
    # 2. 분석 대상 데이터 추출 (최근 N개 Window 이내인 데이터만)
    # time_range 파라미터 적용 (float/int 안전 변환)
    limit_range = float(time_range)
    
    # Frame이 맞고, Window 순위가 범위 내인 데이터만 추출하여 통계 계산
    target_df = df[mask_frame & (df['WIN_RANK'] <= limit_range)].copy()
    
    if target_df.empty:
        # 범위 내 데이터가 없으면 컬럼 정리 후 반환
        if 'WIN_RANK' in df.columns: df.drop(columns=['WIN_RANK'], inplace=True)
        return df

    # -------------------------------------------------------------------------
    # [Loop Logic] GroupBy -> Apply로 Worst 5 Code 선정
    # -------------------------------------------------------------------------
    def get_worst_codes(group):
        # 이미 target_df는 최근 N개 Window만 필터링되어 있으므로 바로 평균 계산
        code_stats = group.groupby('CODE')['DPU'].mean()
        
        # DPU 내림차순 정렬 (Worst 선정)
        sorted_codes = code_stats.sort_values(ascending=False)
        
        # 상위 5개 코드에 대해 Rank 부여
        ranks = {}
        labels = ["1st", "2nd", "3rd", "4th", "5th"]
        
        for i, (code, dpu) in enumerate(sorted_codes.items()):
            if i < 5:
                ranks[code] = labels[i]
            else:
                break 
        return pd.Series(ranks)

    # Group별 Top 5 선정
    # 결과: Index=[FACTORY, PRODUCT, PROCESS, CODE], Value=Rank_Label
    rank_mapping = target_df.groupby(group_keys).apply(get_worst_codes).reset_index()
    
    # 컬럼명 정리 (마지막 컬럼이 Rank 값)
    rank_mapping.columns = group_keys + ['CODE', 'Rank_Label']

    # -------------------------------------------------------------------------
    # [Merge & Update] 조건을 만족하는 행만 업데이트
    # -------------------------------------------------------------------------
    # 1. 원본 데이터에 Rank 정보 매핑 (Left Join)
    # 이 시점에서는 범위 밖(과거) 데이터에도 Code가 같으면 Rank가 붙음
    merged_df = pd.merge(df, rank_mapping, on=group_keys + ['CODE'], how='left')
    
    # 2. [수정 요청 사항 반영] 조건부 업데이트
    # 조건 1: 사용자가 선택한 Time Frame 일 것
    # 조건 2: Rank가 존재할 것 (Top 5 안에 들었을 것)
    # 조건 3: [핵심] 최근 Window 범위(time_range) 이내일 것 (과거 데이터 제외)
    
    condition = (
        (merged_df['WINDOWFRAME'] == time_frame) & 
        (merged_df['Rank_Label'].notna()) & 
        (merged_df['WIN_RANK'] <= limit_range)
    )
    
    # 조건 만족 시 Rank 부여, 아니면 'ignore' 유지
    merged_df['DPU_RANK'] = np.where(condition, merged_df['Rank_Label'], 'ignore')
    
    # 3. 불필요 컬럼 제거 (WIN_RANK, Rank_Label)
    cols_to_drop = ['WINDOW', 'PROCESS_CODE', 'CODE', 'DEFECT_QTY', 'GLS_COUNT', 'DPU', 'WINDOWFRAME', 'FACTORY', 'PRODUCT', 'PROCESS_DESC''Rank_Label', 'WIN_RANK']
    # WIN_RANK는 위에서 생성했으므로 존재하지만 안전하게 확인
    actual_drop = [c for c in cols_to_drop if c in merged_df.columns]
    final_df = merged_df.drop(columns=actual_drop)
    
    return final_df

# ==========================================
# Spotfire Entry Point
# ==========================================
if __name__ == "__main__":
    pass
else:
    # Spotfire 환경 실행
    output_df = calculate_dpu_rank(input_df, 'DATE', 2)
}