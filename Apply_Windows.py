import pandas as pd
from datetime import timedelta

# 1. 윈도우 설정 정의 (입력받은 기준 활용)
WINDOWS_CONFIG = {
    '01W': 7,
    '02W': 14,
    '03W': 21,
    '04W': 28,
    '05W': 35
}

def calculate_windows_column(df):
    # 2. TIMESTAMP 데이터 타입 확인 및 기준일(Max Date) 설정
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    # 시간 정보를 제외한 날짜 정규화 (00:00:00)
    max_date = df['TIMESTAMP'].max().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # 3. 각 행별로 포함되는 WINDOWS 리스트 계산
    window_results = []
    
    for ts in df['TIMESTAMP']:
        # 시간 정보를 제외한 날짜로 계산
        current_date = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        
        best_win_name = ""
        min_duration = float('inf')

        for win_name, days in WINDOWS_CONFIG.items():
            start_date = max_date - timedelta(days=(days - 1))
            # 현재 TIMESTAMP가 해당 윈도우 기간 내에 있는지 확인
            if start_date <= current_date <= max_date:
                if days < min_duration:
                    min_duration = days
                    best_win_name = win_name
        
        window_results.append(best_win_name)
    
    df['WINDOWS'] = window_results
    return df

if __name__ == "__main__":
    # 4. 데이터 적용 (Spotfire에서 전달받은 데이터프레임 변수명 사용)
    # rawdata는 Spotfire 데이터 함수의 입력 파라미터 이름으로 가정함
    # For testing purposes, try reading if file exists, else skip
    try:
        df = pd.read_excel('timestamp.xlsx')
        rawdata = calculate_windows_column(df)
        print(rawdata.head())
        
        # 만약 한 행이 여러 윈도우에 속할 때 행을 분리(Expansion)해야 한다면 아래 코드 추가
        # rawdata['WINDOWS'] = rawdata['WINDOWS'].str.split(',')
        # rawdata = rawdata.explode('WINDOWS').reset_index(drop=True)
    except FileNotFoundError:
        print("timestamp.xlsx not found. Run this script directly to test with the excel file.")
