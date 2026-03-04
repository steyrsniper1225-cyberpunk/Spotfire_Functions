import pandas as pd
import numpy as np
import os

# --------------------------------------------------------------------------------
# Helper Functions for Dummy Results
# --------------------------------------------------------------------------------

def get_dummy_total_trend():
    """Returns a dummy row for total_trend if analysis result is empty."""
    return pd.DataFrame({
        'OBJECT': ['Total_Trend'],
        'WINDOW': ['-'],
        'DEGREE': [0.0],
        'TREND': ['-'],
        'DESCRIPTION': ['No Data Available']
    }).astype({
        'DEGREE': 'float64'
    })

def get_dummy_each_code_trend():
    """Returns a dummy row for each_code_trend if analysis result is empty."""
    return pd.DataFrame({
        'OBJECT': ['Each_Code_Trend'],
        'CODE': ['-'],
        'CORREL': [0.0],
        'STRENGTH': ['-'],
        'DESCRIPTION': ['No Data Available']
    }).astype({
        'CORREL': 'float64'
    })

# --------------------------------------------------------------------------------
# Analysis Functions
# --------------------------------------------------------------------------------

def total_trend(df_input):
    """
    Analyzes the trend of Total DPU over the latest 6 days.
    Logic: Z-Score calculation and Linear Regression Slope.
    """
    '''
    # 1. Input Handling (Spotfire context vs Local)
    if df_input is None:
        try:
            # Check for Spotfire common variable name 'InputTable'
            if 'InputTable' in globals():
                df_input = globals()['InputTable']
            else:
                df_input = pd.read_excel('Util_Generator_PivotTrend_Result.xlsx')
        except Exception:
            return get_dummy_total_trend()
    '''
    # 2. Filter: Latest 6 Days for TOTAL Line
    df = df_input[(df_input['WINDOWFRAME'] == 'DATE') & (df_input['LINE'] == 'TOTAL')].copy()
    if df.empty:
        return get_dummy_total_trend()

    # Aggregate by WINDOW (Sum DPU across all codes)
    df_agg = df.groupby('WINDOW', as_index=False)['DPU'].sum().sort_values('WINDOW')
    df_latest = df_agg.tail(6)
    
    if len(df_latest) < 2:
        return get_dummy_total_trend()

    dpu_values = df_latest['DPU'].values
    windows = df_latest['WINDOW'].values

    # 3. Calculation - Step 1: Z-Score
    mean_val = np.mean(dpu_values)
    std_val = np.std(dpu_values)
    z_scores = (dpu_values - mean_val) / std_val if std_val > 0 else np.zeros_like(dpu_values)

    # 4. Calculation - Step 2: Linear Regression Slope (Degree)
    x = np.arange(len(z_scores))
    degree = np.polyfit(x, z_scores, 1)[0]

    # 5. Logic Evaluation
    # Trend Logic
    if degree >= 0.20:
        trend_status, trend_desc = "증가", "최근 6일간 DPU가 증가하고 있습니다"
    elif degree <= -0.20:
        trend_status, trend_desc = "감소", "최근 6일간 DPU가 감소하고 있습니다"
    else:
        trend_status, trend_desc = "유지", "최근 6일간 DPU가 유지되고 있습니다"

    # Fluctuation Logic (Difference between current and previous day Z-Score)
    fluctuated_windows = []
    # 1번째 인덱스부터 마지막까지 순회하며 이전 Window와의 Z-Score 차이 절대값을 계산
    for i in range(1, len(z_scores)):
        diff = abs(z_scores[i] - z_scores[i-1])
        if diff >= 2.0:
            fluctuated_windows.append(windows[i])

    if fluctuated_windows:
        # 변동이 감지된 모든 Window를 리스트업하여 설명에 포함
        fluct_desc = f"{', '.join(fluctuated_windows)}에서 DPU 변동이 관찰됩니다"
    else:
        fluct_desc = "정상 범위 내 변동"

    # 6. Format Result
    result = pd.DataFrame({
        'OBJECT': ['Total_Trend'],
        'WINDOW': [f"최근 {len(windows)}일"],
        'DEGREE': [degree],
        'TREND': [trend_status],
        'DESCRIPTION': [f"{trend_desc} | {fluct_desc}"]
    })
    
    return result

def each_code_trend(df_input):
    """
    Analyzes correlation between each Code and the Total Trend.
    """
    '''
    # 1. Input Handling
    if df_input is None:
        try:
            if 'InputTable' in globals():
                df_input = globals()['InputTable']
            else:
                df_input = pd.read_excel('Util_Generator_PivotTrend_Result.xlsx')
        except Exception:
            return get_dummy_each_code_trend()
    '''
    # 2. Preparation
    df = df_input[(df_input['WINDOWFRAME'] == 'DATE') & (df_input['LINE'] == 'TOTAL')].copy()
    if df.empty:
        return get_dummy_each_code_trend()

    # Get Total Sum per Window as Baseline
    total_agg = df.groupby('WINDOW')['DPU'].sum().tail(6)
    latest_windows = total_agg.index.tolist()
    total_baseline = total_agg.values
    
    if len(latest_windows) < 2:
        return get_dummy_each_code_trend()

    # 3. Loop through Codes
    codes = sorted(df['CODE'].unique())
    results = []
    
    for code in codes:
        code_data = df[df['CODE'] == code].set_index('WINDOW').reindex(latest_windows).fillna(0)
        code_values = code_data['DPU'].values
        
        # Calculate Correlation
        if np.std(code_values) == 0 or np.std(total_baseline) == 0:
            correl = 0.0
        else:
            correl = np.corrcoef(code_values, total_baseline)[0, 1]
        
        # Logic Evaluation
        strength = "High" if correl >= 0.80 else "Low"
        desc = f"최근 6일간 DPU가 Total_Trend와 {'높은' if strength == 'High' else '낮은'} 상관관계를 보입니다"
        
        results.append({
            'OBJECT': 'Each_Code_Trend',
            'CODE': code,
            'CORREL': correl,
            'STRENGTH': strength,
            'DESCRIPTION': desc
        })

    if not results:
        return get_dummy_each_code_trend()

    return pd.DataFrame(results)

# --------------------------------------------------------------------------------
# Main Execution (for Local Testing)
# --------------------------------------------------------------------------------
df_input = input_table.copy()
res_total = total_trend(df_input)
res_each = each_code_trend(df_input)