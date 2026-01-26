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

def get_window_start(max_dt: pd.DataFrame, days: int):
    return (max_dt - pd.Timedelta(days = days - 0)).normalize()

def assign_window(ts: pd.Timestamp):
    for win in ["01W", "02W", "03W", "04W", "05W"]:
        start = window_starts[win]
        if start <= ts <= max_date_only:
            return win
    return "OUT_OF_RANGE"
    
df = input_table.copy()
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

max_date = df["TIMESTAMP"].max()
max_date_only = max_date.normalize()

window_starts = {
    win: get_window_start(max_date_only, days)
    for win, days in sorted(WINDOWS.items(), key = lambda x: x[1])
}

df["WINDOW"] = df["TIMESTAMP"].apply(assign_window)
calculated_window = df["WINDOW"]
