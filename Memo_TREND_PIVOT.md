{
    "Object" : "Total_Trend",
    "Purpose" : "최근 6일간 Total DPU의 추세 분석",
    "Window" : {
        "WINDOWFRAME" : "DATE",
        "RANGE" : "Latest 6 Days"
    },
    "LINE" : "TOTAL",
    "CODE" : "All Codes",
    "Value" : Sum(DPU),
    "Method" : {
        "Calculation" : {
            "Step1_Mov_Z_Score" : "Calculate Z-Score for each {WINDOW}",
            "Step2_Line_Reg" : "Z-Score를 이용한 Line Regression 기울기(degree) 계산"
        },
        "Logic1" : {
            "증가" : {
                "조건" : "degree >= 0.20",
                "설명" : "{WINDOW}간 {Value}가 증가하고 있습니다"
            },
            "감소" : {
                "조건" : "degree <= -0.20",
                "설명" : "{WINDOW}간 {Value}가 감소하고 있습니다"
            },
            "유지" : {
                "조건" : "Abs(degree) < 0.20",
                "설명" : "{WINDOW}간 {Value}가 유지되고 있습니다"
            }
        },
        "Logic2" : {
            "변동" : {
                "전일" : "Previous Day, WINDOW[n-1]",
                "당일" : "Current Day, WINDOW[n]",
                "조건" : "전일 대비 당일 Z-Score의 절대값의 차이가 2.0 이상인 경우",
                "설명" : "{WINDOW[n]}에서 {Value} 변동이 관찰됩니다"
            }
        }
    }
},
{
    "Object" : "Each_Code_Trend",
    "Purpose" : "Total_Trend와 상관관계를 계산하기 위한 각 Code별 추세 분석",
    "Window" : {
        "WINDOWFRAME" : "DATE",
        "RANGE" : "Latest 6 Days"
    },
    "LINE" : "TOTAL",
    "CODE" : ["PLN1-FM", "PLN1-SM", "PLN2-FM", "PLN2-SM"],
    "Value" : Sum(DPU),
    "Method" : {
        "Correlation" : {
            "Group1" : "Total_Trend",
            "Group2" : "Each_Code_Trend",
            "Calculation" : "Total_Trend와 Each_Code_Trend의 상관계수(correl) 계산"
        },
        "Logic1" : {
            "High" : {
                "조건" : "correl >= 0.80",
                "설명" : "{WINDOW}간 {Value}가 {Group1}와 높은 상관관계를 보입니다"
            },
            "Low" : {
                "조건" : "correl < 0.80",
                "설명" : "{WINDOW}간 {Value}가 {Group1}와 낮은 상관관계를 보입니다"
            }
        }
    }
}
