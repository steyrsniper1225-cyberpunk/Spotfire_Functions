{
    "Function_Concept_Define_Note" : {
        "질문자" : "관리자(상사)",
        "질문" : {
            "변동" : "DPU 변동 원인은 무엇인가",
            "수준" : "DPU가 Target_Line과 Gap이 발생하는 원인은 무엇인가"
        },
        "요구 Output" : "원인 설비, 자재(Lot/Glass), 공정 조건, 환경 변화 등",
        "한계" : {
            "한계1" : "Python Script만으로는 통합적인(또는 맥락을 가진) Output 도출 불가",
            "한계2" : "공장 Layout이나 사례 등 background knowledge는 Spotfire 구현 불가",
            "한계3" : "Spotfire에 구현된 DPU Data에만 접근 가능"
        },
        "엔지니어의 현행 분석 업무 관찰" : {
            "변동" : {
                "정의" : "수치로 정의된 바 없이 know-how를 기반으로 주관적인 판단에 의거",
                "관점" : "baseline을 고려하지 않고 DPU delta에만 집중, 평균보다는 산포에 초점",
                "TimeFrame" : {
                    "Month" : "근본적인 설비의 변화(공정 조건, 물동), Map 변화",
                    "Week" : "점검/유지보수가 필요한 설비(LINE/MACHINE)의 탐색, Map 변화",
                    "Date" : "일시적 이상, 자재 issue(이상발생, Lot 이력 추적 등), 특이 Map"
                },
                "설비" : {
                    "선정 방법" : {
                        "#1" : "해당 DPU의 CODE에 mapping된 Layer 파악",
                        "#2" : "해당 Layer별로 mapping된 설비(LINE) 파악",
                        "#3" : "전체 설비 Trend vs. Layer/공정별 설비(LINE) Trend 상관성 검토",
                        "#4" : "상관성이 가장 높은 단일 설비(LINE)로 분석 범위 좁힘",
                        "#5" : "설비(LINE) 특정되지 않으면 물동량이 많은(1st, 2nd) 설비(LINE)로 분석 범위 좁힘"
                    },
                    "단위" : {
                        "LINE" : "상관성이 높아 선정되었거나 물동량이 많아 선정된 설비(LINE)",
                        "MACHINE" : "선정된 설비(LINE)에 포함된 세부 설비(MACHINE)"
                    }
                },
                "Parameter" : "선정된 설비(LINE/MACHINE)기준 DPU 변동과 상관관계 보이는 모든 Parameter 조사",
                "Map" : "선정된 설비(LINE/MACHINE)기준 Map 변화 여부 확인",
                "물동량" : "변동 전/후 LINE 기준 물동량이 변화했는지 확인",
                "소결론" : "변동을 가장 잘 설명하는 핵심 요소를 기반으로 보고 Story 작성"
            },
            "수준" : {
                "정의" : "수치로 정의된 바 없이 know-how를 기반으로 주관적인 판단에 의거",
                "관점" : "두 Group(보통 Best vs. Worst)의 평균치의 차이에 집중, 산포보다는 평균치의 차이에 초점",
                "TimeFrame" : "Month 또는 최근 5 Weeks 위주로 사용, 근본적인 설비의 변화(공정 조건, 물동), Map 차이"
                "설비" : {
                    "선정 방법" : {
                        "#1" : "비교하려는 두 Group에 대해 CODE, LINE이 지정되어 있음",
                        "#2" : "",
                        "#3" : "",
                        "#4" : "",
                        "#5" : ""
                    },
                    "단위" : {
                        "LINE" : "상관성이 높아 선정되었거나 물동량이 많아 선정된 설비(LINE)",
                        "MACHINE" : "선정된 설비(LINE)에 포함된 세부 설비(MACHINE)"
                    }
                },
                "Parameter" : "선정된 설비(LINE/MACHINE)기준 DPU 변동과 상관관계 보이는 모든 Parameter 조사",
                "Map" : "선정된 설비(LINE/MACHINE)기준 Map 변화 여부 확인",
                "물동량" : "변동 전/후 LINE 기준 물동량이 변화했는지 확인",
                "소결론" : "변동을 가장 잘 설명하는 핵심 요소를 기반으로 보고 Story 작성"
            }
        },
        "DataFunction 작성 전략" : {
            "변동 분석" : {
                "요소" : "변동 정의, Timeframe 설정, 설비/Parameter/Map/물동량 검토, 소결론 작성",
                "전략" : "각 요소에 대해 병렬로 분석하는 소규모 함수를 작성하고, 결과를 통합하는 함수를 작성"
            },
            "수준 분석" : {
                "요소" : "",
                "전략" : ""
            }
        }
    }
},
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
},
{
    "Object" : "Overshooting_Day_Analysis",
    "Purpose" : "Overshooting Day를 설명하기 위한 분석",
    "Window" : {
        "WINDOWFRAME" : "DATE",
        "RANGE" : "Latest 6 Days"
    },
    "LINE" : "non Total, PHTxx or OVNxx 개별 호기",
    "CODE" : ["PLN1-SM", "PLN2-SM"],
    "Value" : Sum(DPU),
    "Method" : {
        "Overshooting_Day" : {
            "Definition" : "each day's Current_DPU > Target_DPU",
            "Target_DPU" : {
                "PLN1-SM" : 0.50,
                "PLN2-SM" : 0.62
            },
            "Filter" : "Glass 수량 >= 10"
        },
        "PHT_Analysis" : {
            "Port" : "최근 2주 내의 Port 유의차 확인",
            "CST_Slot" : "최근 2주 내의 CST Slot 경향성 여부 확인",
            "Unit" : "analyze_photo_unit_anomalies() 함수 실행",
            "FDC_Trend" : "최근 2주 내의 Motor Load Trend 변동 여부 확인",
            "Issue_Glass" : "이상발생통보 ID를 갖고 있는 Glass 출력"
        },
        "OVN_Analysis" : {
            "Port" : "최근 2주 내의 Port 유의차 확인",
            "CST_Slot" : "최근 2주 내의 CST Slot 경향성 여부 확인",
            "Unit" : "최근 2주 내의 Chamber, OvenSlot, CoolSlot 경향성 여부 확인",
            "FDC_Trend" : "최근 2주 내의 FDC Trend 변동 여부 확인",
            "Issue_Glass" : "이상발생통보 ID를 갖고 있는 Glass 출력"
        }
    }
}
