Output : 주간 단위의 DPU 변동을 설명하는 LINE을 표시

계산된 컬럼 미리 만들기 : Concat(Year,"-",Week)

Table1
    Data_Origin: Func_Glass_History
    Name: df_table_01
    Purpose: 사용자가 보는 PPT 지표와 관점 일치
    Groupby: Model, Code, Window(Spotfire)
    DPU: sum(Defect_Qty) / count(Glass_ID)
    Filter:
        [Model, Code]별 Window(Spotfire) 최신 5개까지
        [Model, Code]별 Glass_ID count 300 이상

Table2
    Data_Origin: Table1
    Name: df_table_02
    Purpose: Table1에서 분석할 구간만 sort
    ANALYSIS_NO: Table1에서 [Model, Code]별 Window(Spotfire)를 2개 Index마다 비교
        Abs(index[n].value(DPU) - index[n-1].value(DPU)) >= 0.1이 True
        index[n], index[n-1]을 df_table_02.append(), 고유 문자열 부여(ANALYSIS_01)

Table3
    Data_Origin: Func_Glass_History
    Name: df_table_03
    Purpose: Table2를 참조하여 ANALYSIS_NO마다 변동 설명력이 가장 높은 LINE을 도출
        참조 Data: Model, Code, Window(Spotfire) -> ANALYSIS_NO마다 2개 row씩 읽게 됨
    Groupby:
        Model, Code, Process, Line, Window(Spotfire) -> n
        Model, Code, Process, Line, Window(Spotfire) -> n-1
        DPU: Table2의 DPU 변동을 가장 잘 설명하는 데이터(Logic 연구 필요)

Table4
    Data_Origin: Table3
    Name: df_table_04
    Purpose: Table3이 ANALYSIS_NO마다 도출한 변동 설명 Line을 수집
    Output: .append(Line)