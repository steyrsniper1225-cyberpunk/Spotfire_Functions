import pandas as pd
import openpyxl
import base64
from PIL import Image
import io
import os
import glob
from tqdm import tqdm

# Linux 클라우드 환경 경로 설정
BASE_DIR = "/data_home/user/2025/username/Python"
OUTPUT_FILE = os.path.join(BASE_DIR, "Spotfire_Ready_All.csv")

def convert_image_to_base64(img_data, target_size=(800, 800)):
    """openpyxl에서 추출한 이미지 바이트 데이터를 Base64 문자열로 변환"""
    with Image.open(io.BytesIO(img_data)) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail(target_size)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_str}"

# BASE_DIR 내의 모든 xlsx 파일 검색 (열려있는 임시 파일 제외)
xlsx_files = [f for f in glob.glob(os.path.join(BASE_DIR, "*.xlsx")) if not os.path.basename(f).startswith('~$')]

all_dfs = []

for file_path in xlsx_files:
    file_name = os.path.basename(file_path)
    print(f"처리 중: {file_name}")
    
    # 1. 메타데이터 로드
    df = pd.read_excel(file_path, usecols=lambda x: x not in ['A', 'Unnamed: 0'])
    
    # 2. Excel 이미지 추출
    wb = openpyxl.load_workbook(file_path, data_only=True)
    ws = wb.active
    
    image_mapping = {}
    
    # tqdm을 사용하여 파일별 진행도 표시
    for image in tqdm(ws._images, desc=f"변환 진행도", unit="장"):
        try:
            row_idx = image.anchor._from.row
            img_data = image.ref.getvalue()
            base64_str = convert_image_to_base64(img_data)
            df_idx = row_idx - 1
            image_mapping[df_idx] = base64_str
        except Exception as e:
            print(f"\n[{file_name}] Row {row_idx} 이미지 처리 오류: {e}")
            
    wb.close()
    
    # 3. DataFrame 매핑 및 출처 추적용 컬럼 추가
    df['검사_Image_Base64'] = df.index.map(image_mapping)
    df['Source_File'] = file_name
    
    all_dfs.append(df)
    print("-" * 50)

# 4. 모든 DataFrame 병합 및 CSV 저장
if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"모든 파일 병합 완료. 총 {len(final_df)}행 저장 위치: {OUTPUT_FILE}")
else:
    print("처리할 .xlsx 파일이 없습니다.")