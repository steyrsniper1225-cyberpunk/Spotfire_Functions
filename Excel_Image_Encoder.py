import pandas as pd
import openpyxl
import base64
from PIL import Image
import io
import os
from tqdm import tqdm

# Linux 클라우드 환경 경로 설정
BASE_DIR = "/data_home/user/2025/username/Python"
INPUT_FILE = os.path.join(BASE_DIR, "Original_Data.xlsx")
OUTPUT_FILE = os.path.join(BASE_DIR, "Spotfire_Ready.csv")

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

# 1. 메타데이터 로드 (pandas 2.3.2)
# A열(이미지)을 제외하고 B열부터 로드. (header=0 가정)
df = pd.read_excel(INPUT_FILE, usecols=lambda x: x not in ['A', 'Unnamed: 0'])

# 2. Excel 이미지 추출 (openpyxl 3.1.5)
wb = openpyxl.load_workbook(INPUT_FILE, data_only=True)
ws = wb.active
print("[DONE] Excel File Opening Complete")

image_mapping = {}

# ws._images에 포함된 객체에서 데이터와 위치(Row) 추출
for image in tqdm(ws._images, desc = "Encoding Image by Base64", unit = "ea"):
    try:
        # openpyxl의 이미지 anchor 행(row) 인덱스 (0부터 시작)
        row_idx = image.anchor._from.row
        
        # 이미지 바이트 데이터 읽기
        img_data = image.ref.getvalue()
        
        # Base64 변환
        base64_str = convert_image_to_base64(img_data)
        
        # 첫 행(row=0)이 컬럼명일 경우 데이터의 인덱스는 row_idx - 1이 됨
        df_idx = row_idx - 1
        image_mapping[df_idx] = base64_str
    except Exception as e:
        print(f"\nRow {row_idx} 이미지 처리 오류: {e}")

wb.close()
print("[DONE] Excel File Closed")

# 3. DataFrame에 Base64 이미지 컬럼 매핑
df['검사_Image_Base64'] = df.index.map(image_mapping)
print("[DONE] Base64 Encoding Complete")

# 4. Spotfire 로드용 CSV 저장
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"완료. 파일 저장 위치: {OUTPUT_FILE}")
