import pandas as pd
import base64
from PIL import Image
import io
import os
from tqdm import tqdm

# Linux 클라우드 환경 경로 설정
BASE_DIR = "/data_home/user/2025/username/Python"
INPUT_FILE = os.path.join(BASE_DIR, "Encoded_Data.xlsx")  # Base64가 포함된 엑셀 파일명으로 수정 필요
OUTPUT_DIR = os.path.join(BASE_DIR, "Decoded_Images")

# 저장할 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base64 문자열이 들어있는 컬럼명 리스트
TARGET_COLS = [
    '검사_Image_Base64',
    'FIB_Image_Base64',
    'EDS_Image_Base64',
    'Ion_mapping_Base64'
]

def decode_base64_to_image(base64_str, save_path):
    """Base64 문자열을 JPEG 이미지로 디코딩하여 저장"""
    if pd.isna(base64_str) or not isinstance(base64_str, str):
        return False
        
    # "data:image/jpeg;base64," 등 데이터 URI 스킴 접두사 제거
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
        
    try:
        img_data = base64.b64decode(base64_str)
        with Image.open(io.BytesIO(img_data)) as img:
            # RGB 모드가 아닐 경우 변환 (JPEG 저장을 위함)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(save_path, format="JPEG")
        return True
    except Exception as e:
        raise Exception(f"디코딩 실패: {e}")

# 1. 엑셀 파일 읽기
print(f"데이터 로드 중: {INPUT_FILE}")
df = pd.read_excel(INPUT_FILE)

# 2. 행 단위로 순회하며 이미지 디코딩
total_extracted = 0

for index, row in tqdm(df.iterrows(), total=len(df), desc="이미지 디코딩 진행도"):
    for col in TARGET_COLS:
        if col in df.columns:
            base64_str = row[col]
            
            if pd.notna(base64_str) and str(base64_str).strip() != "":
                # 고유한 파일명 생성 (예: row0_검사_Image_Base64.jpg)
                # 필요시 row['특정_ID_컬럼'] 등을 파일명에 활용할 수 있습니다.
                file_name = f"row{index}_{col}.jpg"
                save_path = os.path.join(OUTPUT_DIR, file_name)
                
                try:
                    is_saved = decode_base64_to_image(base64_str, save_path)
                    if is_saved:
                        total_extracted += 1
                except Exception as e:
                    print(f"\n[오류] 행 {index}, 컬럼 {col} 처리 중 문제 발생: {e}")

print("-" * 50)
print(f"디코딩 완료. 총 {total_extracted}장의 이미지가 다음 경로에 저장되었습니다:")
print(OUTPUT_DIR)