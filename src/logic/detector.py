import json
import io
import os
from PIL import Image
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# 제미나이 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('models/gemini-3-flash-preview')

# YOLO 모델 로드
YOLO_MODEL_PATH = os.path.join("models", "best.pt")
yolo_model = YOLO(YOLO_MODEL_PATH)

def get_json_for_llm(results):
    """YOLO 결과를 제미나이에 보낼 JSON 형식으로 변환 (픽셀 좌표)"""
    yolo_data = []
    for r in results:
        names = r.names
        for box in r.boxes:
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            label = names[int(box.cls[0])]
            yolo_data.append({
                "box_2d": coords,
                "label": label
            })
    return yolo_data

def format_flower_list_for_db(analysis_result):
    """
    DB의 'flower_list' 컬럼에 들어갈 문자열 생성
    예: "rose 2개, tulip 1개"
    """
    counts = {}
    for item in analysis_result:
        name = item.get('corrected_name', 'unknown')
        counts[name] = counts.get(name, 0) + 1
    
    return ", ".join([f"{name} {count}개" for name, count in counts.items()])

async def analyze_flower_ensemble(image_bytes):
    """
    이미지를 분석하여 YOLO + Gemini 앙상블 결과 반환
    - 프론트용: 상세 JSON (좌표 포함)
    - DB용: 요약 문자열 (함수 별도 호출)
    """
    
    # 사용자가 업로드한 바이너리 데이터를 PIL 이미지로 로드
    img_pil = Image.open(io.BytesIO(image_bytes))
    
    # YOLOv8 추론
    results = yolo_model.predict(source=img_pil, conf=0.15)
    yolo_json = get_json_for_llm(results)

    # 프롬프트 구성
    prompt = f"""
    역할: 너는 20년 경력의 수석 플로리스트이자 시각 지능 전문가야.

    입력 데이터:
    1. 사진: 부케 이미지
    2. YOLO JSON (기존 감지 데이터): {json.dumps(yolo_json, indent=2, ensure_ascii=False)}

    임무:
    1. 제공된 YOLO JSON의 **박스 좌표(box_2d)는 1픽셀도 수정하지 마라.**
    2. [정밀 교정]: 해당 박스의 꽃 이름이 정확한지 전문가의 눈으로 검수하고 교정하라. 특히 'rose'와 'ranunculus'를 구분하라.
    3. [신규 발견]: YOLO가 박스를 치지 못한 영역을 샅샅이 탐색하여, 놓친 중요한 꽃들을 모두 찾아내라. **좌표 규칙**: 신규 꽃 좌표는 이미지 전체 크기를 [0, 0, 1000, 1000]으로 가정한 **정규화된 좌표(Normalized Coordinates)** [x1, y1, x2, y2]로 생성하라.
    4. [고유 품종 필터링]: 품종당 가장 형태가 분명한 '대표 꽃 하나'만 남겨라. 결과 리스트에는 품종당 단 하나의 객체만 존재해야 한다.

    출력 규칙 (반드시 지킬 것):
    - 반드시 JSON 리스트 형식으로만 응답할 것.
    - 텍스트 설명이나 인사는 절대로 포함하지 말 것.
    - 각 객체는 반드시 아래 필드를 포함할 것:
    - 'location': [x1, y1, x2, y2] 좌표
    - 'corrected_name': 네가 판단한 최종 꽃 이름 (영문)
    - 'is_new': YOLO가 놓친 꽃이면 true, 아니면 false
    - 'reason': 해당 품종으로 판단한 근거 (한국어, 짧게)

    응답 예시:
    [
    {{
        "location": [100.1, 200.5, 150.2, 250.8],
        "corrected_name": "ranunculus",
        "is_new": false,
        "reason": "꽃잎이 습자지처럼 얇고 겹이 촘촘하며 폼폰 형태를 띠고 있어 라넌큘러스로 판단됨"
    }}
    ]
    """

    try:
        # 제미나이 실행 (이미지 객체와 프롬프트 전달)
        response = gemini_model.generate_content([prompt, img_pil])
        
        # JSON 파싱
        json_str = response.text.strip().replace("```json", "").replace("```", "")
        corrected_data = json.loads(json_str)
        
        return corrected_data

    except Exception as e:
        """
        [Fallback 로직] 제미나이 API 실패 시 서버 중단을 방지하기 위해 
        YOLO가 찾은 기본 정보라도 프론트/DB 규격에 맞춰 반환
        """
        print(f"앙상블 분석 중 에러 발생(YOLO 결과로 대체): {e}")
        
        fallback_result = []
        for item in yolo_json:
            fallback_result.append({
                "location": item["box_2d"],
                "corrected_name": item["label"],
                "is_new": False,
                "reason": "AI 분석 일시적 장애로 인한 YOLO 기본 감지 결과입니다."
            })
        return fallback_result