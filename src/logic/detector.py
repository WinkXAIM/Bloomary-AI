import json
import io
import os
from PIL import Image, ImageOps
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv
import re
import asyncio
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

load_dotenv()

# 제미나이 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('models/gemini-3-flash-preview')

# YOLO 모델 로드
YOLO_MODEL_PATH = os.path.join("models", "best.pt")
yolo_model = YOLO(YOLO_MODEL_PATH)

# ============================================================
# YOLO 모델 88개 클래스 기준: 영문 클래스명 -> 국문명
# ============================================================

FLOWER_NAME_KO_MAP = {
    "alstroemeria": "알스트로메리아",
    "amaryllis": "아마릴리스",
    "anthurium": "안스리움",
    "azalea": "철쭉",
    "bee_balm": "베르가못",
    "bellflower": "초롱꽃",
    "blackberry_lily": "범부채",
    "blanket_flower": "천인국",
    "bougainvillea": "부겐빌레아",
    "bromeliad": "브로멜리아드",
    "calla_lily": "카라",
    "camellia": "동백",
    "canna_lily": "칸나",
    "canterbury_bells": "캔터베리벨",
    "cape_flower": "케이프플라워",
    "carnation": "카네이션",
    "cattleya": "카틀레야",
    "celosia": "맨드라미",
    "chamomile": "캐모마일",
    "chrysanthemum": "국화",
    "clematis": "클레마티스",
    "columbine": "매발톱꽃",
    "coneflower": "에키네시아",
    "cosmos": "코스모스",
    "cyclamen": "시클라멘",
    "daffodil": "수선화",
    "dahlia": "달리아",
    "daisy": "데이지",
    "desert_rose": "석화",
    "doraji": "도라지꽃",
    "eryngo": "에린지움",
    "feather_celosia": "깃털맨드라미",
    "foxglove": "디기탈리스",
    "freesia": "프리지아",
    "fritillaria": "프리틸라리아",
    "garden_phlox": "풀협죽도",
    "gaura": "가우라",
    "gazania": "가자니아",
    "gentian": "용담",
    "geranium": "제라늄",
    "gladiolus": "글라디올러스",
    "globe_thistle": "에키놉스",
    "gloriosa_lily": "글로리오사",
    "gyeongyeopduran": "겹엽두란",
    "gypsophila": "안개꽃",
    "hellebores": "헬레보어",
    "hibiscus": "무궁화",
    "hyacinth": "히아신스",
    "hydrangea": "수국",
    "iris": "아이리스",
    "ixora": "익소라",
    "japanese_anemone": "추명국",
    "kalanchoe": "칼랑코에",
    "lily": "백합",
    "lisianthus": "리시안셔스",
    "magnolia": "목련",
    "marigold": "메리골드",
    "masterwort": "마스터워트",
    "mexican_petunia": "멕시칸 페튜니아",
    "monkshood": "투구꽃",
    "morning_glory": "나팔꽃",
    "mulmangcho": "물망초",
    "nasturtium": "한련화",
    "nigella": "니겔라",
    "orchid": "난초",
    "osteospermum": "오스테오스퍼멈",
    "pansy": "팬지",
    "passion_flower": "시계꽃",
    "peony": "작약",
    "petunia": "페튜니아",
    "pincushion_flower": "스카비오사",
    "pink_primrose": "분홍달맞이꽃",
    "plumeria": "플루메리아",
    "poinsettia": "포인세티아",
    "poppy": "양귀비",
    "primula": "프리뮬라",
    "red_ginger": "레드진저",
    "rose": "장미",
    "sampaguita": "삼파귀타",
    "silverbush": "실버부시",
    "spring_crocus": "봄크로커스",
    "stock": "스토크",
    "sunflower": "해바라기",
    "sweet_pea": "스위트피",
    "sweet_william": "패랭이꽃",
    "trumpet_creeper": "능소화",
    "tulip": "튤립",
    "wallflower": "월플라워",
}

# ============================================================
# 이미지 로드
# ============================================================

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    업로드된 이미지 bytes를 PIL RGB 이미지로 변환
    휴대폰 사진 EXIF 회전 정보도 반영
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


# ============================================================
# 이름 처리
# ============================================================

def resolve_name_ko(name_en: str, gemini_name_ko: str = "") -> str:
    """
    국문명 결정
    1순위: YOLO 88개 클래스 매핑
    2순위: Gemini가 준 name_ko
    3순위: 빈 문자열
    """
    name_en = str(name_en or "").strip()
    gemini_name_ko = str(gemini_name_ko or "").strip()

    if name_en in FLOWER_NAME_KO_MAP:
        return FLOWER_NAME_KO_MAP[name_en]

    return gemini_name_ko


# ============================================================
# Gemini 응답 JSON 파싱
# ============================================================

def parse_gemini_json_list(text: str) -> List[Dict[str, Any]]:
    """
    Gemini 응답에서 JSON 리스트만 추출.
    """
    if not text:
        return []

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return []

    data = json.loads(match.group(0))

    if not isinstance(data, list):
        return []

    return data


# ============================================================
# YOLO 결과를 Gemini 프롬프트용 JSON으로 변환
# ============================================================

def get_json_for_llm(results: Any) -> List[Dict[str, Any]]:
    """
    Ultralytics YOLO 결과를 Gemini에게 넘길 JSON 형태로 변환.

    반환 예:
    [
        {
            "label": "rose",
            "box_2d": [100.1, 200.5, 300.2, 400.8],
            "confidence": 0.87
        }
    ]
    """
    yolo_json = []

    for result in results:
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", None)

        if boxes is None or names is None:
            continue

        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            class_id = int(box.cls[0].detach().cpu().item())

            confidence = None
            if getattr(box, "conf", None) is not None:
                confidence = float(box.conf[0].detach().cpu().item())

            label = str(names[class_id])

            yolo_json.append({
                "label": label,
                "box_2d": [
                    float(xyxy[0]),
                    float(xyxy[1]),
                    float(xyxy[2]),
                    float(xyxy[3]),
                ],
                "confidence": confidence,
            })

    return yolo_json


# ============================================================
# 좌표 처리
# ============================================================

def clip_box2d(
    box2d: List[float],
    image_width: int,
    image_height: int,
) -> Optional[List[int]]:
    """
    bbox를 이미지 범위 안으로 보정하고 int 좌표로 변환.
    """
    if not box2d or len(box2d) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in box2d]
    except (TypeError, ValueError):
        return None

    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


# ============================================================
# 대표 색상 HEX 추출
# ============================================================

def bgr_to_hex(bgr: np.ndarray) -> str:
    b, g, r = [int(v) for v in bgr]
    return f"#{r:02X}{g:02X}{b:02X}"


def extract_dominant_flower_color_hex(
    image_pil: Image.Image,
    box2d: List[int],
    k: int = 4,
) -> Optional[str]:
    """
    bbox 내부에서 대표 꽃 색상을 HEX로 추출.

    방식:
    1. bbox crop
    2. 초록색 계열 제거해서 줄기/잎 영향 줄이기
    3. 너무 어두운 픽셀 제거
    4. 남은 픽셀에 k-means 적용
    5. 가장 큰 클러스터의 중심색을 HEX로 반환
    """
    image_rgb = np.array(image_pil.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    image_height, image_width = image_bgr.shape[:2]

    clipped_box = clip_box2d(
        box2d=box2d,
        image_width=image_width,
        image_height=image_height,
    )

    if clipped_box is None:
        return None

    x1, y1, x2, y2 = clipped_box
    crop = image_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop_h, crop_w = crop.shape[:2]

    # bbox 가장자리의 배경/포장지 영향을 약간 줄임
    margin_x = int(crop_w * 0.05)
    margin_y = int(crop_h * 0.05)

    if crop_w - 2 * margin_x > 5 and crop_h - 2 * margin_y > 5:
        crop = crop[margin_y:crop_h - margin_y, margin_x:crop_w - margin_x]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # OpenCV HSV 기준 H: 0~179
    # 초록색 계열 제거: 줄기/잎 제외 목적
    green_mask = (
        (h >= 35) & (h <= 90) &
        (s >= 35) &
        (v >= 35)
    )

    # 너무 어두운 그림자 영역 제거
    dark_mask = v < 35

    valid_mask = (~green_mask) & (~dark_mask)
    valid_pixels = crop[valid_mask]

    # 초록 제거 후 픽셀이 너무 적으면 어두운 픽셀만 제외하고 fallback
    if len(valid_pixels) < 30:
        valid_pixels = crop[~dark_mask]

    if len(valid_pixels) < 30:
        return None

    # 속도 최적화용 샘플링
    max_pixels = 8000
    if len(valid_pixels) > max_pixels:
        indices = np.random.choice(len(valid_pixels), max_pixels, replace=False)
        valid_pixels = valid_pixels[indices]

    data = np.float32(valid_pixels)

    actual_k = min(k, len(data))
    if actual_k <= 0:
        return None

    cv2.setRNGSeed(42)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1.0,
    )

    _, labels, centers = cv2.kmeans(
        data,
        actual_k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.flatten()
    counts = np.bincount(labels)

    dominant_idx = int(np.argmax(counts))
    dominant_bgr = centers[dominant_idx].astype(np.uint8)

    return bgr_to_hex(dominant_bgr)


# ============================================================
# YOLO + Gemini 앙상블
# ============================================================

async def analyze_flower_ensemble(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    YOLO 1차 탐지 후 Gemini로 꽃 이름 교정 및 누락 꽃 탐지.

    Gemini 응답 형태:
    [
        {
            "name_ko": "장미",
            "name_en": "rose",
            "box2d": [100, 200, 300, 400]
        }
    ]

    주의:
    - name_en은 YOLO 모델 클래스명 기준으로 받음.
    - 모든 box2d는 최종 API 응답에 바로 사용할 실제 이미지 픽셀 좌표로 받음.
    """
    img_pil = load_image_from_bytes(image_bytes)
    image_width, image_height = img_pil.size

    results = yolo_model.predict(source=img_pil, conf=0.15)
    yolo_json = get_json_for_llm(results)

    class_names = list(FLOWER_NAME_KO_MAP.keys())

    prompt = f"""
역할: 너는 20년 경력의 수석 플로리스트이자 시각 지능 전문가야.

입력 데이터:
1. 사진: 부케 이미지
2. 이미지 크기:
    - width: {image_width}
    - height: {image_height}
3. YOLO JSON 기존 감지 데이터:
{json.dumps(yolo_json, indent=2, ensure_ascii=False)}

사용 가능한 꽃 클래스 목록:
{json.dumps(class_names, ensure_ascii=False)}

임무:
1. 제공된 YOLO JSON의 "box_2d" 좌표는 1픽셀도 수정하지 마라.
2. 정밀 교정:
    - 각 박스의 꽃 이름이 정확한지 전문가의 눈으로 검수하고 교정하라.
    - name_en은 반드시 "사용 가능한 꽃 클래스 목록" 중 하나를 그대로 사용하라.
3. 고유 품종 필터링:
    - 품종당 가장 형태가 분명한 대표 꽃 하나만 남겨라.
    - 결과 리스트에는 품종당 단 하나의 객체만 존재해야 한다.
4. 국문명:
    - name_ko는 해당 꽃의 일반적인 국문명을 작성하라.

출력 규칙:
- 반드시 JSON 리스트 형식으로만 응답할 것.
- 텍스트 설명, 인사, 마크다운 코드블록은 절대로 포함하지 말 것.
- 각 객체는 반드시 아래 필드를 포함할 것:
    - "name_ko": 최종 꽃 이름, 국문
    - "name_en": 최종 꽃 이름, 영문 클래스명
    - "box2d": YOLO JSON의 "box_2d" 좌표를 그대로 사용한 [x1, y1, x2, y2]

응답 예시:
[
    {{
        "name_ko": "장미",
        "name_en": "rose",
        "box2d": [100, 200, 300, 400]
    }}
]
"""

    try:
        response = await asyncio.to_thread(
            gemini_model.generate_content,
            [prompt, img_pil],
        )

        corrected_data = parse_gemini_json_list(response.text)

        if not corrected_data:
            raise ValueError("Gemini 응답에서 유효한 JSON 리스트를 찾지 못했습니다.")

        return corrected_data

    except Exception as e:
        """
        Fallback:
        Gemini API 실패 시 서버 중단 방지.
        YOLO 기본 결과라도 API 응답 생성이 가능하도록 반환.
        """
        print(f"앙상블 분석 중 에러 발생. YOLO 결과로 대체: {e}")

        fallback_result = []

        for item in yolo_json:
            name_en = str(item.get("label", "")).strip()
            name_ko = resolve_name_ko(name_en)

            fallback_result.append({
                "name_ko": name_ko,
                "name_en": name_en,
                "box2d": item.get("box_2d", []),
            })

        return fallback_result


# ============================================================
# /ai/detect-flowers API에서 호출할 최종 함수
# ============================================================

async def detect_flowers_for_api(image_bytes: bytes) -> Dict[str, List[Dict[str, Any]]]:
    """
    /ai/detect-flowers API 응답 형태로 반환하는 최종 로직.

    현재 포함:
    - YOLO 1차 분류
    - Gemini 앙상블 교정
    - 신규 꽃 탐지
    - name_ko 생성
    - name_en 생성
    - box2d 정리
    - 대표 색상 color_hex 추출

    추후 연결:
    - meaning은 개별 꽃말 매핑 함수가 완성되면 채울 예정입니다.
    """
    img_pil = load_image_from_bytes(image_bytes)
    image_width, image_height = img_pil.size

    ensemble_result = await analyze_flower_ensemble(image_bytes)

    detected_objects = []

    for item in ensemble_result:
        name_en = str(item.get("name_en", "")).strip()
        gemini_name_ko = str(item.get("name_ko", "")).strip()

        name_ko = resolve_name_ko(
            name_en=name_en,
            gemini_name_ko=gemini_name_ko,
        )

        box2d = clip_box2d(
            box2d=item.get("box2d", []),
            image_width=image_width,
            image_height=image_height,
        )

        if not name_en or box2d is None:
            continue

        color_hex = extract_dominant_flower_color_hex(
            image_pil=img_pil,
            box2d=box2d,
        )

        detected_objects.append({
            "name_ko": name_ko,
            "name_en": name_en,
            "meaning": "",
            "box2d": box2d,
            "color_hex": color_hex,
        })

    return {
        "detected_objects": detected_objects
    }

