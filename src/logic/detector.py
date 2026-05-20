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

def normalize_name_en(name_en: str) -> str:
    name_en = str(name_en or "").strip().lower()
    name_en = name_en.replace("-", "_").replace(" ", "_")
    name_en = re.sub(r"[^a-z_]", "", name_en)
    name_en = re.sub(r"_+", "_", name_en).strip("_")
    return name_en

def resolve_name_ko(name_en: str, gemini_name_ko: str = "") -> str:
    """
    국문명 결정
    1순위: YOLO 88개 클래스 매핑
    2순위: Gemini가 준 name_ko
    3순위: 빈 문자열
    """
    name_en = normalize_name_en(name_en)
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

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def box_intersection(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return max(0, x2 - x1) * max(0, y2 - y1)

def box_iou(a, b):
    inter = box_intersection(a, b)
    union = box_area(a) + box_area(b) - inter

    if union <= 0:
        return 0

    return inter / union

def min_overlap_ratio(a, b):
    """
    작은 박스가 큰 박스 안에 거의 들어간 경우도 중복으로 보기 위한 값
    """
    inter = box_intersection(a, b)
    min_area = min(box_area(a), box_area(b))

    if min_area <= 0:
        return 0

    return inter / min_area

def dedupe_same_object(detected_objects, iou_thresh=0.82, contain_thresh=0.90):
    """
    같은 꽃을 여러 클래스가 잡은 경우 하나만 남김.
    거의 같은 bbox가 겹치는 케이스 제거용.
    """
    result = []

    for obj in detected_objects:
        box = obj.get("box2d")

        if not box or len(box) != 4:
            continue

        is_duplicate = False

        for saved in result:
            saved_box = saved.get("box2d")

            if not saved_box or len(saved_box) != 4:
                continue

            iou = box_iou(box, saved_box)
            contain = min_overlap_ratio(box, saved_box)

            if iou >= iou_thresh or contain >= contain_thresh:
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(obj)

    return result

def dedupe_by_flower_name(detected_objects):
    """
    같은 꽃 이름(name_en)이 여러 번 나온 경우 하나만 남김.
    대표 박스는 bbox 면적이 가장 큰 객체로 선택.
    """
    best_by_name = {}

    for obj in detected_objects:
        name_en = normalize_name_en(obj.get("name_en", ""))
        box = obj.get("box2d")

        if not name_en or not box or len(box) != 4:
            continue

        if name_en not in best_by_name:
            best_by_name[name_en] = obj
            continue

        current_area = box_area(box)
        saved_area = box_area(best_by_name[name_en].get("box2d", [0, 0, 0, 0]))

        if current_area > saved_area:
            best_by_name[name_en] = obj

    return list(best_by_name.values())

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
    1. bbox crop 후 가장자리 영역을 일부 제거해 배경/포장지 영향을 줄인다.
    2. 중앙부 타원 마스크를 적용해 꽃 영역 후보를 우선적으로 사용한다.
    3. HSV 기준으로 초록색 계열과 너무 어두운 픽셀을 제거한다.
    4. 흰색/크림색 꽃은 채도가 낮고 밝기가 높은 픽셀을 우선 사용해 median 색상을 반환한다.
    5. 일반 색상 꽃은 유효 픽셀에 k-means를 적용한다.
    6. 클러스터 크기, 채도, 밝기를 함께 고려해 대표색을 선택하고 HEX로 반환한다.
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

    # bbox 가장자리 포장지/배경 영향 줄이기
    margin_x = int(crop_w * 0.08)
    margin_y = int(crop_h * 0.08)

    if crop_w - 2 * margin_x > 10 and crop_h - 2 * margin_y > 10:
        crop = crop[margin_y:crop_h - margin_y, margin_x:crop_w - margin_x]

    crop_h, crop_w = crop.shape[:2]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # 중앙부 위주로 보기 위한 타원 마스크
    yy, xx = np.ogrid[:crop_h, :crop_w]
    cx = (crop_w - 1) / 2
    cy = (crop_h - 1) / 2
    rx = max(crop_w * 0.48, 1)
    ry = max(crop_h * 0.48, 1)

    center_mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1

    # 줄기/잎 제거
    green_mask = (
        (h >= 35) & (h <= 90) &
        (s >= 35) &
        (v >= 35)
    )

    # 너무 어두운 부분 제거
    dark_mask = v < 45

    base_mask = center_mask & (~green_mask) & (~dark_mask)

    if np.count_nonzero(base_mask) < 50:
        base_mask = (~green_mask) & (~dark_mask)

    if np.count_nonzero(base_mask) < 50:
        base_mask = v >= 45

    # 흰색 / 크림색 꽃 예외 처리
    # 채도 낮고 밝은 픽셀을 꽃잎 후보로 봄
    white_mask = base_mask & (s <= 65) & (v >= 145)

    if np.count_nonzero(white_mask) >= max(80, int(np.count_nonzero(base_mask) * 0.18)):
        white_pixels = crop[white_mask]

        # 너무 어두운 흰색 그림자는 제외하고 밝은 쪽 픽셀만 사용
        white_hsv = cv2.cvtColor(
            white_pixels.reshape(-1, 1, 3),
            cv2.COLOR_BGR2HSV
        ).reshape(-1, 3)

        brightness_cut = np.percentile(white_hsv[:, 2], 60)
        white_pixels = white_pixels[white_hsv[:, 2] >= brightness_cut]

        if len(white_pixels) > 0:
            median_bgr = np.median(white_pixels, axis=0).astype(np.uint8)
            return bgr_to_hex(median_bgr)

    valid_pixels = crop[base_mask]

    if len(valid_pixels) < 30:
        return None

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
    counts = np.bincount(labels, minlength=actual_k).astype(float)

    centers_u8 = np.clip(centers, 0, 255).astype(np.uint8)
    centers_hsv = cv2.cvtColor(
        centers_u8.reshape(1, -1, 3),
        cv2.COLOR_BGR2HSV
    )[0]

    center_h = centers_hsv[:, 0].astype(float)
    center_s = centers_hsv[:, 1].astype(float)
    center_v = centers_hsv[:, 2].astype(float)

    # 단순히 가장 큰 클러스터가 아니라
    # 밝기와 채도까지 고려해서 꽃잎 색 후보를 고름
    scores = counts * (0.6 + center_s / 255.0) * (0.5 + center_v / 255.0)

    # 초록 계열은 강하게 패널티
    green_center = (
        (center_h >= 35) & (center_h <= 90) &
        (center_s >= 35)
    )
    scores[green_center] *= 0.05

    # 너무 어두운 중심부도 패널티
    scores[center_v < 70] *= 0.2

    dominant_idx = int(np.argmax(scores))
    dominant_bgr = centers_u8[dominant_idx]

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
    """
    img_pil = load_image_from_bytes(image_bytes)
    image_width, image_height = img_pil.size

    results = yolo_model.predict(
        source=img_pil,
        conf=0.2,
        iou=0.45,
        agnostic_nms=True,
        max_det=30,
        verbose=False,
    )
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

참고용 YOLO 클래스 목록:
{json.dumps(class_names, ensure_ascii=False)}

임무:
1. 제공된 YOLO JSON의 "box_2d" 좌표는 1픽셀도 수정하지 마라.
2. 각 박스의 꽃 이름이 정확한지 전문가의 눈으로 검수하고 교정하라.
3. name_en은 기본적으로 YOLO JSON의 label을 유지하라.
4. 단, 사진상 명백히 잘못된 경우에만 꽃 이름을 교정하라.
5. 교정할 때도 가능하면 참고용 YOLO 클래스 목록 중 하나를 사용하라.
6. 클래스 목록에 없는 꽃 이름은 정말 확실할 때만 사용하라.
7. name_en은 반드시 소문자 snake_case로 작성하라.
   예: gerbera, spray_rose, white_stock, sweet_pea
8. 색상명, 품질 표현, 장식 표현은 name_en에 넣지 마라.
   나쁜 예: pink_gerbera, beautiful_rose, large_white_daisy
   좋은 예: gerbera, rose, daisy

출력 규칙:
- 반드시 JSON 리스트 형식으로만 응답할 것.
- 텍스트 설명, 인사, 마크다운 코드블록은 절대로 포함하지 말 것.
- 각 객체는 반드시 아래 필드만 포함할 것:
    - "name_ko": 최종 꽃 이름, 국문
    - "name_en": 최종 꽃 이름, 영문 snake_case
    - "box2d": YOLO JSON의 "box_2d" 좌표를 그대로 사용한 [x1, y1, x2, y2]

응답 예시:
[
    {{
        "name_ko": "거베라",
        "name_en": "gerbera",
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
            name_en = normalize_name_en(item.get("label", ""))
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
        name_en = normalize_name_en(item.get("name_en", ""))
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

    detected_objects = dedupe_same_object(detected_objects)
    detected_objects = dedupe_by_flower_name(detected_objects)

    return {
        "detected_objects": detected_objects
    }