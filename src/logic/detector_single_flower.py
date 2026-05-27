import json
import io
import os
import math
from PIL import Image, ImageOps, ImageDraw, ImageFont
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv
import re
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from src.logic.flower_mapper import get_flower_info

load_dotenv()

# 제미나이 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('models/gemini-3-flash-preview')

# YOLO 모델 로드
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    os.path.join("models", "single_baseline.pt")
)

yolo_model = YOLO(YOLO_MODEL_PATH)

YOLO_IMG_SIZE = 704
# single-class flower 후보 생성 모델 기준 기본값
# UI에 직접 그리는 값이 아니라 Gemini에 넘길 후보를 만드는 값
YOLO_CONF = 0.30
YOLO_IOU = 0.40
YOLO_MAX_DET = 100
GEMINI_MAX_CANDIDATES = 24


def get_yolo_class_names():
    names = yolo_model.names

    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names.keys())]

    return [str(name) for name in names]

def is_single_flower_yolo() -> bool:
    """현재 로드된 YOLO가 flower 단일 클래스 모델인지 확인."""
    class_names = [normalize_name_en(name) for name in get_yolo_class_names()]
    return len(class_names) == 1 and class_names[0] == "flower"

def get_reference_flower_names() -> List[str]:
    """Gemini가 참고할 꽃 이름 후보 목록."""
    if is_single_flower_yolo():
        return sorted(FLOWER_NAME_KO_MAP.keys())

    return get_yolo_class_names()

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
    "gerbera": "거베라",
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

# Gemini/모델이 자주 쓰는 별칭을 내부 표준명으로 보정
FLOWER_ALIAS_EN_MAP = {
    "baby_breath": "gypsophila",
    "babys_breath": "gypsophila",
    "babybreath": "gypsophila",
    "baby_s_breath": "gypsophila",
    "matricaria": "chamomile",
    "matricaria_daisy": "chamomile",
    "gerbera_daisy": "gerbera",
    "transvaal_daisy": "gerbera",
}

# 꽃이 아닌 소재/배경/장식은 최종 결과에서 제외
NON_FLOWER_NAMES = {
    "eucalyptus",
    "silver_dollar_eucalyptus",
    "baby_blue_eucalyptus",
    "greenery",
    "foliage",
    "leaf",
    "leaves",
    "stem",
    "stems",
    "fern",
    "ribbon",
    "wrapping_paper",
    "paper",
    "card",
    "background",
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
    return FLOWER_ALIAS_EN_MAP.get(name_en, name_en)

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

def filter_yolo_json_for_llm(
    yolo_json: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    max_candidates: int = GEMINI_MAX_CANDIDATES,
) -> List[Dict[str, Any]]:
    """
    Gemini에 넘기기 전 YOLO 후보를 정리한다.
    최종 API 응답 형식은 바꾸지 않고 내부 후보 개수만 줄인다.
    """
    filtered = []
    image_area = max(1, image_width * image_height)

    for item in yolo_json:
        box = item.get("box_2d")
        if not box or len(box) != 4:
            continue

        clipped = clip_box2d(
            box2d=box,
            image_width=image_width,
            image_height=image_height,
        )

        if clipped is None:
            continue

        x1, y1, x2, y2 = clipped
        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            continue

        area_ratio = (bw * bh) / image_area
        aspect_ratio = bw / max(bh, 1)

        try:
            conf = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0

        # 너무 작은 박스 제거
        if area_ratio < 0.0006:
            continue

        # 너무 큰 박스 제거: 포장지/배경/군집 전체 오탐 방지
        if area_ratio > 0.20:
            continue

        # 너무 길쭉한 박스 제거
        if aspect_ratio > 3.2 or aspect_ratio < 0.30:
            continue

        filtered.append({
            "label": item.get("label", "flower"),
            "box_2d": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": conf,
        })

    filtered.sort(
        key=lambda x: float(x.get("confidence") or 0.0),
        reverse=True,
    )

    return filtered[:max_candidates]


def make_crop_grid_for_gemini(
    image_pil: Image.Image,
    yolo_json: List[Dict[str, Any]],
    cell_size: int = 180,
    cols: int = 4,
    crop_padding_ratio: float = 0.18,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    YOLO bbox 후보들을 번호가 붙은 crop grid 이미지 1장으로 만든다.

    Gemini 호출은 1회만 유지하고, Gemini가 전체 이미지 좌표를 헷갈리지 않도록
    crop grid의 candidate_id를 기준으로 꽃 이름을 반환하게 한다.

    반환값:
    - crop_grid_img: Gemini에 같이 넘길 grid 이미지
    - candidates_json: 프롬프트에 넣을 candidate 목록. _crop은 제거되어 있음.
    """
    image_width, image_height = image_pil.size
    candidates = []

    for item in yolo_json:
        box = clip_box2d(
            box2d=item.get("box_2d", []),
            image_width=image_width,
            image_height=image_height,
        )

        if box is None:
            continue

        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            continue

        pad = int(max(bw, bh) * crop_padding_ratio)

        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(image_width, x2 + pad)
        cy2 = min(image_height, y2 + pad)

        crop = image_pil.crop((cx1, cy1, cx2, cy2)).convert("RGB")

        candidates.append({
            "candidate_id": len(candidates) + 1,
            "label": item.get("label", "flower"),
            "box_2d": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": item.get("confidence"),
            "_crop": crop,
        })

    if not candidates:
        empty_grid = Image.new("RGB", (cell_size, cell_size), "white")
        return empty_grid, []

    cols = max(1, cols)
    rows = math.ceil(len(candidates) / cols)

    grid_width = cols * cell_size
    grid_height = rows * cell_size
    grid = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, candidate in enumerate(candidates):
        crop = candidate["_crop"]

        row = i // cols
        col = i % cols
        cell_x = col * cell_size
        cell_y = row * cell_size

        label_h = 24
        image_area_size = cell_size - label_h

        # 원본 crop은 candidate에 남겨두지 않을 거라 여기서 resize해도 괜찮다.
        crop.thumbnail((image_area_size, image_area_size))

        paste_x = cell_x + (cell_size - crop.width) // 2
        paste_y = cell_y + label_h + (image_area_size - crop.height) // 2
        grid.paste(crop, (paste_x, paste_y))

        label = f"#{candidate['candidate_id']}"
        draw.rectangle(
            [cell_x, cell_y, cell_x + 48, cell_y + label_h],
            fill="white",
            outline="black",
        )
        draw.text((cell_x + 5, cell_y + 5), label, fill="black", font=font)
        draw.rectangle(
            [cell_x, cell_y, cell_x + cell_size - 1, cell_y + cell_size - 1],
            outline="black",
        )

    clean_candidates = []

    for candidate in candidates:
        clean_candidate = dict(candidate)
        clean_candidate.pop("_crop", None)
        clean_candidates.append(clean_candidate)

    return grid, clean_candidates

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

def get_internal_confidence(obj: Dict[str, Any]) -> float:
    try:
        return float(obj.get("_confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0

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
    같은 꽃을 여러 박스가 잡은 경우 하나만 남김.
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
    """
    best_by_name = {}

    for obj in detected_objects:
        name_en = normalize_name_en(obj.get("name_en", ""))

        if not name_en:
            continue

        if name_en not in best_by_name:
            best_by_name[name_en] = obj
            continue

        current_conf = get_internal_confidence(obj)
        saved_conf = get_internal_confidence(best_by_name[name_en])

        # 같은 이름이면 bbox가 있는 객체를 우선하고, 둘 다 bbox 상태가 같으면 confidence 우선
        current_has_box = isinstance(obj.get("box2d"), list) and len(obj.get("box2d")) == 4
        saved_has_box = isinstance(best_by_name[name_en].get("box2d"), list) and len(best_by_name[name_en].get("box2d")) == 4

        if current_has_box and not saved_has_box:
            best_by_name[name_en] = obj
        elif current_has_box == saved_has_box and current_conf > saved_conf:
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
    4. 흰색/크림색 예외 처리 전에 채도 높은 컬러 픽셀을 먼저 검사한다.
    5. 파란 안개꽃처럼 bbox 안에 흰 꽃이 같이 들어간 경우에도
    채도 높은 컬러 픽셀이 충분하면 해당 컬러를 우선 반환한다.
    6. 채도 높은 컬러 픽셀이 부족한 경우에만 흰색/크림색 꽃 예외 처리를 수행한다.
    7. 일반 색상 꽃은 유효 픽셀에 k-means를 적용한다.
    8. 클러스터 크기, 채도, 밝기를 함께 고려해 대표색을 선택하고 HEX로 반환한다.
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

    base_count = np.count_nonzero(base_mask)

    if base_count < 30:
        return None

    # ============================================================
    # 1순위: 채도 높은 컬러 픽셀 우선 처리
    # ============================================================
    # 파란 안개꽃처럼 작은 컬러 꽃이 bbox 안에 있고,
    # 흰 꽃/밝은 배경이 같이 들어간 경우를 보정한다.
    colored_mask = (
        base_mask &
        (~green_mask) &
        (s >= 70) &
        (v >= 70)
    )

    colored_count = np.count_nonzero(colored_mask)

    # 전체 base 픽셀 중 컬러 픽셀이 조금이라도 의미 있게 있으면
    # 흰색 예외 처리보다 컬러 클러스터를 우선한다.
    if colored_count >= max(60, int(base_count * 0.035)):
        valid_pixels = crop[colored_mask]

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

        scores = counts * (0.8 + center_s / 255.0) * (0.6 + center_v / 255.0)

        # 혹시 남아 있는 초록 계열은 패널티
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
    # 2순위: 흰색 / 크림색 꽃 예외 처리
    # ============================================================
    white_mask = base_mask & (s <= 65) & (v >= 145)

    if np.count_nonzero(white_mask) >= max(80, int(base_count * 0.18)):
        white_pixels = crop[white_mask]

        white_hsv = cv2.cvtColor(
            white_pixels.reshape(-1, 1, 3),
            cv2.COLOR_BGR2HSV
        ).reshape(-1, 3)

        brightness_cut = np.percentile(white_hsv[:, 2], 60)
        white_pixels = white_pixels[white_hsv[:, 2] >= brightness_cut]

        if len(white_pixels) > 0:
            median_bgr = np.median(white_pixels, axis=0).astype(np.uint8)
            return bgr_to_hex(median_bgr)

    # ============================================================
    # 3순위: 일반 대표색 k-means 처리
    # ============================================================
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
    YOLO는 꽃송이 위치 후보만 만들고, Gemini는 crop grid를 보고 꽃 이름을 분류한다.

    핵심:
    - Gemini API 호출은 1회만 한다.
    - 원본 이미지 + 번호 붙은 crop grid 이미지 + 후보 JSON을 한 번에 전달한다.
    - Gemini는 candidate_id만 반환한다.
    - 서버에서 candidate_id를 원래 bbox 좌표로 복원한다.

    최종 반환 형태:
    [
        {
            "name_ko": "장미",
            "name_en": "rose",
            "box2d": [100, 200, 300, 400],
            "confidence": 0.87
        }
    ]
    """
    img_pil = load_image_from_bytes(image_bytes)
    image_width, image_height = img_pil.size

    single_flower_mode = is_single_flower_yolo()

    results = yolo_model.predict(
        source=img_pil,
        imgsz=YOLO_IMG_SIZE,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        agnostic_nms=False,
        max_det=YOLO_MAX_DET,
        verbose=False,
    )

    yolo_json = get_json_for_llm(results)

    # crop grid가 너무 복잡해지면 Gemini가 더 헷갈린다.
    # single-class / multi-class 상관없이 최종 후보 수를 줄인다.
    yolo_json = filter_yolo_json_for_llm(
        yolo_json=yolo_json,
        image_width=image_width,
        image_height=image_height,
        max_candidates=GEMINI_MAX_CANDIDATES,
    )

    if not yolo_json:
        return []

    crop_grid_img, gemini_candidates = make_crop_grid_for_gemini(
        image_pil=img_pil,
        yolo_json=yolo_json,
        cell_size=180,
        cols=4,
        crop_padding_ratio=0.18,
    )

    if not gemini_candidates:
        return []

    class_names = get_reference_flower_names()

    prompt = f"""
역할: 너는 20년 경력의 수석 플로리스트이자 시각 지능 전문가야.

입력 데이터:
1. 첫 번째 이미지: 전체 부케 원본 이미지
2. 두 번째 이미지: YOLO 후보 bbox를 crop해서 번호를 붙인 crop grid 이미지
3. 이미지 크기:
    - width: {image_width}
    - height: {image_height}

후보 JSON:
{json.dumps(gemini_candidates, indent=2, ensure_ascii=False)}

참고용 꽃 이름 후보 목록:
{json.dumps(class_names, ensure_ascii=False)}

중요 전제:
- crop grid 이미지에서 각 칸 왼쪽 위의 #번호가 candidate_id다.
- 후보 JSON의 candidate_id와 crop grid의 #번호는 같은 후보를 의미한다.
- YOLO JSON의 label이 "flower"라면 이것은 꽃 이름이 아니라 개별 꽃송이 위치 후보라는 뜻이다.
- YOLO는 꽃 이름을 확정하지 않는다.
- 전체 원본 이미지는 맥락 확인용이고, 꽃 이름 판단은 crop grid를 우선 사용한다.
- 후보 JSON의 box_2d 좌표는 직접 수정하지 않는다. 너는 candidate_id만 고른다.

임무:
1. crop grid의 각 candidate가 실제 꽃송이인지 판단하라.
2. 실제 꽃송이로 보이는 candidate만 결과에 포함하라.
3. 포장지, 리본, 잎, 줄기, 카드, 배경, 손, 옷, 조명, 바닥, 유칼립투스, 그린 소재로 보이는 candidate는 결과에서 제외하라.
4. 각 candidate의 꽃 이름을 name_ko와 name_en으로 분류하라.
5. name_en은 가능하면 참고용 꽃 이름 후보 목록 중 하나를 사용하라.
6. 참고용 목록에 없지만 확실한 꽃이면 일반적인 영어 꽃 이름을 소문자 snake_case로 작성하라.
7. 색상명, 품질 표현, 장식 표현은 name_en에 넣지 마라.
   나쁜 예: pink_rose, white_flower, beautiful_gerbera
   좋은 예: rose, gerbera, carnation
8. 같은 실제 꽃송이를 중복 candidate가 잡은 경우 가장 적절한 candidate_id 하나만 남겨라.
9. 최종 결과는 꽃 종류별 대표 candidate_id 1개만 남겨라. 같은 name_en은 한 번만 출력하라.
10. 유칼립투스, 잎, 줄기, 그린 소재, 필러 잎은 꽃 종류가 아니므로 절대 결과에 포함하지 마라.
11. candidate_id가 없는 꽃은 출력하지 마라. 반드시 후보 JSON에 있는 candidate_id 중 하나만 사용하라.
12. confidence는 후보 JSON의 confidence 값을 그대로 사용하라.

출력 규칙:
- 반드시 JSON 리스트 형식으로만 응답할 것.
- 텍스트 설명, 인사, 마크다운 코드블록은 절대로 포함하지 말 것.
- 각 객체는 반드시 아래 필드만 포함할 것:
    - "candidate_id": crop grid의 번호
    - "name_ko": 최종 꽃 이름, 국문
    - "name_en": 최종 꽃 이름, 영문 snake_case
    - "confidence": 후보 JSON의 confidence 값을 그대로 사용

응답 예시:
[
    {{
        "candidate_id": 3,
        "name_ko": "거베라",
        "name_en": "gerbera",
        "confidence": 0.87
    }}
]
"""

    try:
        # Gemini 호출은 여기 딱 1번만 한다.
        response = await asyncio.to_thread(
            gemini_model.generate_content,
            [prompt, img_pil, crop_grid_img],
        )

        corrected_data = parse_gemini_json_list(response.text)

        if not corrected_data:
            raise ValueError("Gemini 응답에서 유효한 JSON 리스트를 찾지 못했습니다.")

        candidate_by_id = {
            int(candidate["candidate_id"]): candidate
            for candidate in gemini_candidates
        }

        final_data = []

        for item in corrected_data:
            try:
                candidate_id = int(item.get("candidate_id"))
            except (TypeError, ValueError):
                continue

            candidate = candidate_by_id.get(candidate_id)

            if not candidate:
                continue

            final_data.append({
                "name_ko": item.get("name_ko", ""),
                "name_en": item.get("name_en", ""),
                "box2d": candidate.get("box_2d", []),
                "confidence": candidate.get("confidence"),
            })

        if not final_data:
            raise ValueError("Gemini 응답을 candidate_id 기준으로 복원하지 못했습니다.")

        return final_data

    except Exception as e:
        """
        Fallback:
        Gemini API 실패 시 서버 중단 방지.
        multi-class YOLO는 기본 결과라도 반환하고, single-class YOLO는 flower라는 가짜 이름을 반환하지 않는다.
        """
        print(f"앙상블 분석 중 에러 발생. YOLO 결과로 대체: {e}")

        if single_flower_mode:
            return []

        fallback_result = []

        for item in yolo_json:
            name_en = normalize_name_en(item.get("label", ""))
            name_ko = resolve_name_ko(name_en)

            if not name_en or name_en in NON_FLOWER_NAMES:
                continue

            fallback_result.append({
                "name_ko": name_ko,
                "name_en": name_en,
                "box2d": item.get("box_2d", []),
                "confidence": item.get("confidence"),
            })

        return fallback_result

# ============================================================
# /ai/detect-flowers API에서 호출할 최종 함수
# ============================================================

async def detect_flowers_for_api(image_bytes: bytes) -> Dict[str, List[Dict[str, Any]]]:
    """
    /ai/detect-flowers API 응답 형태로 반환하는 최종 로직.
    최종 객체의 필드명은 기존과 동일하게 유지한다.
    """
    img_pil = load_image_from_bytes(image_bytes)
    image_width, image_height = img_pil.size

    ensemble_result = await analyze_flower_ensemble(image_bytes)

    detected_objects = []

    for item in ensemble_result:
        name_en = normalize_name_en(item.get("name_en", ""))
        gemini_name_ko = str(item.get("name_ko", "")).strip()

        if not name_en:
            continue

        name_ko = resolve_name_ko(
            name_en=name_en,
            gemini_name_ko=gemini_name_ko,
        )

        if name_en in NON_FLOWER_NAMES:
            continue

        box2d = clip_box2d(
            box2d=item.get("box2d", []),
            image_width=image_width,
            image_height=image_height,
        )

        if box2d is None:
            continue

        color_hex = extract_dominant_flower_color_hex(
            image_pil=img_pil,
            box2d=box2d,
        )

        flower_info = get_flower_info(
            name_en=name_en,
            name_ko=name_ko,
        )

        detected_objects.append({
            "name_ko": flower_info.get("name_ko") or name_ko,
            "name_en": flower_info.get("name_en") or name_en,
            "meaning": flower_info.get("meaning") or "",
            "box2d": box2d,
            "color_hex": color_hex,
            "_confidence": item.get("confidence"),
        })

    detected_objects.sort(
        key=lambda obj: get_internal_confidence(obj),
        reverse=True,
    )

    detected_objects = dedupe_same_object(detected_objects)
    detected_objects = dedupe_by_flower_name(detected_objects)

    for obj in detected_objects:
        obj.pop("_confidence", None)

    return {
        "detected_objects": detected_objects
    }
