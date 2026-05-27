import asyncio
import json
import os
import sys
from pathlib import Path

from PIL import ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# ============================================================
# detector import 전에 YOLO 모델 경로 지정
# ============================================================

SINGLE_FLOWER_MODEL_PATH = PROJECT_ROOT / "models" / "single_baseline.pt"

if not SINGLE_FLOWER_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO 모델 파일을 찾을 수 없습니다:\n{SINGLE_FLOWER_MODEL_PATH}\n\n"
        f"Colab에서 학습한 best.pt를 아래 위치로 복사하세요:\n"
        f"{SINGLE_FLOWER_MODEL_PATH}"
    )

os.environ["YOLO_MODEL_PATH"] = str(SINGLE_FLOWER_MODEL_PATH)

import src.logic.detector_single_flower as detector
from src.logic.detector_single_flower import detect_flowers_for_api, load_image_from_bytes


TEST_IMAGE_PATH = Path(__file__).parent / "test-2.png"
ANNOTATED_IMAGE_PATH = Path(__file__).parent / "annotated_result.jpg"
RESULT_JSON_PATH = Path(__file__).parent / "detect_result.json"


def load_font(size=32):
    font_candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # mac
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # mac fallback
        "C:/Windows/Fonts/malgun.ttf",  # windows
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # linux fallback
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except IOError:
            continue

    return ImageFont.load_default()


def safe_color_hex(value):
    if isinstance(value, str) and value.startswith("#") and len(value) == 7:
        return value
    return "#FFFFFF"


def draw_label_with_background(draw, xy, text, font):
    x, y = xy

    try:
        text_bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(
            [
                text_bbox[0] - 4,
                text_bbox[1] - 4,
                text_bbox[2] + 4,
                text_bbox[3] + 4,
            ],
            fill="white",
        )
    except Exception:
        pass

    draw.text(
        [x, y],
        text,
        fill="black",
        font=font,
    )


async def main():
    if not TEST_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"테스트 이미지를 찾을 수 없습니다: {TEST_IMAGE_PATH}\n"
            f"test 폴더 안에 test_1.jpeg 파일을 넣어주세요."
        )

    with open(TEST_IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    print("꽃 탐지 테스트 시작")
    print("사용 detector 파일:", getattr(detector, "__file__", "unknown"))
    print("사용 YOLO 모델:", os.environ["YOLO_MODEL_PATH"])
    print("테스트 이미지:", TEST_IMAGE_PATH)
    print()

    result = await detect_flowers_for_api(image_bytes)

    print("탐지 결과:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # bbox 확인용 이미지 생성
    image = load_image_from_bytes(image_bytes)
    draw = ImageDraw.Draw(image)
    font = load_font(size=28)

    detected_objects = result.get("detected_objects", [])

    boxed_count = 0
    skipped_invalid_box_count = 0

    for idx, obj in enumerate(detected_objects, start=1):
        box2d = obj.get("box2d")
        name_en = obj.get("name_en", "")
        name_ko = obj.get("name_ko", "")
        color_hex = safe_color_hex(obj.get("color_hex"))

        if not isinstance(box2d, list) or len(box2d) != 4:
            skipped_invalid_box_count += 1
            print(f"[WARN] bbox 없는 객체 스킵: idx={idx}, name={name_ko}({name_en}), box2d={box2d}")
            continue

        boxed_count += 1

        x1, y1, x2, y2 = [int(v) for v in box2d]

        # bbox 그리기
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="black",
            width=5,
        )

        # 대표 색상 칩 그리기
        swatch_size = 36
        swatch_x1 = x1
        swatch_y1 = max(0, y1 - swatch_size - 12)
        swatch_x2 = x1 + swatch_size
        swatch_y2 = swatch_y1 + swatch_size

        try:
            draw.rectangle(
                [swatch_x1, swatch_y1, swatch_x2, swatch_y2],
                fill=color_hex,
                outline="black",
                width=2,
            )
        except ValueError:
            pass

        label = f"{idx}. {name_ko}({name_en}) {color_hex}"

        text_x = x1 + swatch_size + 8
        text_y = swatch_y1 + 3

        draw_label_with_background(
            draw=draw,
            xy=(text_x, text_y),
            text=label,
            font=font,
        )

    image.save(ANNOTATED_IMAGE_PATH)

    print()
    print("요약:")
    print("전체 detected_objects:", len(detected_objects))
    print("bbox 그린 객체:", boxed_count)
    print("bbox 없어서 스킵한 객체:", skipped_invalid_box_count)

    print()
    print("JSON 저장 완료:")
    print(RESULT_JSON_PATH)

    print()
    print("바운딩박스 확인용 이미지 저장 완료:")
    print(ANNOTATED_IMAGE_PATH)

    print()
    print("테스트 종료")


if __name__ == "__main__":
    asyncio.run(main())