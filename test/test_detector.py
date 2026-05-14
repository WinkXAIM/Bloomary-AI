import asyncio
import json
import sys
from pathlib import Path

from PIL import ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.logic.detector import detect_flowers_for_api, load_image_from_bytes


TEST_IMAGE_PATH = Path(__file__).parent / "test_bouquet.jpeg"
ANNOTATED_IMAGE_PATH = Path(__file__).parent / "annotated_result.jpg"


async def main():
    if not TEST_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"테스트 이미지를 찾을 수 없습니다: {TEST_IMAGE_PATH}\n"
            f"test 폴더 안에 test_bouquet.jpeg 파일을 넣어주세요."
        )

    with open(TEST_IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    print("꽃 탐지 테스트 시작\n")

    result = await detect_flowers_for_api(image_bytes)

    print("탐지 결과:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # bbox 확인용 이미지 생성
    image = load_image_from_bytes(image_bytes)
    draw = ImageDraw.Draw(image)

    detected_objects = result.get("detected_objects", [])

    for idx, obj in enumerate(detected_objects, start=1):
        box2d = obj.get("box2d")
        name_en = obj.get("name_en", "")
        name_ko = obj.get("name_ko", "")
        color_hex = obj.get("color_hex") or "#FFFFFF"

        if not box2d or len(box2d) != 4:
            continue

        x1, y1, x2, y2 = box2d

        # bbox 그리기
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="black",
            width=5,
        )

        # 대표 색상 칩 그리기
        swatch_size = 40
        swatch_x1 = x1
        swatch_y1 = max(0, y1 - swatch_size - 10)
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

        # 라벨 그리기
        # 기본 폰트는 한글이 깨질 수 있어서 name_en 중심으로 표시
        label = f"{idx}. {name_en} {color_hex}"

        text_x = x1 + swatch_size + 8
        text_y = swatch_y1 + 8

        draw.text(
            [text_x, text_y],
            label,
            fill="red",
        )

    image.save(ANNOTATED_IMAGE_PATH)

    print(f"\n바운딩박스 확인용 이미지 저장 완료:")
    print(ANNOTATED_IMAGE_PATH)

    print("\n테스트 종료")


if __name__ == "__main__":
    asyncio.run(main())