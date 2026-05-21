import json
from functools import lru_cache
from pathlib import Path

FLOWER_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "flower_db.json"


@lru_cache(maxsize=1)
def _load_db() -> dict[str, dict]:
    if not FLOWER_DB_PATH.exists():
        raise FileNotFoundError(
            f"꽃말 DB를 찾을 수 없습니다: {FLOWER_DB_PATH}\n"
            "백엔드팀 flower_db.json이 data/ 폴더에 있는지 확인해주세요."
        )

    with open(FLOWER_DB_PATH, encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    return {
        entry["name_en"].strip().lower(): {
            "name_ko": entry.get("name_ko", ""),
            "name_en": entry.get("name_en", ""),
            "meaning": entry.get("meaning") or "",
        }
        for entry in raw
        if entry.get("name_en")
    }


def map_flowers_from_yolo(yolo_names: list[str]) -> list[dict]:
    db = _load_db()
    result = []

    for raw_name in yolo_names:
        key = raw_name.strip().lower()
        entry = db.get(key)

        if entry:
            result.append({
                "name_ko": entry["name_ko"],
                "name_en": entry["name_en"],
                "meaning": entry["meaning"],
            })
        else:
            result.append({
                "name_ko": raw_name,
                "name_en": raw_name,
                "meaning": "",
            })

    return result