import json
from functools import lru_cache
from pathlib import Path

FLOWER_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "floriography.json"


@lru_cache(maxsize=1)
def _load_db() -> dict[str, dict]:
    if not FLOWER_DB_PATH.exists():
        raise FileNotFoundError(
            f"꽃말 DB를 찾을 수 없습니다: {FLOWER_DB_PATH}\n"
            "data/floriography.json 파일이 data/ 폴더에 있는지 확인해주세요."
        )

    with open(FLOWER_DB_PATH, encoding="utf-8") as f:
        raw: dict[str, list[str]] = json.load(f)

    return {
        name_en.strip().lower(): {
            "name_ko": "",
            "name_en": name_en.strip().lower(),
            "meaning": ", ".join(meanings) if isinstance(meanings, list) else str(meanings or ""),
        }
        for name_en, meanings in raw.items()
        if str(name_en or "").strip()
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

def get_flower_info(name_en: str, name_ko: str = "") -> dict:
    db = _load_db()

    key = str(name_en or "").strip().lower()
    entry = db.get(key)

    if entry:
        return {
            "name_ko": entry.get("name_ko") or name_ko or name_en,
            "name_en": entry.get("name_en", name_en),
            "meaning": entry.get("meaning", ""),
        }

    return {
        "name_ko": name_ko or name_en,
        "name_en": name_en,
        "meaning": "",
    }