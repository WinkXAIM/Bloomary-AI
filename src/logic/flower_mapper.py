import json
from pathlib import Path
from typing import List, Dict, Any


BASE_DIR = Path(__file__).resolve().parents[2]
FLOWER_DATA_PATH = BASE_DIR / "data" / "floriography.json"


def load_flower_data() -> Dict[str, Any]:
    with open(FLOWER_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


FLOWER_DATA = load_flower_data()


def map_flower_meanings(flower_names: List[str]) -> List[Dict[str, str]]:
    results = []

    for flower_name in flower_names:
        normalized_name = flower_name.strip().lower()
        flower_info = FLOWER_DATA.get(normalized_name)

        if not flower_info:
            results.append({
                "name_ko": flower_name,
                "name_en": flower_name,
                "meaning": ""
            })
            continue

        results.append({
            "name_ko": flower_info.get("name_ko", flower_name),
            "name_en": flower_info.get("name_en", flower_name),
            "meaning": flower_info.get("meaning", "")
        })

    return results