import os
import json
import re
from typing import List, Dict, Any

import google.generativeai as genai


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("LLM 응답에서 JSON을 찾을 수 없습니다.")

    return json.loads(match.group())


async def generate_combined_message(
    flowers: List[Dict[str, str]]
) -> Dict[str, str]:
    """
    /ai/combine-floriography
    확정된 꽃 리스트를 바탕으로 꽃말 조합 문구 생성
    """

    prompt = f"""
너는 꽃말을 자연스럽게 엮어서 감성적인 꽃다발 메시지를 만들어주는 AI야.

사용자가 꽃다발에 들어간 꽃 리스트를 보내주면,
각 꽃의 꽃말을 바탕으로 하나의 자연스러운 메시지처럼 이어줘.

[꽃 정보]
{json.dumps(flowers, ensure_ascii=False, indent=2)}

[입력 데이터 설명]
- nameKo 또는 name_ko: 꽃 이름 국문명
- nameEn 또는 name_en: 꽃 이름 영문명
- meaning: 개별 꽃말

[메시지 만들 때 중요한 점]
- 꽃말을 그냥 나열하지 마.
- 여러 꽃 의미가 하나의 감정처럼 이어져야 해.
- 너무 오글거리거나 시처럼 쓰지 마.
- 실제 꽃 선물 카드에 적혀 있어도 자연스러운 느낌으로 써줘.
- 따뜻하고 부드러운 분위기로 작성해줘.
- 20대 사용자가 읽어도 부담스럽지 않은 말투로 해줘.
- 꽃 이름을 모두 억지로 넣지 않아도 돼.
- 대신 주요 꽃말은 메시지 안에 자연스럽게 반영해줘.
- 꽃말이 비어 있는 꽃은 분위기에 맞게 자연스럽게 녹여줘.
- 부정적이거나 무거운 의미는 긍정적이고 부드럽게 완화해줘.
- 한국어 문장은 1~2문장 정도로 짧고 예쁘게 써줘.
- 영어 문장은 짧은 카드 문구 느낌으로 써줘.
- summary_en은 히스토리에서 보일 짧은 영어 요약으로 써줘.

[출력 조건]
아래 JSON 형식만 출력해.
설명, 마크다운, 코드블록은 절대 넣지 마.

{{
  "combined_message_ko": "꽃말 조합 감성 문구",
  "story_preview_en": "A short emotional English sentence.",
  "summary_en": "Short English summary"
}}
"""

    response = model.generate_content(prompt)
    result = parse_json_response(response.text)

    return {
        "combined_message_ko": result.get("combined_message_ko", ""),
        "story_preview_en": result.get("story_preview_en", ""),
        "summary_en": result.get("summary_en", "")
    }


async def generate_bouquet_recommendation(
    user_situation: str
) -> Dict[str, Any]:
    """
    /ai/recommend-bouquet
    사용자 상황 기반 꽃다발 추천
    """

    prompt = f"""
너는 상황에 어울리는 꽃다발을 추천해주는 AI야.

사용자가 어떤 상황인지 설명하면,
그 분위기에 잘 어울리는 꽃들이랑 꽃말 기반 메시지를 추천해줘.

[사용자 상황]
{user_situation}

[상황을 볼 때 생각할 것]
- 사용자가 누구에게 꽃을 주려는지 파악해.
  예: 연인, 친구, 부모님, 선생님, 동료, 가족, 본인
- 어떤 상황인지 파악해.
  예: 생일, 졸업, 입학, 취업, 위로, 사과, 고백, 감사, 응원
- 사용자가 어떤 감정을 전하고 싶은지 파악해.
  예: 사랑, 감사, 응원, 위로, 존경, 축하, 그리움, 사과
- 어떤 톤이 어울리는지도 생각해.
  예: 다정한, 담백한, 진심 어린, 밝은, 차분한, 위로하는

[추천할 때 중요한 점]
- 상황에 맞는 꽃 3개를 추천해줘.
- 각 꽃은 꽃말과 상황이 자연스럽게 연결되어야 해.
- 너무 흔하거나 의미 없는 조합은 피해줘.
- 부정적이거나 너무 무거운 꽃말은 피하거나 부드럽게 바꿔줘.
- 실제 꽃다발 추천 서비스처럼 자연스럽고 감성적으로 작성해줘.
- 너무 광고 같거나 과한 표현은 쓰지 마.
- title은 꽃다발 컨셉 느낌으로 짧게 만들어줘.
- combined_message는 따뜻한 느낌으로 1~2문장 정도 작성해줘.
- flowers 배열에는 name_ko, name_en, meaning만 넣어줘.

[출력 조건]
아래 JSON 형식만 출력해줘.
설명, 마크다운, 코드블록은 절대 넣지 마.

{{
  "title": "꽃다발 제목",
  "combined_message": "꽃다발 꽃말 조합 문구",
  "flowers": [
    {{
      "name_ko": "꽃 이름 국문",
      "name_en": "Flower English Name",
      "meaning": "꽃말"
    }}
  ]
}}
"""

    response = model.generate_content(prompt)
    result = parse_json_response(response.text)

    return {
        "title": result.get("title", ""),
        "combined_message": result.get("combined_message", ""),
        "flowers": result.get("flowers", [])
    }