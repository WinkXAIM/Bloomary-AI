import json
import os
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
model = genai.GenerativeModel("models/gemini-3.1-flash-lite")


def combine_floriography(flowers: list[dict]) -> dict:
    flower_list = "\n".join(
        f"- {f['name_ko']}({f['name_en']}): {f['meaning']}" for f in flowers
    )

    prompt = f"""
아래 꽃들의 꽃말을 조합해서 자연스럽고 감성적인 문구를 만들어주세요.
반드시 아래 JSON 형식으로만 응답하세요. 다른 설명은 쓰지 마세요.

꽃 목록:
{flower_list}

응답 형식:
{{
  "combined_message_ko": "꽃말을 조합한 한국어 감성 문구 (2~3문장)",
  "story_preview_en": "스토리 미리보기용 영문 문구 (1문장)",
  "summary_en": "히스토리 페이지용 짧은 요약 (3~5단어)"
}}
"""

    response = model.generate_content(prompt)
    text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def recommend_bouquet(user_situation: str) -> dict:
    prompt = f"""
아래 상황에 어울리는 꽃다발을 추천해주세요.
반드시 아래 JSON 형식으로만 응답하세요. 다른 설명은 쓰지 마세요.

상황: {user_situation}

응답 형식:
{{
  "title": "꽃다발 제목 (10자 이내)",
  "combined_message": "꽃다발 전체 꽃말 조합 문구 (2~3문장)",
  "flowers": [
    {{
      "name_ko": "꽃 이름(국문)",
      "name_en": "꽃 이름(영문)",
      "meaning": "꽃말(국문)"
    }}
  ]
}}

flowers는 2~4개로 구성하고, 실제로 존재하는 꽃만 사용하세요.
"""

    response = model.generate_content(prompt)
    text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)