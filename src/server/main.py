import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.logic.detector import detect_flowers_for_api
from src.logic.llm_handler import combine_floriography, recommend_bouquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()


class FlowerForCombination(BaseModel):
    name_ko: str = Field(..., alias="nameKo")
    name_en: str = Field(..., alias="nameEn")
    meaning: str


class CombineFloriographyRequest(BaseModel):
    flowers: list[FlowerForCombination]


class RecommendBouquetRequest(BaseModel):
    user_situation: str


def error_response(status_code: int, error_code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error_code": error_code, "message": message},
    )


def raise_invalid_request(message: str) -> None:
    raise HTTPException(
        status_code=400,
        detail={"error_code": "INVALID_REQUEST", "message": message},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail

    if isinstance(detail, dict) and "error_code" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": "INTERNAL_SERVER_ERROR" if exc.status_code >= 500 else "INVALID_REQUEST",
            "message": str(detail),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("요청 검증 실패: %s", exc)

    return error_response(
        status_code=400,
        error_code="INVALID_REQUEST",
        message="잘못된 요청 파라미터입니다.",
    )


@app.exception_handler(Exception)
async def internal_exception_handler(request: Request, exc: Exception):
    logger.exception("서버 내부 오류: %s", exc)

    return error_response(
        status_code=500,
        error_code="INTERNAL_SERVER_ERROR",
        message="서버 내부 오류가 발생했습니다.",
    )


@app.post("/ai/detect-flowers")
async def detect_flowers(image: UploadFile = File(...)):
    logger.info(
        "요청 수신 — filename=%s, content_type=%s, size=%s",
        image.filename,
        image.content_type,
        image.size,
    )

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail={"error_code": "INVALID_REQUEST", "message": "이미지 파일을 업로드해야 합니다."},
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "INVALID_REQUEST", "message": "빈 파일입니다."},
        )

    try:
        result = await detect_flowers_for_api(image_bytes)
    except Exception as e:
        logger.exception("AI 처리 실패: %s", e)
        raise HTTPException(
            status_code=422,
            detail={"error_code": "AI_PROCESSING_ERROR", "message": str(e)},
        )

    return JSONResponse(content=result)


@app.post("/ai/combine-floriography")
async def combine_floriography_endpoint(payload: CombineFloriographyRequest):
    logger.info(
        "요청 수신 — endpoint=/ai/combine-floriography, flower_count=%s",
        len(payload.flowers),
    )

    if not payload.flowers:
        raise_invalid_request("flowers는 1개 이상이어야 합니다.")

    flowers: list[dict[str, str]] = []

    for flower in payload.flowers:
        name_ko = flower.name_ko.strip()
        name_en = flower.name_en.strip()
        meaning = flower.meaning.strip()

        if not name_ko or not name_en or not meaning:
            raise_invalid_request("flowers의 nameKo, nameEn, meaning은 모두 필수입니다.")

        flowers.append({
            "name_ko": name_ko,
            "name_en": name_en,
            "meaning": meaning,
        })

    try:
        result = await asyncio.to_thread(combine_floriography, flowers)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("AI 처리 실패: %s", e)
        raise HTTPException(
            status_code=422,
            detail={"error_code": "AI_PROCESSING_ERROR", "message": str(e)},
        )

    return JSONResponse(content=result)


@app.post("/ai/recommend-bouquet")
async def recommend_bouquet_endpoint(payload: RecommendBouquetRequest):
    user_situation = payload.user_situation.strip()

    logger.info(
        "요청 수신 — endpoint=/ai/recommend-bouquet, situation_length=%s",
        len(user_situation),
    )

    if not user_situation:
        raise_invalid_request("user_situation은 필수입니다.")

    try:
        result = await asyncio.to_thread(recommend_bouquet, user_situation)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("AI 처리 실패: %s", e)
        raise HTTPException(
            status_code=422,
            detail={"error_code": "AI_PROCESSING_ERROR", "message": str(e)},
        )

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server.main:app", host="0.0.0.0", port=8000, reload=True)