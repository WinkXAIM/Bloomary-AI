import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.logic.detector import detect_flowers_for_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server.main:app", host="0.0.0.0", port=8000, reload=True)
