# Bloomary-AI

## Directory Structure
- **data/**: 프로젝트에 필요한 데이터 보관
- **models/**: 학습된 YOLO 모델 가중치 보관
- **src/server/**: FastAPI 서버 및 API 관련 소스 코드
- **src/logic/**: LLM 프롬프트 및 추천 로직 코드
- **notebooks/**: 학습 실험 및 데이터 분석 기록 (.ipynb)
- **Dockerfile**: 서버 배포를 위한 도커 설정 파일

## Branch Strategy
1. **main**: 최종 배포 브랜치 
2. **feature/이슈번호-기능명**: 각자 맡은 기능을 개발하는 작업 공간
   - 예: `feature/1-yolov8-fastapi`, `feature/2-llm-prompt`
   - 작업 완료 시 `main` 브랜치로 Pull Request(PR)를 날려주세요.
