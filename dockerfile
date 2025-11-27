# 1. 빌드 스테이지 (Build Stage)
FROM python:3.11-slim as builder

WORKDIR /app

COPY requirements.txt .

# 의존성 패키지를 설치합니다.
RUN pip install --no-cache-dir -r requirements.txt

# 2. 최종 스테이지 (Final Stage)
FROM python:3.11-slim

WORKDIR /app

# 빌드 스테이지에서 설치된 파이썬 패키지들을 최종 이미지로 복사합니다.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 애플리케이션 코드를 복사합니다.
COPY . .

EXPOSE 8000


# [Development CMD 대체 옵션] 개발 목적으로만 사용하려면 다음 CMD를 사용하세요.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]