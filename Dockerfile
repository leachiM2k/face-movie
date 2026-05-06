FROM python:3.12-slim

LABEL maintainer="leachim2k@leachim2k.de"

# ffmpeg + minimal libs for opencv-python and mediapipe
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py start.sh ./

VOLUME ["/app/payload/"]
CMD ["./start.sh"]
