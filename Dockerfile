FROM python:3.8-slim-buster as builder

COPY ./requirements.txt .

RUN apt-get update && apt-get install -y g++ cmake

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.8.2-slim

LABEL maintainer="leachim2k@leachim2k.de"

WORKDIR /app

COPY --from=builder /root/.local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY . .

RUN apt-get update \
    && apt-get install -y ffmpeg  \
    && apt-get clean
VOLUME ["/app/payload/"]
CMD [ "./start.sh"]
