FROM python:3.8-slim-buster
WORKDIR /app
COPY . .
RUN apt-get update \
    && apt-get install -y g++ cmake imagemagick ffmpeg \
    && pip3 install -r requirements.txt \
    && apt-get remove -y g++ cmake \
    && apt-get clean
VOLUME ["/app/payload/"]
CMD [ "./start.sh"]
