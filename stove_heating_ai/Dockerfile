# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile
ARG BUILD_FROM
FROM $BUILD_FROM

WORKDIR /usr/src/app

RUN pip install uv

COPY app/requirements.lock ./

RUN uv pip install --no-cache --system -r ./requirements.lock

COPY app/src ./src

ENV CUDA_VISIBLE_DEVICES=""

CMD [ "python", "src/main.py" ]
