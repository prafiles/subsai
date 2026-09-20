#FROM python:3.10.6
# Base must match the torch pin in requirements.txt: the old
# pytorch/pytorch:2.0.1-... base baked vulnerable torch 2.0.1 into the image
# layer, so bumping requirements.txt alone left the container exposed.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /subsai

COPY requirements.txt .

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY ./src ./src
COPY ./assets ./assets

RUN pip install .

CMD ["python", "src/subsai/webui.py"]

