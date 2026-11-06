#FROM python:3.10.6
# Base must match the torch pin in requirements.txt: the old
# pytorch/pytorch:2.0.1-... base baked vulnerable torch into the image layer,
# so bumping requirements.txt alone left the container exposed.
FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

WORKDIR /subsai

COPY requirements.txt .

ENV DEBIAN_FRONTEND=noninteractive
# The 2.13 base is Ubuntu 24.04: PEP 668 marks its system Python as
# externally managed, and this image is a purpose-built app container,
# so pip installs into it directly as before.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get -y install gcc g++ cmake mono-mcs && \
    rm -rf /var/lib/apt/lists/*

# The base image is Python 3.12: pandas 1.5.3 and openai-whisper have no
# cp312 wheels, so their sdists build against a pre-installed toolchain
# (setuptools<81: their setup.py imports pkg_resources, removed in
# setuptools>=81) with --no-build-isolation.
# CMAKE_POLICY_VERSION_MINIMUM: the base ships CMake 4.x, which rejects the
# old pybind11 vendored by pywhispercpp 1.1.1 (cmake_minimum_required < 3.5).
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN pip install --no-cache-dir "setuptools<81" wheel "cython>=0.29.33,<3" "numpy<2" && \
    grep -v faster_whisper requirements.txt > requirements-nofw.txt && \
    pip install --no-cache-dir --no-build-isolation -r requirements-nofw.txt

# faster-whisper 0.5.1 pins av<11; av 10.x has no cp312 wheels and its sdist
# needs ffmpeg dev headers. Install a modern av wheel plus faster-whisper's
# other deps, then faster-whisper itself with --no-deps (same workaround as
# .github/workflows/tests.yml; av>=14 works fine).
RUN pip install --no-cache-dir "av>=14" ctranslate2 tokenizers huggingface_hub onnxruntime && \
    pip install --no-cache-dir --no-deps faster-whisper==0.5.1

COPY pyproject.toml .
COPY ./src ./src
COPY ./assets ./assets

# All runtime deps were installed above with the cp312 workarounds; --no-deps
# stops this from re-resolving requirements.txt (which would pull av<11 again).
RUN pip install . --no-deps

CMD ["python", "src/subsai/webui.py"]

