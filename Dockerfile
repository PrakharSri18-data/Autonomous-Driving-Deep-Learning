FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow/opencv-style image libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY pyproject.toml .

ENV PYTHONPATH=/app/src/part1DeepLearning

WORKDIR /app/src/part1DeepLearning

# Default: run inference. Override with `docker run <image> python train.py ...` etc.
# Requires a mounted volume with the dataset (see README "Dataset" section — the
# raw images are not redistributed in this image due to dataset license/size).
ENTRYPOINT ["python"]
CMD ["inference.py", "--help"]
