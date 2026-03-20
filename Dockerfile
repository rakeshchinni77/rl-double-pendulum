FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /app

# System packages required by pygame/pymunk + plotting/gif generation.
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	gcc \
	git \
	libgl1 \
	libglib2.0-0 \
	libsdl2-2.0-0 \
	libsdl2-image-2.0-0 \
	libsdl2-mixer-2.0-0 \
	libsdl2-ttf-2.0-0 \
	libx11-6 \
	libxext6 \
	libxrender1 \
	libxrandr2 \
	libxi6 \
	libxcursor1 \
	ffmpeg \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
	python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 && \
	grep -v '^torch==' /app/requirements.txt > /app/requirements-no-torch.txt && \
	python -m pip install -r /app/requirements-no-torch.txt

COPY . /app

# Make python imports stable in container execution.
ENV PYTHONPATH=/app

CMD ["python", "train.py"]
