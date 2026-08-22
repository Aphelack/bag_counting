FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# RTMDet detector stack (MMDetection), pinned/ordered exactly as validated
# in training/ (see training/README.md for why each step is needed):
#   - torch/torchvision pinned to a version OpenMMLab has prebuilt mmcv
#     wheels for (their index tops out well below the latest torch/CUDA).
#   - setuptools<81 + openmim: mim's CLI imports pkg_resources, removed
#     from setuptools 81+.
#   - mim install (not pip): picks the correct prebuilt mmcv wheel for the
#     installed torch/CUDA build.
#   - mmcv/mmdet's own deps drag in plain opencv-python, which conflicts
#     with opencv-python-headless already installed (breaks cv2) — forced
#     back with --no-deps so reinstalling it can't drag numpy along.
#   - numpy pinned back to <2.0 LAST, and only after the opencv fix: our
#     torch build was compiled against NumPy 1.x, and reinstalling
#     opencv-python-headless without --no-deps would silently pull numpy
#     back to 2.x, undoing an earlier pin — hence both --no-deps here AND
#     doing this step last.
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir "setuptools<81" openmim \
    && mim install mmengine "mmcv>=2.0.0,<2.2.0" mmdet \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless \
    && pip install --no-cache-dir "numpy<2.0"

COPY app ./app

ENV DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
