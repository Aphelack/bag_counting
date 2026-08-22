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
#   - mmcv/mmdet's own deps drag in plain opencv-python and numpy>=2,
#     which respectively conflict with opencv-python-headless (breaks cv2)
#     and our torch build (compiled against NumPy 1.x) — force both back
#     as the last step.
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir "setuptools<81" openmim \
    && mim install mmengine "mmcv>=2.0.0,<2.2.0" mmdet \
    && pip install --no-cache-dir "numpy<2.0" \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall opencv-python-headless

COPY app ./app

ENV DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
