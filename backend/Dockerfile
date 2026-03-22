FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential python3-dev gfortran cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Install CPU-only PyTorch first to avoid downloading the large CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# pybinding 0.9.5 bundles pybind11 2.2.1 (2017) which doesn't support Python 3.9+.
# Pre-install a modern pybind11 and build pybinding without isolation so its
# setup.py picks up the system pybind11 instead of the bundled old copy.
RUN pip install --no-cache-dir "pybind11>=2.10"
RUN pip install --no-cache-dir pybinding==0.9.5 --no-build-isolation

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
