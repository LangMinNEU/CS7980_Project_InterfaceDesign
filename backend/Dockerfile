FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential gfortran cmake curl \
    && rm -rf /var/lib/apt/lists/*

# pybinding 0.9.5 bundles pybind11 ~2.4 which predates Python 3.9.
# Download the sdist, swap in pybind11 v2.11.1, then build from source.
RUN pip download --no-deps pybinding==0.9.5 -d /tmp/pb_dl && \
    cd /tmp && tar xzf pb_dl/pybinding-0.9.5.tar.gz && \
    curl -fsSL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz \
         -o /tmp/pybind11.tar.gz && \
    tar xzf /tmp/pybind11.tar.gz && \
    rm -rf pybinding-0.9.5/pybind11 && \
    mv pybind11-2.11.1 pybinding-0.9.5/pybind11 && \
    pip install --no-cache-dir /tmp/pybinding-0.9.5 && \
    rm -rf /tmp/pb_dl /tmp/pybinding-0.9.5 /tmp/pybind11.tar.gz

WORKDIR /app
COPY requirements.txt .

# Install CPU-only PyTorch first to avoid downloading the large CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
