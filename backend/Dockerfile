FROM continuumio/miniconda3:24.1.2-0

RUN apt-get update && apt-get install -y \
    build-essential gfortran cmake \
    && rm -rf /var/lib/apt/lists/*

# Install pybinding from conda-forge, which provides pre-built binaries with
# a modern pybind11 — the PyPI source distribution only builds on Python <=3.8.
RUN conda install -y -c conda-forge python=3.10 pybinding && \
    conda clean -afy

WORKDIR /app
COPY requirements.txt .

# Install CPU-only PyTorch first to avoid downloading the large CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
