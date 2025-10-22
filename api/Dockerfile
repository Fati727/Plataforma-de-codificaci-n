# Usa una imagen oficial de Python 3.12 slim
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /api

# Copia los archivos de requerimientos primero (para aprovechar la cache)
COPY requirements.txt .

# system build deps (small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU then other requirements
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código fuente de la aplicación
COPY . .

# Expone el puerto 8000 (FastAPI/uvicorn)
EXPOSE 8000

# Use main:app because main.py lives at the container root (/api/main.py)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
