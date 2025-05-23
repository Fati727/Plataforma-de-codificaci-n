# Usa una imagen oficial de Python 3.12 slim
FROM python:3.12-slim

# Copia los archivos de requerimientos
COPY requirements.txt .

# Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de la aplicación
#COPY ./app app

# Establece el directorio de trabajo
#WORKDIR /app

# Expone el puerto en el que correrá la app
EXPOSE 8000

# Comando por defecto para ejecutar la app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

