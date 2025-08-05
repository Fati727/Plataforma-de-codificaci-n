# Importación de dependencias necesarias para la API
from typing import Union, Dict, Any, List
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import requests

# Se crea una instancia principal de la aplicación FastAPI
app = FastAPI()
# Se crea un router con prefijo /api/v1 para agrupar los endpoints
router = APIRouter(prefix="/api/v1")

# Middleware CORS para permitir el acceso desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- ENDPOINTS BÁSICOS ----------------------------------

@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# --------------------------- ENDPOINT DE SELECCIÓN DE MODELO ----------------------
# Modelo de datos para representar un modelo (nombre e id)
class Modelo(BaseModel):
    nombre: str
    id: int
# Lista de modelos de lenguaje disponibles
modelos_disponibles = [
    {"id": 1, "nombre": "BERT"},
    {"id": 2, "nombre": "RoBERTa"},
    {"id": 3, "nombre": "DistilBERT"}
]

# Endpoint que retorna los modelos disponibles
@router.get("/modelos-disponibles/", response_model=List[Modelo])
async def obtener_modelos_disponibles():
    modelos = [{"nombre": modelo["nombre"], "id": modelo["id"]} for modelo in modelos_disponibles]
    return modelos

# -------------------------- ENDPOINT DE FINE TUNING  ------------------------------

@router.post("/fine-tuning/")
async def fine_tuning(
    file: UploadFile = File(...),     # Archivo CSV subido     
    columna_texto: str = Form(...),   # Nombre de la columna de texto
    columna_clase: str = Form(...)    # Nombre de la columna de clasificacion
) -> Dict[str, str]:
    
    # Muestra información del archivo recibido
    print(f"Archivo recibido: {file.filename}")
    print(f"Columna de texto: {columna_texto}")
    print(f"Columna de clasificación: {columna_clase}")
    
    # Retorna una respuesta de confirmación con los datos
    return {
        "status": "ok",
        "archivo": file.filename,
        "columna_texto": columna_texto,
        "columna_clase": columna_clase
    }

# -------------------------- ENDPOINT DE ENTRENAMIENTO DESDE CERO  -----------------

@router.post("/entrenamiento-desde-cero/")
async def entrenamiento_desde_cero(
    file: UploadFile = File(...),
    columna_texto: str = Form(...),
    columna_clase: str = Form(...)
) -> Dict[str, str]:
    
    # Muestra información del archivo recibido
    print(f"Archivo recibido: {file.filename}")
    print(f"Columna de texto: {columna_texto}")
    print(f"Columna de clasificación: {columna_clase}")
    
     # Retorna confirmación
    return {
        "status": "ok",
        "archivo": file.filename,
        "columna_texto": columna_texto,
        "columna_clase": columna_clase
    }

# ------------------------- ENDPOINT DE EVALUACIÓN DE MODELOS ----------------------

@router.post("/evaluate-model/")
async def evaluate_model(
    model_name: str = Form(...),          # Nombre del modelo a evaluar
    csv_file: UploadFile = File(...),      # Archivo CSV con datos de prueba
    col_texto: str = Form(None),           #Columna texto
    col_clasificacion: str = Form(None)    #Columna Clasificacion
) -> Dict[str, Any]:
    try:
         # Verifica que el archivo sea un CSV válido
        if not csv_file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")

        # Lee el contenido del archivo
        contents = await csv_file.read()
        data = io.StringIO(contents.decode('utf-8'))
        df = pd.read_csv(data)

        # Asigna columnas por defecto si no se especifican
        if col_texto is None:
            col_texto = df.columns[0]
        if col_clasificacion is None:
            col_clasificacion = df.columns[1]

        # Verifica que las columnas existan en el archivo
        if col_texto not in df.columns or col_clasificacion not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Columnas '{col_texto}' o '{col_clasificacion}' no encontradas en el CSV"
            )
        
        # Función interna para obtener y_true y y_pred
        def obtener_valores_y(df: pd.DataFrame, col_clasificacion: str, col_texto: str):
            y_true = df[col_clasificacion].astype(str).tolist() #Valores reales

            try:
                # Se envían los textos al servicio del INEGI para codificación automática
                textos = df[col_texto].astype(str).tolist()
                payload = {"data": textos}
                url_inegi = "http://lcidbind.inegi.gob.mx:5194/api/codificacion/enigh/t1/scian"
                respuesta = requests.post(url_inegi, json=payload)
                respuesta.raise_for_status()
                resultado_inegi = respuesta.json()
                
                # Obtiene la primera etiqueta por texto
                codificaciones = []
                for etiquetas in resultado_inegi.get("tags", []):
                    if etiquetas:
                        codificaciones.append(etiquetas[0])  # Solo el código
            except requests.exceptions.RequestException as e:
                codificaciones = []
                resultado_inegi = {"errors": [f"Error al conectar con INEGI: {str(e)}"]}

            y_pred = codificaciones # Valores predicho
            return y_true, y_pred
        
        # Llama a la función para obtener las etiquetas reales y predichas
        y_true, y_pred = obtener_valores_y(df, col_clasificacion, col_texto)
        
        # Calcula métricas de evaluación
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()

        # Retorna un diccionario con los resultados
        return {
            "model": model_name,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": cm
            },
            "dataset_stats": {
                "samples": len(df),  # Cantidad de registro
                "features": len(df.columns) - 1 # Número de características (menos la etiqueta)

            }
        }

# Manejo de errores para distintos problemas comunes
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="El archivo no tiene codificación UTF-8 válida")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")

# --------------------- REGISTRO DEL ROUTER EN LA APP PRINCIPAL ---------------------
# Se registra el router en la aplicación principal para activar todos los endpoints definidos
app.include_router(router)

    
     


    

    
