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

# Endpoint que retorna los modelos disponibles
@router.get("/modelos-disponibles/", response_model=List[Modelo])
async def obtener_modelos_disponibles():
# Lista de modelos de lenguaje disponibles
    modelos_disponibles = [
        {"id": 1, "nombre": "T1 ENIGH SCIAN"},
        {"id": 2, "nombre": "T1 ENIGH SINCO"},
        #{"id": 3, "nombre": "DistilBERT"}
    ]
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
        df = await obtener_dataframe_desde_csv(csv_file)

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
        
        y_true = df[col_clasificacion].astype(str).tolist() #Valores reales
        
        # Llama a la función para obtener las etiquetas reales y predichas
        y_pred = await codifica_(model_name, df, col_texto)
        
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
async def obtener_dataframe_desde_csv(csv_file: UploadFile) -> pd.DataFrame:
    # Verifica que el archivo sea un CSV válido
    if not csv_file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")

    contents = await csv_file.read()
    data = io.StringIO(contents.decode('utf-8'))
    df = pd.read_csv(data)
    return df


# Función interna para obtener y_true y y_pred
#def obtener_valores_y(df: pd.DataFrame, col_clasificacion: str, col_texto: str):
@router.post("/modelos/{model_name}/codifica/")
async def codifica(
    model_name: str,          # Nombre del modelo a usar
    col_texto: str,           #Columna texto
    csv_file: UploadFile = File(...),      # Archivo CSV con datos de prueba
):
    df = await obtener_dataframe_desde_csv(csv_file)

    tiempo_inicio = time.time()
    resultado = await codifica_(model_name, df, col_texto)
    delta_tiempo = time.time() - tiempo_inicio

    df["resultado_codficacion"] = resultado
    return {
            "model": model_name,
            "dataset": df.to_dict(orient="split"),
            "performance": {
                "computation_time_seconds": delta_tiempo,  # Tiempo de cómputo (si se mide)
                "computation_time_per_sample_seconds": delta_tiempo/len(df)  # Tiempo por muestra {tiempo_total / número_de_muestras
            }
        }

    return 

async def codifica_(
    model_name: str = Form(...),          # Nombre del modelo a usar
    df: UploadFile = File(...),      # Archivo CSV con datos de prueba
    col_texto: str = Form(None),           #Columna texto
):
    if model_name == "T1 ENIGH SCIAN":
        y_pred = obtener_enigh_t1(model="scian", df=df, col_texto=col_texto)
    elif model_name == "T1 ENIGH SINCO":
        y_pred = obtener_enigh_t1(model="sinco", df=df, col_texto=col_texto)
    else:
        raise HTTPException(status_code=400, detail=f"Modelo '{model_name}' no soportado")


    return y_pred

def obtener_enigh_t1(
    model: str,          # Nombre del modelo a usar
    df: pd.DataFrame ,      # Archivo CSV con datos de prueba
    col_texto: str,           #Columna texto
):
    try:
        # Se envían los textos al servicio del INEGI para codificación automática
        textos = df[col_texto].astype(str).tolist()
        payload = {"data": textos}
        url_inegi = f"http://lcidbind.inegi.gob.mx:5194/api/codificacion/enigh/t1/{model}"
        respuesta = requests.post(url_inegi, json=payload)
        respuesta.raise_for_status()
        resultado_inegi = respuesta.json()
        
        # Obtiene la primera etiqueta por texto
        y_pred = []
        for etiquetas in resultado_inegi.get("tags", []):
            if etiquetas:
                y_pred.append(etiquetas[0])  # Solo el código
    except requests.exceptions.RequestException as e:
        y_pred = []
        resultado_inegi = {"errors": [f"Error al conectar con INEGI: {str(e)}"]}
    return y_pred

# --------------------- REGISTRO DEL ROUTER EN LA APP PRINCIPAL ---------------------
# Se registra el router en la aplicación principal para activar todos los endpoints definidos
app.include_router(router)

    
     


    

    
