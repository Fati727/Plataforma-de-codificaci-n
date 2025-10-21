# Importación de dependencias necesarias para la API
from typing import Union, Dict, Any, List
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import requests
import traceback
import time
from fastapi.responses import JSONResponse
import uuid
import tempfile
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch
from transformers import AutoConfig
from datetime import datetime
import os

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
    modelos = [
        {"id": 1, "nombre": "T1 ENIGH SCIAN"},
        {"id": 2, "nombre": "T1 ENIGH SINCO"},
        #{"id": 3, "nombre": "DistilBERT"}
    ]
    for modelo in get_modelos_disco():
        modelos.append({"id": 100 + len(modelos) + 1, "nombre": modelo})
    return modelos

def get_modelos_disco():
    modelos = []
    base_path = "/models"
    for dir_name in os.listdir(base_path):
        dir_path = os.path.join(base_path, dir_name)
        if os.path.isdir(dir_path) and dir_name.startswith("trainjob_"):
            status_file = os.path.join(dir_path, "status.txt")
            if os.path.isfile(status_file):
                with open(status_file, "r", encoding="utf-8") as f:
                    if "listo" in f.read():
                        modelos.append(dir_name)
    return modelos

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
        tiempo_inicio = time.time()
        y_pred = await codifica_(model_name, df, col_texto)
        delta_tiempo = time.time() - tiempo_inicio
        
        # Calcula métricas de evaluación
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        #cm = confusion_matrix(y_true, y_pred).tolist()

        # Retorna un diccionario con los resultados
        return {
            "model": model_name,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                #"confusion_matrix": cm
            },
            "dataset_stats": {
                "samples": len(df),  # Cantidad de registro
                "features": len(df.columns) - 1 # Número de características (menos la etiqueta)
            },
            "performance": {
                "computation_time_seconds": delta_tiempo,  # Tiempo de cómputo (si se mide)
                "computation_time_per_sample_seconds": delta_tiempo/len(df)  # Tiempo por muestra {tiempo_total / número_de_muestras
            }
        }

# Manejo de errores para distintos problemas comunes
    except pd.errors.EmptyDataError:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="El archivo no tiene codificación UTF-8 válida")
    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo")
    

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
    elif model_name.startswith("trainjob_"):
        y_pred = codifica_local(model_name=model_name, df=df, col_texto=col_texto)
    else:
        raise HTTPException(status_code=400, detail=f"Modelo '{model_name}' no soportado")


    return y_pred

def codifica_local(model_name: str, df: pd.DataFrame, col_texto: str):
    model_route = f"/models/{model_name}/outputs/best_model"
    classes_path = f"/models/{model_name}/classes.npy"

    # Carga las clases originales
    classes = np.load(classes_path, allow_pickle=True)
    le = LabelEncoder()
    le.classes_ = classes

    # Crea el modelo de clasificación
    model = ClassificationModel(
        "bert",
        model_route,
        use_cuda=torch.cuda.is_available(),
        cuda_device=0 if torch.cuda.is_available() else -1,
        ignore_mismatched_sizes=True
    )

    textos = df[col_texto].astype(str).tolist()
    y_pred_raw, _ = model.predict(textos)
    y_pred = le.inverse_transform(y_pred_raw)
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

def freeze_base_and_unfreeze_classifier(skt_model):
    """
    Freeze all parameters then unfreeze classifier-like parameters.
    This tries to match common HF attribute names.
    """
    # Freeze all
    for _, p in skt_model.model.named_parameters():
        p.requires_grad = False

    # Try to unfreeze commonly-named classifier modules
    unfreeze_patterns = ["classifier", "pooler", "out_proj", "score", "classifier_head", "head"]
    for name, p in skt_model.model.named_parameters():
        for pat in unfreeze_patterns:
            if pat in name:
                p.requires_grad = True

    # If the underlying model has attribute .classifier, explicitly unfreeze
    if hasattr(skt_model.model, "classifier"):
        for p in skt_model.model.classifier.parameters():
            p.requires_grad = True

# -------------------------- ENDPOINT DE FINE TUNING  ------------------------------

@router.post("/fine-tuning/")
async def fine_tuning(
    csv_file: UploadFile = File(...),
    text_col: str = Form(...),
    label_col: str = Form(...),
    model_name: str = Form("T1 ENIGH SCIAN"),
    learning_rate: float = Form(5e-5),
    num_train_epochs: int = Form(5),
    train_batch_size: int = Form(8),
    test_size: float = Form(0.1),
):
    # Save uploaded file to temp
    tmp_dir = Path("/models") / f"trainjob_{datetime.now().strftime("%y%m%d_%H%M")}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    write_status("iniciado", tmp_dir)

    # Load original classes for the chosen model
    if model_name == "T1 ENIGH SCIAN":
        model_type = "bert"
        model_route = "/models/enigh-scian-model"
        classes_path = "/models/enigh-scian-classes.npy"
    elif model_name == "T1 ENIGH SINCO":
        model_type = "bert"
        model_route = "/models/enigh-sinco-model"
        classes_path = "/models/enigh-sinco-classes.npy"

        

    try:
        df = await obtener_dataframe_desde_csv(csv_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo CSV: {e}")

    if text_col not in df.columns or label_col not in df.columns:
        raise HTTPException(status_code=400, detail="Columnas proporcionadas no existen en el archivo.")

    df = df[[text_col, label_col]].dropna().rename(columns={text_col: "text", label_col: "label"}).reset_index(drop=True)
    if df.shape[0] < 2:
        raise HTTPException(status_code=400, detail="Dataset muy pequeño para entrenar.")

    # Encode labels
    le = LabelEncoder()
    df["labels"] = le.fit_transform(df["label"])
    num_labels = len(le.classes_)
    classes_path = tmp_dir / "classes.npy"
    np.save(classes_path, le.classes_)

    # Split train/eval
    if df.shape[0] >= 10 and 0.0 < test_size < 0.5:
        train_df, eval_df = train_test_split(df, test_size=test_size)
    else:
        # small dataset: use same for eval
        train_df = df.copy()
        eval_df = df.copy()

    # Create model args
    model_args = ClassificationArgs()
    model_args.learning_rate = learning_rate
    model_args.num_train_epochs = num_train_epochs
    model_args.train_batch_size = train_batch_size
    model_args.reprocess_input_data = False
    model_args.evaluate_during_training = True
    model_args.overwrite_output_dir = True
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.warmup_ratio = 0.06

    # unique output dir for this job
    output_dir = tmp_dir / "outputs"
    model_args.output_dir = str(output_dir)
    model_args.best_model_dir = str(output_dir / "best_model")

    # device
    use_cuda = torch.cuda.is_available()
    cuda_device = 0 if use_cuda else -1

    # force new classifier head
    config = AutoConfig.from_pretrained(model_route, num_labels=num_labels)

    model = ClassificationModel(
        model_type,
        model_route,
        num_labels=num_labels,   # this alone triggers a new head
        args=model_args,
        use_cuda=use_cuda,
        cuda_device=cuda_device,
        ignore_mismatched_sizes=True
    )

    # Freeze base and unfreeze classification head only
    freeze_base_and_unfreeze_classifier(model)

    # Train
    try:
        write_status("entrenando", tmp_dir)
        train_result = model.train_model(train_df, acc=accuracy_score, eval_df=eval_df)
    except Exception as e:
        write_status("error", tmp_dir)
        raise HTTPException(status_code=500, detail=f"Error durante entrenamiento: {e}")
    write_status("listo", tmp_dir)
    # Save encoder classes and return paths and simple summary
    response = {
        "status": "ok",
        "output_dir": str(output_dir),
        "best_model_dir": str(model_args.best_model_dir),
        "classes_npy": str(classes_path),
        "num_train_samples": int(train_df.shape[0]),
        "num_eval_samples": int(eval_df.shape[0]),
        "num_labels": int(num_labels),
    }

    return JSONResponse(response)

def write_status(message: str, tmp_dir: Path):
    status_file = tmp_dir / "status.txt"
    with open(status_file, "w") as f:
        f.write(f"{message}")
# --------------------- REGISTRO DEL ROUTER EN LA APP PRINCIPAL ---------------------
# Se registra el router en la aplicación principal para activar todos los endpoints definidos
app.include_router(router)

    
     


    

    
