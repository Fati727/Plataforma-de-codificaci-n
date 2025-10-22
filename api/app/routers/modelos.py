from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from pydantic import BaseModel
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import traceback
from app.services.training_service import fine_tune_model

router = APIRouter()
from app.services.modelos_service import (
    get_modelos_disco,
    codifica_
)
from app.utils.io_utils import obtener_dataframe_desde_csv

router = APIRouter(prefix="/api/v1")

# --------------------- modelos disponibles ---------------------

class ModeloOut(BaseModel):
    id: int
    nombre: str

@router.get("/modelos-disponibles/", response_model=List[ModeloOut])
async def obtener_modelos_disponibles():
    modelos = [
        {"id": 1, "nombre": "T1 ENIGH SCIAN"},
        {"id": 2, "nombre": "T1 ENIGH SINCO"},
    ]
    for modelo in get_modelos_disco():
        modelos.append({"id": 100 + len(modelos) + 1, "nombre": modelo})
    return modelos


# --------------------- entrenamiento desde cero ---------------------

@router.post("/entrenamiento-desde-cero/")
async def entrenamiento_desde_cero(
    file: UploadFile = File(...),
    columna_texto: str = Form(...),
    columna_clase: str = Form(...)
) -> Dict[str, str]:
    print(f"Archivo recibido: {file.filename}")
    return {
        "status": "ok",
        "archivo": file.filename,
        "columna_texto": columna_texto,
        "columna_clase": columna_clase
    }


# --------------------- evaluación de modelos ---------------------

@router.post("/evaluate-model/")
async def evaluate_model(
    model_name: str = Form(...),
    csv_file: UploadFile = File(...),
    col_texto: str = Form(None),
    col_clasificacion: str = Form(None)
) -> Dict[str, Any]:
    try:
        df = await obtener_dataframe_desde_csv(csv_file)

        if col_texto is None:
            col_texto = df.columns[0]
        if col_clasificacion is None:
            col_clasificacion = df.columns[1]

        if col_texto not in df.columns or col_clasificacion not in df.columns:
            raise HTTPException(status_code=400,
                detail=f"Columnas '{col_texto}' o '{col_clasificacion}' no encontradas en el CSV")

        y_true = df[col_clasificacion].astype(str).tolist()
        tiempo_inicio = time.time()
        y_pred = await codifica_(model_name, df, col_texto)
        delta_tiempo = time.time() - tiempo_inicio

        return {
            "model": model_name,
            "metrics": {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0),
            },
            "dataset_stats": {"samples": len(df), "features": len(df.columns) - 1},
            "performance": {
                "computation_time_seconds": delta_tiempo,
                "computation_time_per_sample_seconds": delta_tiempo / len(df)
            }
        }

    except pd.errors.EmptyDataError:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="El archivo no tiene codificación UTF-8 válida")
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error al procesar el archivo")


# --------------------- codificación con modelo ---------------------

@router.post("/modelos/{model_name}/codifica/")
async def codifica(model_name: str, col_texto: str, csv_file: UploadFile = File(...)):
    df = await obtener_dataframe_desde_csv(csv_file)
    inicio = time.time()
    resultado = await codifica_(model_name, df, col_texto)
    delta = time.time() - inicio
    df["resultado_codificacion"] = resultado
    return {
        "model": model_name,
        "dataset": df.to_dict(orient="split"),
        "performance": {
            "computation_time_seconds": delta,
            "computation_time_per_sample_seconds": delta / len(df)
        }
    }


@router.post("/fine-tuning/")
async def fine_tuning_endpoint(
    csv_file: UploadFile = File(...),
    text_col: str = Form(...),
    label_col: str = Form(...),
    model_name: str = Form("T1 ENIGH SCIAN"),
    learning_rate: float = Form(5e-5),
    num_train_epochs: int = Form(5),
    train_batch_size: int = Form(8),
    test_size: float = Form(0.1),
):
    result = fine_tune_model(
        csv_file,
        text_col,
        label_col,
        model_name,
        learning_rate,
        num_train_epochs,
        train_batch_size,
        test_size,
    )
    return JSONResponse(result)
