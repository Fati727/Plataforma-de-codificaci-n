from typing import Union, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = FastAPI()

# Middleware CORS para permitir el acceso desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/evaluate-model/")
async def evaluate_model(
    model_name: str,
    csv_file: UploadFile = File(...),
    col_texto: str = "y_pred",
    col_clasificacion: str = "y_true"
) -> Dict[str, Any]:
    """
    Endpoint para evaluar modelos de ML con métricas reales.
    """
    try:
        if not csv_file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")

        contents = await csv_file.read()
        data = io.StringIO(contents.decode('utf-8'))
        df = pd.read_csv(data)

        # Verificación de columnas requeridas
        if col_texto not in df.columns or col_clasificacion not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Columnas '{col_texto}' o '{col_clasificacion}' no encontradas en el CSV"
            )

        y_true = df[col_clasificacion]
        y_pred = df[col_texto]

        # Calcular métricas reales
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)

        # Estructura de salida
        metrics = {
            "model": model_name,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": {
                    "true_positives": int(cm[1][1]),
                    "false_positives": int(cm[0][1]),
                    "true_negatives": int(cm[0][0]),
                    "false_negatives": int(cm[1][0])
                }
            },
            "dataset_stats": {
                "samples": len(df),
                "features": len(df.columns) - 1
            }
        }

        return metrics

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="El archivo no tiene codificación UTF-8 válida")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")
