from typing import Union, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #dominio frontend
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
    model_name: str , 
    csv_file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Endpoint para evaluar modelos de ML
    
    Args:
        model_name: Nombre del modelo a evaluar (ahora requerido)
        csv_file: Archivo CSV con datos de evaluación
    
    Returns:
        Métricas de evaluación en formato JSON
    """
    try:
        # Validar extensión del archivo
        if not csv_file.filename.lower().endswith ('.csv'):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")
        
        # Leer y procesar CSV
        contents = await csv_file.read()
        data = io.StringIO(contents.decode('utf-8'))
        df = pd.read_csv(data)
        
        # Debug: mostrar primeros 5 elementos
        print(f"\nEvaluando modelo: {model_name}")
        print("Muestra de datos (5 primeros registros):")
        print(df.head())
        
        # Generar métricas ficticias (ejemplo)
        metrics = {
            "model": model_name,
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.93,
                "f1_score": 0.935,
                "confusion_matrix": {
                    "true_positives": 45,
                    "false_positives": 2,
                    "true_negatives": 48,
                    "false_negatives": 5
                }
            },
            "dataset_stats": {
                "samples": len(df),
                "features": len(df.columns) - 1 if len(df.columns) > 0 else 0
            }
        }
        
        return metrics
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="El archivo no tiene codificación UTF-8 válida")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")