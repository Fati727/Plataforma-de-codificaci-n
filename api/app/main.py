from typing import Union, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/evaluate-model/") 
async def evaluate_model(
    # model_name: str = Form(...), 
    csv_file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Endpoint para evaluar modelos de ML
    
    Args:
        model_name: Nombre del modelo a evaluar
        csv_file: Archivo CSV con datos de evaluación
    
    Returns:
        Métricas de evaluación en formato JSON
    """
    try:
        # Validar extensión del archivo
        if not csv_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")
        
        # Leer y procesar CSV
        contents = await csv_file.read()
        data = io.StringIO(contents.decode('utf-8'))
        df = pd.read_csv(data)
        
        # Debug: mostrar primeros 10 elementos
        print(f"\nEvaluando modelo: {model_name}")
        print("Muestra de datos (10 primeros registros):")
        print(df.head(10))
        
        # Generar métricas ficticias
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
                "features": len(df.columns) - 1  # Asumiendo última columna es target
            }
        }
        
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")