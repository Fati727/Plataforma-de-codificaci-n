from typing import Union, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import requests

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

@app.post("/fine-tuning/")
async def fine_tuning(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Endpoint de Fine Tuning.
    Recibe un archivo y responde con 'ok'.
    """
    return {"status": "ok"}


@app.post("/evaluate-model/")
async def evaluate_model(
    model_name: str,
    csv_file: UploadFile = File(...),
    col_texto: str = None,
    col_clasificacion: str = None
) -> Dict[str, Any]:
    """
    Endpoint para evaluar modelos de ML con métricas reales y consultar etiquetas desde la API del INEGI.
    """
    try:
        if not csv_file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")

        contents = await csv_file.read()
        data = io.StringIO(contents.decode('utf-8'))
        df = pd.read_csv(data)

        if col_texto is None:
            col_texto = df.columns[0]
        if col_clasificacion is None:
            col_clasificacion = df.columns[1]

        if col_texto not in df.columns or col_clasificacion not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Columnas '{col_texto}' o '{col_clasificacion}' no encontradas en el CSV"
            )


        def obtener_valores_y(df: pd.DataFrame, col_clasificacion: str, col_texto: str):
            y_true = df[col_clasificacion].astype(str).tolist()
                
            #  Llamar a la API de INEGI 
            try:
                textos = df[col_texto].astype(str).tolist()
                payload = {"data": textos}
                url_inegi = "http://lcidbind.inegi.gob.mx:5194/api/codificacion/enigh/t1/sinco"
                respuesta = requests.post(url_inegi, json=payload)
                respuesta.raise_for_status()
                resultado_inegi = respuesta.json()

                codificaciones = []
                for etiquetas in resultado_inegi.get("tags", []):
                    if etiquetas:
                        e = etiquetas[0]
                        #codificaciones.append({"codigo": e[0], "score": e[1]})
                        codificaciones.append(e)

            except requests.exceptions.RequestException as e:
                codificaciones = []
                resultado_inegi = {"errors": [f"Error al conectar con INEGI: {str(e)}"]}
            y_pred = codificaciones  # Placeholder, aquí puedes poner tu modelo real
            return y_true, y_pred

        y_true, y_pred = obtener_valores_y(df, col_clasificacion, col_texto)

        print(y_true)
        print(y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()

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
                "samples": len(df),
                "features": len(df.columns) - 1
            },
            #"codificaciones_inegi": codificaciones,
            #"advertencias_inegi": resultado_inegi.get("errors", [])
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="El archivo no tiene codificación UTF-8 válida")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")

    

    
