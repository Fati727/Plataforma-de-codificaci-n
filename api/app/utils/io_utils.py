import io
import pandas as pd
from fastapi import UploadFile, HTTPException

async def obtener_dataframe_desde_csv(csv_file: UploadFile) -> pd.DataFrame:
    if not csv_file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")
    contents = await csv_file.read()
    data = io.StringIO(contents.decode('utf-8'))
    return pd.read_csv(data)
