import os
import numpy as np
from pathlib import Path
import torch
from simpletransformers.classification import ClassificationModel
from sklearn.preprocessing import LabelEncoder
import requests
from fastapi import HTTPException, Form
from datetime import datetime

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


async def codifica_(model_name: str, df, col_texto: str):
    if model_name == "T1 ENIGH SCIAN":
        return obtener_enigh_t1("scian", df, col_texto)
    elif model_name == "T1 ENIGH SINCO":
        return obtener_enigh_t1("sinco", df, col_texto)
    elif model_name.startswith("trainjob_"):
        return codifica_local(model_name, df, col_texto)
    else:
        raise HTTPException(status_code=400, detail=f"Modelo '{model_name}' no soportado")


def codifica_local(model_name: str, df, col_texto: str):
    model_route = f"/models/{model_name}/outputs/best_model"
    classes_path = f"/models/{model_name}/classes.npy"
    classes = np.load(classes_path, allow_pickle=True)
    le = LabelEncoder()
    le.classes_ = classes
    model = ClassificationModel("bert", model_route,
        use_cuda=torch.cuda.is_available(),
        cuda_device=0 if torch.cuda.is_available() else -1,
        ignore_mismatched_sizes=True)
    textos = df[col_texto].astype(str).tolist()
    y_pred_raw, _ = model.predict(textos)
    return le.inverse_transform(y_pred_raw)


def obtener_enigh_t1(model: str, df, col_texto: str):
    try:
        textos = df[col_texto].astype(str).tolist()
        url = f"http://lcidbind.inegi.gob.mx:5194/api/codificacion/enigh/t1/{model}"
        resp = requests.post(url, json={"data": textos})
        resp.raise_for_status()
        result = resp.json()
        y_pred = []
        for etiquetas in result.get("tags", []):
            if etiquetas:
                y_pred.append(etiquetas[0])
        return y_pred
    except requests.exceptions.RequestException as e:
        return []


def write_status(message: str, tmp_dir: Path):
    with open(tmp_dir / "status.txt", "w") as f:
        f.write(message)

