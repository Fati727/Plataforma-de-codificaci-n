from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from simpletransformers.classification import (
    ClassificationModel,
    ClassificationArgs,
)
from transformers import AutoConfig
from fastapi import HTTPException

from app.utils.io_utils import obtener_dataframe_desde_csv


def fine_tune_model(
    csv_file,
    text_col: str,
    label_col: str,
    model_name: str = "T1 ENIGH SCIAN",
    learning_rate: float = 5e-5,
    num_train_epochs: int = 5,
    train_batch_size: int = 8,
    test_size: float = 0.1,
):
    """
    Ejecuta el proceso de fine-tuning y devuelve los metadatos del entrenamiento.
    """
    tmp_dir = Path("/models") / f"trainjob_{datetime.now().strftime('%y%m%d_%H%M')}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    write_status("iniciado", tmp_dir)

    # Selección del modelo base
    if model_name == "T1 ENIGH SCIAN":
        model_type = "bert"
        model_route = "/models/enigh-scian-model"
    elif model_name == "T1 ENIGH SINCO":
        model_type = "bert"
        model_route = "/models/enigh-sinco-model"
    else:
        raise HTTPException(status_code=400, detail="Modelo no reconocido.")

    try:
        df = obtener_dataframe_desde_csv(csv_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo CSV: {e}")

    if text_col not in df.columns or label_col not in df.columns:
        raise HTTPException(status_code=400, detail="Columnas proporcionadas no existen en el archivo.")

    df = df[[text_col, label_col]].dropna().rename(columns={text_col: "text", label_col: "label"}).reset_index(drop=True)
    if df.shape[0] < 2:
        raise HTTPException(status_code=400, detail="Dataset muy pequeño para entrenar.")

    # Codificación de etiquetas
    le = LabelEncoder()
    df["labels"] = le.fit_transform(df["label"])
    num_labels = len(le.classes_)
    classes_path = tmp_dir / "classes.npy"
    np.save(classes_path, le.classes_)

    # División de datos
    if df.shape[0] >= 10 and 0.0 < test_size < 0.5:
        train_df, eval_df = train_test_split(df, test_size=test_size)
    else:
        train_df = df.copy()
        eval_df = df.copy()

    # Configuración de argumentos del modelo
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

    # Directorios de salida
    output_dir = tmp_dir / "outputs"
    model_args.output_dir = str(output_dir)
    model_args.best_model_dir = str(output_dir / "best_model")

    # Configuración del dispositivo
    use_cuda = torch.cuda.is_available()
    cuda_device = 0 if use_cuda else -1

    # Nueva cabeza clasificadora
    config = AutoConfig.from_pretrained(model_route, num_labels=num_labels)

    model = ClassificationModel(
        model_type,
        model_route,
        num_labels=num_labels,
        args=model_args,
        use_cuda=use_cuda,
        cuda_device=cuda_device,
        ignore_mismatched_sizes=True,
    )

    freeze_base_and_unfreeze_classifier(model)

    # Entrenamiento
    try:
        write_status("entrenando", tmp_dir)
        model.train_model(train_df, acc=accuracy_score, eval_df=eval_df)
    except Exception as e:
        write_status("error", tmp_dir)
        raise HTTPException(status_code=500, detail=f"Error durante entrenamiento: {e}")

    write_status("listo", tmp_dir)

    return {
        "status": "ok",
        "output_dir": str(output_dir),
        "best_model_dir": str(model_args.best_model_dir),
        "classes_npy": str(classes_path),
        "num_train_samples": int(train_df.shape[0]),
        "num_eval_samples": int(eval_df.shape[0]),
        "num_labels": int(num_labels),
    }


def write_status(message: str, tmp_dir: Path):
    status_file = tmp_dir / "status.txt"
    with open(status_file, "w") as f:
        f.write(message)


def freeze_base_and_unfreeze_classifier(skt_model):
    for _, p in skt_model.model.named_parameters():
        p.requires_grad = False
    unfreeze_patterns = ["classifier", "pooler", "out_proj", "score", "classifier_head", "head"]
    for name, p in skt_model.model.named_parameters():
        for pat in unfreeze_patterns:
            if pat in name:
                p.requires_grad = True
    if hasattr(skt_model.model, "classifier"):
        for p in skt_model.model.classifier.parameters():
            p.requires_grad = True
