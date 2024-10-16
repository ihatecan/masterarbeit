import numpy as np
import pandas as pd
from ts2vec import TS2Vec
import torch
import os
import datautils
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from datautils import load_forecast_csv
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
from tasks import eval_forecasting
import time
import datetime

data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(name="electricity", univar=True, )
train_data = data[:, train_slice]


device = "cuda" if torch.cuda.is_available() else "cpu"
augmentations = ["jittering", "scaling", "permutation", "magnitude_warp", "time_warp", "window_slice"]
augment_type = "time"
loader = "forecast_csv_univar"
strengths = ["leicht", "mittel", "schwer"]  # Stärken für Augmentation
repr_dims = 320

# Training für jede Augmentation und jede Stärke starten
for aug in augmentations:
    for strength in strengths:
        # Erstellen des Verzeichnisses für jede Augmentation und Stärke
        run_dir = 'training/' + "electricity" + '__' + name_with_datetime(f"time_forecast_{aug}_{strength}")  # Verzeichnis für das aktuelle Modell
        print(f"Versuche, das Verzeichnis zu erstellen: {run_dir}")  # Debugging-Ausgabe
        
        try:
            os.makedirs(run_dir, exist_ok=True)
            print(f"Verzeichnis '{run_dir}' erfolgreich erstellt.")
        except Exception as e:
            print(f"Fehler beim Erstellen des Verzeichnisses: {e}")
        
        print(f"\nTraining für die Augmentation '{aug}' mit Stärke '{strength}' startet")
        
        model = TS2Vec(
            input_dims=7,
            device=device,
            output_dims=repr_dims
        )
        
        loss_log = model.fit(
            train_data,
            verbose=True,
            augment_type=augment_type,
            augment_strength=strength, 
            augment_method_time=[aug],  
            augment_method_freq=None,
            n_epochs=200
        )

        model.save(f'{run_dir}/model.pkl')
        
        out, eval_res = eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
        
        # Ergebnisse speichern
        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        
        print('Evaluation result:', eval_res)

print("Alle Trainings abgeschlossen.")



import os

augmentations_without_strength = ["kalman", "reverse"]  # Augmentationen ohne Stärke
augment_type = "time"
loader = "forecast_csv_univar"
repr_dims = 320

# Training für Augmentationen ohne Stärke (kalman und reverse)
for aug in augmentations_without_strength:
    # Erstellen des Verzeichnisses für die aktuelle Augmentation
    run_dir = 'training/' + "electricity" + '__' + name_with_datetime(f"time_forecast_{aug}")  
    print(f"Versuche, das Verzeichnis zu erstellen: {run_dir}")  # Debugging-Ausgabe
    
    try:
        os.makedirs(run_dir, exist_ok=True)
        print(f"Verzeichnis '{run_dir}' erfolgreich erstellt.")
    except Exception as e:
        print(f"Fehler beim Erstellen des Verzeichnisses: {e}")
    
    print(f"\nTraining für die Augmentation '{aug}' startet")
    
    model = TS2Vec(
        input_dims=7,
        device=device,
        output_dims=repr_dims
    )
    
    loss_log = model.fit(
        train_data,
        verbose=True,
        augment_type=augment_type,
        augment_strength=None,  
        augment_method_time=[aug], 
        augment_method_freq=None,
        n_epochs=200
    )

    model.save(f'{run_dir}/model.pkl')
    
    out, eval_res = eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
    
    pkl_save(f'{run_dir}/out.pkl', out)
    pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
    
    print('Evaluation result:', eval_res)

print("Training für 'kalman' und 'reverse' abgeschlossen.")
