
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ts2vec import TS2Vec
import torch
import numpy as np
import pandas as pd
import datautils
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import sys 
import os
import warnings
warnings.filterwarnings("ignore")
from functions import generate_multiple_simple_sine_series,evaluate_models,generate_multiple_sine_with_features_7dgen,evaluate_models_finetuned,extract_augmentation_info
# Verwende die Funktion

def evaluate_models(model_paths, data_with_features, output_dims=320):
    results = []
    # Suche alle Modelle mit "time_forecast" im Namen
    
    
    # Daten für das Training und Testen vorbereiten
    train = data_with_features[0]  # Hier wird eine Serie verwendet
    train = np.expand_dims(train, axis=0)
    
    # Schleife über alle gefundenen Modelle
    for path in model_paths:
        print(f"Evaluating model: {path}")
        
        # Lade das Modell
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Setze das Gerät
        model = TS2Vec(input_dims=7, device=device, output_dims=output_dims)
        state_dict = torch.load(path)
        model.net.load_state_dict(state_dict)
        
        # Berechne die Repräsentation der Daten
        repr = model.encode(train, encoding_window="sliding")
        
        # Teile die Daten in Training und Test auf
        y_train = train[:, :900,:]
        y_train_repr = repr[:, :900, :]
        y_test_repr = repr[:, -100:, :]
        y_test = train[:, 900:,:]

        y_train =y_train.squeeze(0)
        y_train_repr = y_train_repr.squeeze(0)
        y_test_repr = y_test_repr.squeeze(0)
        y_test = y_test.squeeze(0)
        
        # Regressor für das Forecasting
        regressor = LinearRegression()
        forecaster = make_reduction(regressor, strategy="recursive", window_length=12)
        forecaster.fit(y_train_repr)
        # Vorhersagehorizont
        fh = np.arange(1, 101)
        y_pred = forecaster.predict(fh=fh)

        mse_repr = mean_squared_error(y_test_repr, y_pred)
        mae_repr = mean_absolute_error(y_test_repr,y_pred)

        print(f"MSE Repr: {mse_repr}")
        print(f"MAE Repr: {mae_repr}")
        
        # Berechne den eigentlichen Wert der Zeitreihe
        regressor.fit(y_train_repr, y_train)
        y_pred_sin = regressor.predict(y_pred)
        mse = mean_squared_error(y_test, y_pred_sin)
        mae = mean_absolute_error(y_test, y_pred_sin)
        
        # Speichere die Ergebnisse
        results.append({
            "model": path,
            "mse_repr": mse_repr,
            "mae_repr": mae_repr,
            "mse": mse,
            "mae": mae
        })
        
    # Gebe die Ergebnisse als DataFrame zurück
    return pd.DataFrame(results)


n_series = 1000  
n_steps = 1000  
data_with_simple_sines = generate_multiple_simple_sine_series(n_series, n_steps)
scaler = StandardScaler()
for i in range(data_with_simple_sines.shape[2]):
    data_with_simple_sines[:, :, i] = scaler.fit_transform(data_with_simple_sines[:, :, i])


model_directory = "training/"  
model_paths_time = [os.path.join(model_directory, f) for f in os.listdir(model_directory) if "electricity__time_forecast_" in f] 
model_paths_with_file_time = [os.path.join(path, "model.pkl") for path in model_paths_time]


model_paths_freq = [os.path.join(model_directory, f) for f in os.listdir(model_directory) if "electricity__frequency_forecast" in f] 
model_paths_with_file_freq = [os.path.join(path, "model.pkl") for path in model_paths_freq]

# Rufe die Funktion auf
results_df_time = evaluate_models(model_paths_with_file_time,data_with_simple_sines)
# Ergebnisse anzeigen
results_df_time["model"] = results_df_time["model"].str.replace("training/electricity__time_forecast_", "").str.rsplit("_20", n=1).str[0]
results_df_time.sort_values("mse")
results_df_time.to_csv("results/TimeGenerisch.csv")

# Rufe die Funktion auf
results_df_freq = evaluate_models(model_paths_with_file_freq,data_with_simple_sines)
results_df_freq["model"] = results_df_freq["model"].str.replace("training/electricity__frequency_forecast_", "").str.rsplit("_202", n=1).str[0]
results_df_freq.sort_values("mse")
results_df_freq.to_csv("results/FreqGenerisch.csv")


data_with_simple_sines_complex = generate_multiple_sine_with_features_7dgen(n_series, n_steps)
scaler = StandardScaler()
for i in range(data_with_simple_sines_complex.shape[2]):
    data_with_simple_sines_complex[:, :, i] = scaler.fit_transform(data_with_simple_sines_complex[:, :, i])


# Rufe die Funktion auf
results_df_time_pre_compl = evaluate_models(model_paths_with_file_time,data_with_simple_sines_complex)
# Ergebnisse anzeigen
results_df_time_pre_compl["model"] = results_df_time_pre_compl["model"].str.replace("training/electricity__time_forecast_", "").str.rsplit("_20", n=1).str[0]
results_df_time_pre_compl.sort_values("mse")

results_df_time_pre_compl.to_csv("results/TimeGenerischComplex.csv")

# Rufe die Funktion auf
results_df_freq_pre_compl = evaluate_models(model_paths_with_file_freq,data_with_simple_sines_complex)
results_df_freq_pre_compl["model"] = results_df_freq_pre_compl["model"].str.replace("training/electricity__frequency_forecast_", "").str.rsplit("_20", n=1).str[0]
results_df_freq_pre_compl.sort_values("mse")

results_df_freq_pre_compl.to_csv("FreqGenerischComplex.csv")
