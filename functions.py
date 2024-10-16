import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib as plt
import os
import csv
import glob
import plotly.express as px
from datetime import datetime
import random 
import string
from fuzzywuzzy import process, fuzz
import re
import hashlib
import pgeocode
import torch
from scipy.interpolate import CubicSpline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from filterpy.kalman import KalmanFilter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sktime.forecasting.compose import make_reduction
from sklearn.linear_model import LinearRegression


def alter_group(alter):
    if alter > 0 and alter <30:
        return "junge Kunden"
    elif 30 <= alter < 55:
        return "mittelalte Kunden"
    elif alter >= 55:
        return "ältere Kunden"
    else:
        return "Unbekannt"

def hohe_beitrag(df,monthly_columns, threshold_factor=5000):
    hohe_zahlungen = df[monthly_columns] > threshold_factor
    hohe_kunden = df[hohe_zahlungen.any(axis=1)]
    return hohe_kunden

def visualize_value_counts(df):
    for column in df.columns:
        value_counts = df[column].value_counts()

        fig = px.bar(value_counts, x=value_counts.index, y=value_counts.values, labels={"x":column, "y": "Value Counts"}, title=f"Value counts für {column}")
        fig.show()

#Funktion um Alter aus Geb-Datum zu ermitteln
def calculate_age(row):
    if pd.notna(row["geb_dat"]):
        return row["buch_jhr"] - row["geb_dat"].year
    else:
        return np.nan

# Funktion zum Erzeugen eines zufälligen Strings

#Dictionary mit allen Klarnamen einbauen, damit z.B der Robert immer xyz
#sh vorname_nachname_gebdatum = einzigartiger Hash
def generate_random_string(length=8):
    letters = string.ascii_letters
    return "".join(random.choice(letters) for i in range(length))

#Funktion zum Anonymisieren

def anonymize(column):
    return column.apply(lambda x: generate_random_string())


def prepare_postcodes(postcodes):
        postcodes = postcodes.fillna(0) 
        postcodes = postcodes.astype(int).astype(str)
        return postcodes.str.zfill(5)

def create_hash(value):
    #salt = os.urandom(16)
    #geb_dat = row.get('geb_dat', '01-01-1900')
    #unique_string =  f"{row['vorname']}_{row['nachname']}_{row['kdnr_vt']}"
    hash_object = hashlib.sha3_512(str(value).encode("utf-8")).hexdigest()
    return hash_object

def replace_string(text, replacements):
    for item in replacements:
        text = text.replace(item, "")
    return text

def find_similar_names(name,choices,threshold=80):
    results = process.extract(name, choices, scorer=fuzz.token_sort_ratio)
    return [result[0] for result in results if result[1] >= threshold]

def kleine_einzahlungen_delete(row):
    zero_count = (row[[f"wert{i}" for i in range(1,13)]] == 0).sum()
    non_zero_values = row[[f"wert{i}" for i in range(1,13)]][row[[f"wert{i}" for i in range(1,13)]] != 0]
    return zero_count >= 11 and len(non_zero_values) == 1 and non_zero_values.iloc[0] < 50

def fill_missing_info(row, latest_info, columns):
    for column in columns:
        if pd.isna(row[column]) or row[column] == "Unbekannt":
            if pd.notna(latest_info.loc[row["vsnr"], column]):
                row[column] = latest_info.loc[row["vsnr"], column]
    return row


def fill_missing_info(row, latest_info, columns):
    for column in columns:
        if pd.isna(row[column]):
            if pd.notna(latest_info.loc[row["vsnr"], column]):
                row[column] = latest_info.loc[row["vsnr"], column]
    return row

def get_bundesland_by_plz(plz, country="DE"):
    nomi = pgeocode.Nominatim(country)
    location = nomi.query_postal_code(plz)
    state = location.state_name
    return state

def glätten_buchungsverzögerung(df, spalten):
    for i in range(1,len(spalten)):
        condition = (df[spalten[i-1]] ==0) & (df[spalten[i]] == df[spalten[i-2]]*2)

        df.loc[condition, spalten[i-1]] = df.loc[condition, spalten[i-2]]

        df.loc[condition, spalten[i]] = df.loc[condition, spalten[i-2]]
        
    return df

def aufsummieren_der_verträge(df):
    df_counts = df.groupby(["hashed_values","buch_jhr","ktobew"]).size().reset_index(name="Anzahl")
    df_counts_mehrfach = df_counts[df_counts["Anzahl"]>1]
    if not df_counts_mehrfach.empty:
        df_filtered = df.merge(df_counts_mehrfach[["hashed_values","buch_jhr","ktobew"]], on=["hashed_values", "buch_jhr","ktobew"])
        beitrag_columns = [col for col in df.columns if col.startswith("wert")]
        df_summed = df_filtered.groupby(["hashed_values","buch_jhr","ktobew"], as_index=False)[beitrag_columns].sum()
        # df_cleaned = df[~df[["hashed_values","buch_jhr","ktobew"]].duplicated(keep=False)].reset_index(drop=True)
        # df_final = pd.concat([df_cleaned, df_summed], ignore_index=True)
        #df_final = df_cleaned._append(df_summed, ignore_index=False)
        df_merged = pd.merge(df, df_summed, on=["hashed_values","buch_jhr","ktobew"], how="left",suffixes=("","_summed"))
        df_final = pd.DataFrame()
        for col in beitrag_columns:
            df_merged[col] = df_merged[col+"_summed"].combine_first(df_merged[col])

        df_final = df_merged.drop(columns=[col+"_summed" for col in beitrag_columns])
        df_final = df_final.drop_duplicates(subset=["hashed_values","buch_jhr","ktobew"]).sort_values(by=["buch_jhr"]).reset_index(drop=True)
    else:
        print("Keine Treffer")
    
    return df_final
                    
def prepare_dataframe():

    direct_path = f"Daten/"
    file_pattern = direct_path + "*neubeitrag.xlsx" 
    file_list_einzel = glob.glob(file_pattern)
    columns_list_einzel = ["gfeld","insolvenz","branche","kunde","plz_kunde",	
"portal_nutz_art","vtrg_prd_schl","koll_nr","kollektiv","vsnr",
"tochter_sl","vtrg_stat","persl_nr","nachname","vorname","geb_dat","kdnr_vt",
"vbeg","buch_jhr","ktobew","ktr_nr",		
"wert1","wert2","wert3","wert4","wert5","wert6","wert7","wert8","wert9","wert10","wert11","wert12","wert13"	
,"sumjahr","anl_art_sl_kd","grvv_beginn","ak_satz","primärbank","grvv_beginn"]

    einzahlung_columns = [
        "wert1","wert2","wert3","wert4","wert5","wert6","wert7","wert8","wert9","wert10","wert11","wert12","wert13","sumjahr"
        ]
    
    monthly_columns = [
        "wert1","wert2","wert3","wert4","wert5","wert6","wert7","wert8","wert9","wert10","wert11","wert12"
        ]

    data_frames_einzel = []

    for file in file_list_einzel:
        df = pd.read_excel(file, engine="openpyxl", sheet_name="neubeitrag")
        data_frames_einzel.append(df)

    combined_df_einzel = pd.concat(data_frames_einzel, ignore_index=True)
    combined_df_einzel.to_csv("Daten/LAZ-Dataframe-Rohdaten.csv")
    # Auf die wichtigsten Spalten beschränken
    combined_df_einzel = combined_df_einzel[columns_list_einzel]

    # Nur auf LAZ-bedingte Daten filtern
    combined_df_einzel = combined_df_einzel.loc[combined_df_einzel["gfeld"] == "LAZ"]

    # Überschüsse in Fonds investieren /Überschussverwendung eher nicht darauf gehen da die Datenbasis zu niedrig ist
    combined_df_einzel = combined_df_einzel.loc[combined_df_einzel["anl_art_sl_kd"] != "LAZUI03"]
    combined_df_einzel = combined_df_einzel.loc[combined_df_einzel["anl_art_sl_kd"] != "LAZI01"]
    combined_df_einzel = combined_df_einzel.loc[combined_df_einzel["anl_art_sl_kd"] != "LAZI02"]
    combined_df_einzel = combined_df_einzel.loc[combined_df_einzel["anl_art_sl_kd"] != "LAZUI01"]
    combined_df_einzel = combined_df_einzel.loc[combined_df_einzel["anl_art_sl_kd"] != "LAZUI02"]
    combined_df_einzel = combined_df_einzel.loc[combined_df_einzel["anl_art_sl_kd"] != "LAZLVXX"]

    spalten_zum_füllen = ["anl_art_sl_kd", "primärbank", "plz_kunde", "geb_dat"]

    latest_info = combined_df_einzel[spalten_zum_füllen + ["vsnr"]].groupby("vsnr").first()

    combined_df_einzel = combined_df_einzel.apply(fill_missing_info, axis=1, latest_info=latest_info, columns=spalten_zum_füllen)

    # Alter aus der Geb_dat spalte berechnen, da geb_dat anonymisiert bzw. verwendet werden darf
    combined_df_einzel["geb_dat"] = pd.to_datetime(combined_df_einzel["geb_dat"])
    combined_df_einzel["alter"] = combined_df_einzel.apply(calculate_age, axis=1)
    combined_df_einzel["Altersgruppen"] = combined_df_einzel["alter"].apply(alter_group)


    #Nur zwischen R+V und Fremdportalen unterscheiden => zusammenfassen der Fremdportale
    combined_df_einzel["portal_nutz_art"] = combined_df_einzel["portal_nutz_art"].fillna("Unbekannt")
    combined_df_einzel["portal_nutz_art"] = combined_df_einzel["portal_nutz_art"].apply(lambda x: "Fremdportal" if "msg" in x else x)
    combined_df_einzel["portal_nutz_art"] = combined_df_einzel["portal_nutz_art"].apply(lambda x: "Fremdportal" if "HPBS" in x else x)
    combined_df_einzel["portal_nutz_art"] = combined_df_einzel["portal_nutz_art"].apply(lambda x: "Fremdportal" if "TWTH" in x else x)
    combined_df_einzel["portal_nutz_art"] = combined_df_einzel["portal_nutz_art"].apply(lambda x: "Fremdportal" if "R+V Nichtportalkunde" in x else x)


    #Zusammenfassen der Firmen die in eine Genossenschaft gehören
    combined_df_einzel["kunde_zsm"] = combined_df_einzel["kunde"] 
    combined_df_einzel["kunde_zsm"] = np.where(combined_df_einzel["kunde_zsm"].str.contains("R\+V", case=False, na=False),"R+V", combined_df_einzel["kunde"])
    combined_df_einzel["kunde_zsm"] = np.where(combined_df_einzel["kunde_zsm"].str.contains("Siemens", case=False, na=False),"Siemens", combined_df_einzel["kunde"])
    combined_df_einzel["kunde_zsm"] = np.where(combined_df_einzel["kunde_zsm"].str.contains("Heraeus", case=False, na=False),"Heraeus", combined_df_einzel["kunde"])
    combined_df_einzel["kunde_zsm"] = np.where(combined_df_einzel["kunde_zsm"].str.contains("Roche", case=False, na=False),"Roche", combined_df_einzel["kunde"])
    combined_df_einzel["kunde_zsm"] = np.where(combined_df_einzel["kunde_zsm"].str.contains("Atruvia", case=False, na=False),"Atruvia", combined_df_einzel["kunde"])
    combined_df_einzel["kunde_zsm"] = np.where(combined_df_einzel["kunde_zsm"].str.contains("KPMG", case=False, na=False),"KPMG", combined_df_einzel["kunde"])
    combined_df_einzel["kunde_zsm"] = np.where(combined_df_einzel["kunde_zsm"].str.contains("profine", case=False, na=False),"profine", combined_df_einzel["kunde"])

    
    irrelvant_words = ["AG", "GmbH", "eG", "GbR", "GmbH & Co.KG","GmbH & Co. KG","SE & Co. KGaA", "KG", "Solutions", "Services","Service","& Co.", "Co.","Deutschland", "SE"]
    print("Mitte vom Code beginnt")

    #Entfernen der Abkürzungen wie GmbH etc. um Genossenschaften zusammenfassen zu können
    combined_df_einzel["kunde_vorverarbeitet"] = combined_df_einzel["kunde_zsm"].astype(str).apply(replace_string, replacements=irrelvant_words)

    firmennamen = combined_df_einzel["kunde_vorverarbeitet"].unique()
    gruppen = {}

    # Zusammenfassen der ähnlichen Firmen 
    for name in firmennamen:
        if name not in gruppen:
            similar_names = find_similar_names(name, firmennamen)
            for similar_name in similar_names:
                gruppen[similar_name] = name

    combined_df_einzel["kunde_zsm"] = combined_df_einzel["kunde_vorverarbeitet"].map(gruppen)

    #Anonymisieren der Spalten die personenbezogene Informationen beinhalten
    combined_df_einzel["hashed_values"] = combined_df_einzel["nachname"].astype(str).str[:3] +combined_df_einzel["kdnr_vt"].astype(str) +  combined_df_einzel["geb_dat"].fillna("Nicht Bekannt").astype(str) + combined_df_einzel['plz_kunde'].astype(str).str[:3]+ combined_df_einzel['kunde'].astype(str).str[:3]
    #unique_ids = combined_df_einzel["kdnr_vt"].unique()
    #hashed_id_dict = {id_: create_hash(id_) for id_ in unique_ids}
    combined_df_einzel["hashed_values"] = combined_df_einzel["hashed_values"].apply(create_hash)
    #combined_df_einzel["hashed_values"] = combined_df_einzel["kdnr_vt"].apply(create_hash)

    #Bei den älteren Excels gibt es bei der Einzahlung noch das Format, dass ein - vor der eingezahlten Summe auftritt somit wieder in eine pos. Zahl umwandeln
    for spalte in einzahlung_columns:
        mask = (combined_df_einzel["ktobew"] == "Einzahlung") & (combined_df_einzel[spalte] <0)
        combined_df_einzel.loc[mask, spalte] = -combined_df_einzel.loc[mask, spalte]
        #combined_df_einzel.loc[combined_df_einzel["ktobew"] == "Einzahlung", spalte] = combined_df_einzel.loc[columns_list_einzel["ktobew"],spalte].apply(lambda x: -x if pd.notnull(x) and x< 0 else x)

    for spalte in einzahlung_columns:
        mask = (combined_df_einzel["ktobew"] == "Auszahlung") & (combined_df_einzel[spalte]  > 0)
        combined_df_einzel.loc[mask, spalte] = -combined_df_einzel.loc[mask, spalte]

    #Auffüllen der NaN-Werte
    combined_df_einzel["plz_kunde"] = prepare_postcodes(combined_df_einzel["plz_kunde"])
    combined_df_einzel["anl_art_sl_kd"] = combined_df_einzel["anl_art_sl_kd"].fillna("Unbekannt")
    combined_df_einzel["primärbank"] = combined_df_einzel["primärbank"].fillna("Unbekannt")
    combined_df_einzel["alter"] = combined_df_einzel["alter"].fillna(0)
    combined_df_einzel["ak_satz"] = combined_df_einzel["ak_satz"].fillna(0)
    combined_df_einzel["ak"] = combined_df_einzel["ak_satz"].fillna(0)

    unique_plz = combined_df_einzel["plz_kunde"].unique()
    print(unique_plz)
    bundesland_mapping = {plz: get_bundesland_by_plz(plz) for plz in unique_plz}

    combined_df_einzel["Bundesland"] = combined_df_einzel["plz_kunde"].map(bundesland_mapping)
    combined_df_einzel['plz_kunde'] = combined_df_einzel['plz_kunde'].astype(str).str[:3]

    combined_df_einzel = combined_df_einzel.dropna(subset=["koll_nr"])
    combined_df_einzel = combined_df_einzel.dropna(subset=["grvv_beginn"])
    combined_df_einzel = combined_df_einzel.drop(columns=["kunde_vorverarbeitet", "gfeld" ,"geb_dat", "vsnr","nachname", "vorname"])

    rows_to_remove = combined_df_einzel.apply(kleine_einzahlungen_delete, axis=1)
    print("Ende vom Code beginnt")
    combined_df_einzel = combined_df_einzel[~rows_to_remove]
    combined_df_einzel_ganz = combined_df_einzel.copy()
    combined_df_einzel = combined_df_einzel.loc[combined_df_einzel["vtrg_prd_schl"] != "LAZ12T"]
    combined_df_einzel = combined_df_einzel.loc[combined_df_einzel["vtrg_prd_schl"] != "LAZ05T"]
    #combined_df_einzel = combined_df_einzel.drop_duplicates(subset=["hashed_values","buch_jhr", "ktobew"], keep="last").reset_index(drop=True)
    #combined_df_einzel_ganz = combined_df_einzel_ganz.drop_duplicates(subset=["hashed_values","buch_jhr", "ktobew"], keep="last").reset_index(drop=True)

    combined_df_einzel = aufsummieren_der_verträge(combined_df_einzel)
    combined_df_einzel = glätten_buchungsverzögerung(combined_df_einzel, monthly_columns)


    return combined_df_einzel, combined_df_einzel_ganz
    


####################################################################################################################
####################################################################################################################
#####################################  Augmentationsverfahren für TS ###############################################
####################################################################################################################
####################################################################################################################

 
def reverse_1d(x):
    if x.ndim ==1:
        return np.flip(x)
    else:
        return np.flip(x,axis=1)

# Hinzufügen von Rauschen
def jitter_1d(x, sigma=0.02):
    if x.ndim == 1:
        x = np.expand_dims(x,axis=1)
    
    max_value = np.max(np.abs(x))
    jittered = x + np.random.normal(0,sigma * max_value, size=x.shape)
    return jittered.squeeze()

def scaling_1d(x, sigma=1.1):
    x = np.array(x)

    # 1D-Daten (eine Zeitreihe)
    if x.ndim == 1:
        factor = np.random.normal(loc=1., scale=sigma, size=x.shape[0])  # Skalierungsfaktor für jede Zeitstempel
        scaled_x = x * factor  # Anwenden der Skalierung

    # 2D-Daten (mehrere Zeitreihen oder Features)
    elif x.ndim == 2:
        factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[1]))  # Skalierungsfaktor für jede Zeile (Feature)
        scaled_x = np.multiply(x, factor)  # Anwenden der Skalierung auf die Zeitreihen

    return scaled_x

def permutation_1d(x, max_segments=5, seg_mode="equal"):
    x = np.array(x)

    # 1D-Daten (z.B. eine einzelne Zeitreihe)
    if x.ndim == 1:
        orig_steps = np.arange(x.shape[0])
        num_segs = np.random.randint(2, max_segments + 1)  # Mindestens 2 Segmente

        if num_segs > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[0] - 1, num_segs - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)  # Aufteilen in Segmente
            else:
                splits = np.array_split(orig_steps, num_segs)  # Gleichmäßig aufteilen

            # Permutieren der Segmentreihenfolge
            permuted_indices = np.random.permutation(len(splits))
            permuted_segments = [x[splits[i]].flatten() for i in permuted_indices]
            permuted_data = np.concatenate(permuted_segments, axis=0)  # Segmente zusammenführen
        else:
            permuted_data = x  # Falls nur 1 Segment, unverändert zurückgeben

        return permuted_data

    # 2D-Daten (z.B. mehrere Zeitreihen oder Features)
    elif x.ndim == 2:
        orig_steps = np.arange(x.shape[1])  # Spaltenanzahl für jede Dimension (Monate)
        ret = np.zeros_like(x)

        # Iteriere durch jede Zeile (jede Zeitreihe)
        for i, pat in enumerate(x):
            num_segs = np.random.randint(1, min(max_segments, x.shape[1]))  # Anzahl Segmente auf max. Spalten beschränken

            if num_segs > 1:
                if seg_mode == "random":
                    split_points = np.random.choice(x.shape[1] - 1, num_segs - 1, replace=False)
                    split_points.sort()
                    splits = np.split(orig_steps, split_points)
                else:
                    splits = np.array_split(orig_steps, num_segs)
                
                # Permutieren der Segmentreihenfolge
                permuted_indices = np.random.permutation(len(splits))
                permuted_segments = [pat[splits[j]] for j in permuted_indices]
                permuted_data = np.concatenate(permuted_segments)

                ret[i, :len(permuted_data)] = permuted_data  # Fülle die permutierten Daten in die Zeile
            else:
                ret[i] = pat  # Falls nur 1 Segment, unverändert zurückgeben

        return ret
    

def magnitude_warp_1d(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline

    # Falls die Eingabe 1D ist, wandle sie in 2D um
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)  # Umwandeln in 2D, um die Funktion zu nutzen
        is_1d = True  # Merken, dass die Eingabe 1D war
    else:
        is_1d = False  # 2D-Fall

    orig_steps = np.arange(x.shape[0])  # Schritte entlang der Zeitachse

    # Bereite die Zufallsverzerrungen vor, je nach Dimension der Daten
    if is_1d:
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2))
        warp_steps = np.linspace(0, x.shape[0] - 1, num=knot + 2)

        # Sicherstellen, dass warp_steps strikt ansteigend ist
        if np.any(np.diff(warp_steps) <= 0):
            warp_steps += np.arange(len(warp_steps)) * 1e-4  # Minimale Störung hinzufügen

        ret = np.zeros_like(x)
        
        # 1D-Verarbeitung
        warper = CubicSpline(warp_steps, random_warps)(orig_steps)
        ret[:, 0] = x[:, 0] * warper  # Amplitude verzerren

    else:
        # Für 2D- oder 3D-Daten
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[1]))
        warp_steps = np.linspace(0, x.shape[1] - 1, num=knot + 2)

        # Sicherstellen, dass warp_steps strikt ansteigend ist
        if np.any(np.diff(warp_steps) <= 0):
            warp_steps += np.arange(len(warp_steps)) * 1e-4  # Minimale Störung hinzufügen

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            warper = CubicSpline(warp_steps, random_warps[i, :, 0])(orig_steps)
            ret[i] = pat * warper  # Verzerrte Amplitude anwenden

    # Rückgabe als 1D-Array, wenn die Eingabe ursprünglich 1D war
    if is_1d:
        ret = np.squeeze(ret, axis=1)

    return ret

def magnitude_warp_1d(x, sigma=0.2, knot=4):
    # Falls die Eingabe 1D ist, wandle sie in 2D um
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)  # In 2D umwandeln
        is_1d = True  # Merken, dass die Eingabe 1D war
    else:
        is_1d = False  # 2D-Fall

    orig_steps = np.arange(x.shape[0])  # Schritte entlang der Zeitachse

    # Zufallsverzerrungen und Kontrollpunkte für die Spline-Interpolation
    if is_1d:
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2))
        warp_steps = np.linspace(0, x.shape[0] - 1, num=knot + 2)

        # Sicherstellen, dass warp_steps strikt ansteigend ist
        if np.any(np.diff(warp_steps) <= 0):
            warp_steps += np.arange(len(warp_steps)) * 1e-4  # Minimale Störung hinzufügen

        ret = np.zeros_like(x)

        # 1D-Verarbeitung
        warper = CubicSpline(warp_steps, random_warps)(orig_steps)
        ret[:, 0] = x[:, 0] * warper  # Amplitude verzerren

    else:
        # Für 2D-Daten
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, x.shape[1]))
        warp_steps = np.linspace(0, x.shape[0] - 1, num=knot + 2)

        ret = np.zeros_like(x)
        for i in range(x.shape[1]):
            warper = CubicSpline(warp_steps, random_warps[:, i])(orig_steps)
            ret[:, i] = x[:, i] * warper  # Amplitude verzerren

    # Rückgabe als 1D-Array, wenn die Eingabe ursprünglich 1D war
    if is_1d:
        ret = np.squeeze(ret, axis=1)

    return ret

def time_warp_1d(x, sigma=0.2, knot=4):
    # Falls die Eingabe 1D ist, wandle sie in 2D um
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)  # In 2D umwandeln
        is_1d = True  # Merken, dass die Eingabe 1D war
    else:
        is_1d = False  # 2D-Fall

    orig_steps = np.arange(x.shape[0])  # Schritte entlang der Zeitachse

    ret = np.zeros_like(x)

    for i in range(x.shape[1]):  # Schleife über jede Spalte (jede Zeitreihe)
        # Generiere stärkere Verzerrungen
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2))
        warp_steps = np.linspace(0, x.shape[0] - 1, num=knot + 2)

        # Sicherstellen, dass warp_steps strikt ansteigend ist
        if np.any(np.diff(warp_steps) <= 0):
            warp_steps += np.arange(len(warp_steps)) * 1e-4  # Minimale Störung hinzufügen

        warper = CubicSpline(warp_steps, random_warps)(orig_steps)

        # Skalierung der Zeitverzerrung auf die Länge der Originaldaten
        scale = (x.shape[0] - 1) / warper[-1]
        warped_indices = np.clip(scale * warper, 0, x.shape[0] - 1)

        # Interpolation der Daten basierend auf den verzerrten Indizes
        ret[:, i] = np.interp(orig_steps, warped_indices, x[:, i])

    # Rückgabe als 1D-Array, wenn die Eingabe ursprünglich 1D war
    if is_1d:
        ret = np.squeeze(ret, axis=1)

    return ret

def reverse_1d(x):
    if x.ndim ==1:
        return np.flip(x)
    else:
        return np.flip(x,axis=1)

def window_slice_1d(x, reduce_ratio=0.9):
    # Überprüfen, ob es sich um 1D- oder 2D-Daten handelt
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)  # In 2D umwandeln, falls 1D
        is_1d = True
    else:
        is_1d = False

    # Berechne die Länge des verkleinerten Fensters
    target_len = int(np.ceil(reduce_ratio * x.shape[0]))

    # Falls keine Reduktion möglich ist, gebe das Original zurück
    if target_len >= x.shape[0]:
        return x

    # Starte den Zufallsausschnitt
    start = np.random.randint(0, x.shape[0] - target_len)
    end = start + target_len

    # Neue Matrix für die verkleinerten und interpolierten Daten
    ret = np.zeros((x.shape[0], x.shape[1]))

    # Interpolation für jede Dimension durchführen
    for dim in range(x.shape[1]):
        # Interpolation von start:end auf die Länge der Originaldaten
        ret[:, dim] = np.interp(np.linspace(0, target_len - 1, num=x.shape[0]),
                                np.arange(target_len),
                                x[start:end, dim])
    # Falls die Daten ursprünglich 1D waren, konvertiere zurück in 1D
    if is_1d:
        ret = np.squeeze(ret, axis=1)

    return ret

 
def adjust_strength(level, low, medium, high):
    """Passt den Parameterwert basierend auf der Stärke an."""
    if level == 'leicht':
        return low
    elif level == 'mittel':
        return medium
    elif level == 'schwer':
        return high
    else:
        raise ValueError("Stärke muss 'leicht', 'mittel' oder 'schwer' sein")

# Reverse augmentation
def reverse(x):
    return np.flip(x, axis=1)

# Jitter augmentation
def jitter(x, stärke='mittel'):
    sigma = adjust_strength(stärke, 0.03, 0.09, 0.15)  # Stärke für Jitter
    max_value = np.max(np.abs(x), axis=1, keepdims=True)
    jittered = x + np.random.normal(0, sigma * max_value, size=x.shape)
    return jittered

# Scaling augmentation
def scaling(x, stärke='mittel'):
    sigma = adjust_strength(stärke, 0.05, 0.1, 0.2)  # Stärke für Scaling
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[1], 1))
    return x * factor

# Permutation augmentation
def permutation(x, stärke='mittel', seg_mode="equal"):
    max_segments = adjust_strength(stärke, 3, 5, 10)  # Stärke für Permutation
    ret = np.zeros_like(x)
    for batch in range(x.shape[0]):
        orig_steps = np.arange(x.shape[1])
        num_segs = np.random.randint(1, max_segments + 1)
        if num_segs > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 1, num_segs - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs)
            permuted_indices = np.random.permutation(len(splits))
            permuted_segments = [x[batch, splits[j], :] for j in permuted_indices]
            permuted_data = np.concatenate(permuted_segments, axis=0)
            ret[batch, :len(permuted_data), :] = permuted_data
        else:
            ret[batch] = x[batch]
    return ret

# Magnitude warp augmentation
def magnitude_warp(x, stärke='mittel'):
    sigma = adjust_strength(stärke, 0.1, 0.2, 0.3)  # Stärke für Magnitude Warp
    knot = adjust_strength(stärke, 2, 4, 6)  # Mehr Knotenpunkte bei höherer Stärke
    orig_steps = np.arange(x.shape[1])
    ret = np.zeros_like(x)
    for i in range(x.shape[0]):
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, x.shape[2]))
        warp_steps = np.linspace(0, x.shape[1] - 1, num=knot + 2)
        for dim in range(x.shape[2]):
            warper = CubicSpline(warp_steps, random_warps[:, dim])(orig_steps)
            ret[i, :, dim] = x[i, :, dim] * warper
    return ret

# Time warp augmentation
def time_warp(x, stärke='mittel'):
    sigma = adjust_strength(stärke, 0.1, 0.2, 0.3)
    knot = adjust_strength(stärke, 2, 4, 6)
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    return ret

# Window slice augmentation
def window_slice(x, stärke='mittel'):
    reduce_ratio = adjust_strength(stärke, 0.02, 0.04, 0.08)  # Stärke für Window Slice
    target_len = int(np.ceil(reduce_ratio * x.shape[1]))
    if target_len >= x.shape[1]:
        return x  # Keine Reduktion möglich, original zurückgeben
    start = np.random.randint(0, x.shape[1] - target_len)
    end = start + target_len
    ret = np.zeros_like(x)
    for batch in range(x.shape[0]):
        for dim in range(x.shape[2]):
            ret[batch, :, dim] = np.interp(
                np.linspace(0, target_len - 1, num=x.shape[1]),
                np.arange(target_len),
                x[batch, start:end, dim]
            )
    return ret

def apply_kalman(time_serie):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.], [0.]])
    kf.F = np.array([[1.,1.], [0.,1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 1000.
    kf.R = 5
    kf.Q = 0.1

    filtered_series = []
    for value in time_serie:
        kf.predict()
        kf.update([value])
        filtered_series.append(kf.x[0,0])

    return np.array(filtered_series)

def apply_kalman_to_3d(data_3d):
    filtered_data = np.zeros_like(data_3d)
    for i in range(data_3d.shape[0]):
        for j in range(data_3d.shape[2]):
            time_series = data_3d[i,:,j]
            filtered_series = apply_kalman(time_series)
            filtered_data[i,:,j] = filtered_series
    
    return filtered_data
####################################################################################################################
####################################################################################################################
#####################################  Augmentationsverfahren für FQ ###############################################
####################################################################################################################
####################################################################################################################

def reverse_frequency(x):
    return torch.flip(x, dims=[0])


# Entfernt Frequenzen basierend auf der Stärke
def remove_frequency(x, stärke='mittel'):
    pertub_ratio = adjust_strength(stärke, low=0.1, medium=0.2, high=0.4)  # Je nach Stärke wird das Verhältnis angepasst
    mask = torch.FloatTensor(x.shape).uniform_() > pertub_ratio
    mask = mask.to(x.device)
    return x * mask
# Fügt Rauschen basierend auf der Stärke hinzu
def add_frequency(x, stärke='mittel'):
    pertub_ratio = adjust_strength(stärke, low=0.3, medium=0.5, high=0.8)  # Die Stärke beeinflusst die Intensität des Rauschens
    mask = torch.FloatTensor(x.shape).uniform_() > (1 - pertub_ratio)
    mask = mask.to(x.device)
    max_amplitude = torch.abs(x).max()
    random_am = torch.rand(mask.shape) * (max_amplitude * adjust_strength(stärke, low=0.05, medium=0.1, high=0.2))  # Unterschiedliche Amplitudenstärken
    pertub_matrix = mask * random_am
    return x + pertub_matrix
# Skalierung der Frequenz basierend auf der Stärke
def scaling_frequency(x, stärke='mittel'):
    sigma = adjust_strength(stärke, low=0.05, medium=0.1, high=0.2)  # Je nach Stärke wird die Skalierung angepasst
    factor = torch.normal(mean=1., std=sigma, size=x.shape)
    return x * factor
# Mischung zwischen zwei Zeitreihen mit einstellbarer Stärke
def frequency_mixing(x, y, stärke='mittel'):
    mix_rate = adjust_strength(stärke, low=0.3, medium=0.5, high=0.7)  # Je nach Stärke wird der Mischanteil angepasst
    mask = torch.rand(x.shape) > mix_rate
    inverted_mask = 1 - mask
    return (x * mask) + (y * inverted_mask)
# Rotieren der Frequenzen mit einstellbarer Stärke
def rotation_frequency(freq_series, angle=None, stärke='mittel'):
    if angle is None:
        angle = torch.rand(1) * 2 * np.pi * adjust_strength(stärke, low=0.5, medium=1.0, high=2.0)  # Je nach Stärke wird der Winkel angepasst
    phase_shift = torch.exp(1j * angle)
    return freq_series * phase_shift


def inverse_fft(freq_ts):
    return torch.fft.ifft(freq_ts).real



class StandardScaler3D(BaseEstimator, TransformerMixin):
    
    def __init__(self, copy=True):
        self.scaler = StandardScaler(copy=copy)
    
    def fit(self, X, y=None):
        if len(X.shape) !=3:
            raise ValueError("Input array should be a 3D array")
        self.scaler.fit(X.reshape(-1, X.shape[-1]))
        return self
    
    def transform(self, X):
        if not hasattr(self.scaler, "scale_"):
            raise ValueError("The scaler is not fitted yet.")
        X_transformed = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        return X_transformed.reshape(X.shape)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        if not hasattr(self.scaler, "scale_"):
            raise ValueError("The scaler is not fitted yet")
        X_inversed = self.scaler.inverse_transform(X.reshape(-1, X.shape[-1]))
        return X_inversed.reshape(X.shape)
    

###############################################################################################################################################
###############################################################################################################################################
############################################## Funktionen für automatisches Modelltraining ####################################################
###############################################################################################################################################
###############################################################################################################################################

def remove_timestamp(path):
    return re.sub(r"_20\d{6}_\d{6}.*","",path)

def extract_augmentation_info(path, type):
    
    clean_path = remove_timestamp(path)
    print(clean_path)
    if type == "time":
        pattern = r"training/electricity__time_forecast_([\w-]+)_([\w-]+)"
    else:
        pattern = r"training/electricity__frequency_forecast_([\w-]+)_([\w-]+)"
    
    match = re.search(pattern,clean_path)

    if match:
        augmentation_method = match.group(1)
        strength = match.group(2)
        return augmentation_method, strength
    else:
        return None,None
    
def generate_simple_sine_wave(n_steps=1000, n_series=3000):
    """Generiert einfache multiple Sinuswellen mit Variationen."""
    all_series = []
    x = np.linspace(0, 2 * np.pi, n_steps)
    for i in range(n_series):
        wave_1 = np.sin(x)
        wave_2 = 0.5 * np.sin(2 * x + np.random.uniform(0, np.pi))
        wave_3 = 0.3 * np.sin(3 * x + np.random.uniform(0, np.pi))
        simple_wave = wave_1 + wave_2 + wave_3  # Einfache Sinuswellen
        all_series.append(simple_wave)
    return np.array(all_series), x
def generate_multiple_simple_sine_series(n_series, n_steps):
    """Generiert mehrere einfache Sinuswellen-Serien."""
    sine_data, x = generate_simple_sine_wave(n_steps, n_series)
    
    all_series_with_features = []
    
    for i in range(n_series):
        series_with_features = [
            sine_data[i],  
            sine_data[i],  
            sine_data[i],  
            sine_data[i],  
            sine_data[i],  
            sine_data[i],  
            sine_data[i]   
        ]
        
        all_series_with_features.append(np.stack(series_with_features, axis=-1))
    
    return np.array(all_series_with_features)

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
        
        # Berechne den eigentlichen Wert der Zeitreihe
        regressor.fit(y_train_repr, y_train)
        y_pred_sin = regressor.predict(y_pred)
        mse = mean_squared_error(y_test, y_pred_sin)
        mae = mean_absolute_error(y_test, y_pred_sin)
        
        # Speichere die Ergebnisse
        results.append({
            "model": path,
            "mse": mse,
            "mae": mae
        })
        
    # Gebe die Ergebnisse als DataFrame zurück
    return pd.DataFrame(results)

def evaluate_models_finetuned(model_paths, data_with_features, output_dims=320, type=None):
    results = []
    # Suche alle Modelle mit "time_forecast" im Namen
    
    
    # Daten für das Training und Testen vorbereiten
    train = data_with_features[0]  # Hier wird eine Serie verwendet
    train = np.expand_dims(train, axis=0)

    train_data = data_with_features
    
    # Schleife über alle gefundenen Modelle
    for path in model_paths:
        print(f"Evaluating model: {path}")
        
        augmentation_method, strength = extract_augmentation_info(path, type)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Setze das Gerät

        print("Augmentationsmethode:",augmentation_method)
        print("Stärke:",strength)
        # Lade das Modell
        model = TS2Vec(input_dims=7, device=device, output_dims=output_dims)
        state_dict = torch.load(path)
        model.net.load_state_dict(state_dict)

        # Finetunen auf die generischen Daten
        model = TS2Vec(
                input_dims=7,
                device=device,
                output_dims=320
            )
    
        if type == "time":
            # Training des Modells mit der aktuellen Augmentation
            loss_log = model.fit(
                train_data,
                verbose=True,
                augment_type=type,
                augment_strength = strength,
                augment_method_time=augmentation_method, 
                augment_method_freq=None,
                n_epochs=25
            )
        else:
            loss_log = model.fit(
                train_data,
                verbose=True,
                augment_type=type,
                augment_strength = strength,
                augment_method_time=None,  
                augment_method_freq=augmentation_method,
                n_epochs=25
            )
  
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
        
        # Berechne den eigentlichen Wert der Zeitreihe
        regressor.fit(y_train_repr, y_train)
        y_pred_sin = regressor.predict(y_pred)
        mse = mean_squared_error(y_test, y_pred_sin)
        mae = mean_absolute_error(y_test, y_pred_sin)
        
        # Speichere die Ergebnisse
        results.append({
            "model_tuned": path,
            "mse_tuned": mse,
            "mae_tuned": mae
        })
        
    # Gebe die Ergebnisse als DataFrame zurück
    return pd.DataFrame(results)



 
# Funktionen anpassen mit "_7dgen" für jede Funktion
def generate_clean_sine_wave_7dgen(n_steps=1000, n_series=3000):
    """Generiert multiple glatte Sinuswellen mit Variationen."""
    all_series = []
    x = np.linspace(0, 4 * np.pi, n_steps)
    for i in range(n_series):
        wave_1 = np.sin(x)
        wave_2 = 0.5 * np.sin(2 * x + np.random.uniform(0, np.pi))
        wave_3 = 0.3 * np.sin(3 * x + np.random.uniform(0, np.pi))
        clean_wave = wave_1 + wave_2 + wave_3  # Saubere, glatte Sinuswellen
        all_series.append(clean_wave)
    return np.array(all_series), x
def smooth_series_7dgen(series, window_size=10):
    """Glättet die Serie mit einem Moving Average."""
    return np.convolve(series, np.ones(window_size)/window_size, mode='same')
def add_jitter_7dgen(series, jitter_strength=0.05):
    """Fügt der Serie leichtes Jitter hinzu."""
    jitter = np.random.randn(len(series)) * jitter_strength
    return series + jitter
def generate_time_features_7dgen(n_steps):
    """Generiert kontinuierliche zeitbasierte Features."""
    t = np.linspace(0, 2 * np.pi, n_steps)
    sin_t = np.sin(t)  # Sinus der Zeit (zyklisches Feature)
    cos_t = np.cos(t)  # Cosinus der Zeit (zyklisches Feature)
    return np.stack([sin_t, cos_t], axis=1)
def generate_multiple_sine_with_features_7dgen(n_series, n_steps):
    """Generiert mehrere Sinuswellen mit zusätzlichen sinnvollen Features."""
    sine_data, x = generate_clean_sine_wave_7dgen(n_steps, n_series)
    
    # Generiere kontinuierliche zeitbasierte Features (z.B. Sinus und Cosinus der Zeit)
    time_features = generate_time_features_7dgen(n_steps)
    
    all_series_with_features = []
    
    for i in range(n_series):
        # Erzeuge die 7-Dimensionen: Originaldaten + Zeitfeatures (Sinus/Cosinus der Zeit)
        series_with_features = [
            sine_data[i],  # Dimension 0: Original saubere Sinuswelle
            sine_data[i] + np.random.randn(n_steps) * 0.01,  # Dimension 1: Leichtes Rauschen (Jitter)
            smooth_series_7dgen(sine_data[i]),  # Dimension 2: Glättung
            np.gradient(sine_data[i]),  # Dimension 3: Ableitung der Sinuswelle
            sine_data[i] * 0.02,  # Dimension 4: Skalierte Sinuswelle
            np.sin(x * 2 + np.random.uniform(0, np.pi)),  # Dimension 5: Sinuswelle mit Phasenverschiebung
            add_jitter_7dgen(sine_data[i], jitter_strength=0.01)  # Dimension 6: Leichtes Jitter
        ]
        
        all_series_with_features.append(np.stack(series_with_features, axis=-1))
    
    return np.array(all_series_with_features)