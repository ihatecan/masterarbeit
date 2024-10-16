import numpy as np
import time
from . import _eval_protocols as eval_protocols

def generate_pred_samples(features, data, pred_len, drop=0):
    # Extrahiere die Anzahl an Zeipunkten/Ereignisse
    n = data.shape[1]
    # die Features werden um die Predictionslänge gekürzt
    features = features[:, :-pred_len]
    #Labels werden entlang einer neuen Achse gestapelt, um die Samples für die Predictions zu erzeugen
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]

    # Features werden nach festgelegten drop wert gekürzt, allerdings ist hier drop standardmäßig 0 somit keine kürzung
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])

def cal_metrics(pred, target):
    # MSE und MAE zwischen Prediction und Wahrheit wird berechnet
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }
    
def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols):
    padding = 200
    
    t = time.time()
    #Daten werden encodiert
    all_repr = model.encode(
        data,
        causal=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t
    
    #Unterteilung der Repräsentationen in Trainings, Validierungs und Testabschnitte
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    
    #Unterteilung der Repräsentationen in Trainings, Validierungs und Testpackete
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
    
    #Abspeichern der Ergebnisse und Zeitprotokolle
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        # Es werden Samples aus den verschiedenen Paketen generiert
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
        
        #Anschließend wird ein Ridge-Modell mit den train und val daten trainiert
        t = time.time()
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t
        
        t = time.time()
        #Erzeugen von Predictions
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        # Vorhersagen znd Kabels werden zurück in die ursprüngliche Form gebracht
        #ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(-1, test_pred.shape[-1])
        test_labels = test_labels.reshape(-1, test_labels.shape[-1])
        
        #print(ori_shape)
        # print(train_data.shape)
        # print(valid_data.shape)
        # print(test_data.shape)
        # print(test_pred.shape)
        #Inverse Transformation wird durchgeführt
        if test_data.shape[0] > 1:
            # test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
            # test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
            test_pred_inv = scaler.inverse_transform(test_pred).reshape(test_data.shape)
            test_labels_inv = scaler.inverse_transform(test_pred).reshape(test_data.shape)
        else:
            test_pred_inv = scaler.inverse_transform(test_pred)
            test_labels_inv = scaler.inverse_transform(test_labels)

            
        # test_pred_inv = test_pred_inv.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2])
        # test_labels_inv = test_labels_inv.reshape(test_labels_inv.shape[0], test_labels_inv.shape[1], test_labels_inv.shape[2])
        # Vorhersage Metriken etc werden anschließend berechnet und am Ende abgespeichert
        out_log[pred_len] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }
    

    # print(test_pred_inv.shape)
    # print(test_labels_inv.shape)
        
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res
