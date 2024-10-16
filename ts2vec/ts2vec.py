import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSEncoder
from models.losses import hierarchical_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import math
import sys 
import os
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from functions import jitter, scaling, permutation, magnitude_warp, time_warp ,window_slice, reverse,remove_frequency, add_frequency, scaling_frequency, reverse_frequency,inverse_fft, apply_kalman_to_3d


class TS2Vec:
    '''The TS2Vec model'''
   
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        ''' Initialize a TS2Vec model.
       
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
       
        super().__init__()
        self.device = device
        self.lr = lr # Lernrate
        self.batch_size = batch_size #Batchsize für das Training
        self.max_train_length = max_train_length # maximal Länge der Sequenzen für das Training
        self.temporal_unit = temporal_unit # Minimale Größe für den temporalen Kontrast
       
        #Initialisieren des Encoders mit den festgelegten Parametern
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
  
        #AverageModel hilft dabei, um die Mittelwerte der Gewichte zu ermitteln -> kann Stabilität des Modells verbessern
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
       
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
       
        self.n_epochs = 0
        self.n_iters = 0
   
    def fit(self, train_data, n_epochs=None, n_iters=None, augment_type=None, augment_strength=None,augment_method_time=None,augment_method_freq=None,verbose=False):
        ''' Training the TS2Vec model.
       
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
           
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
  
        #Trainingsdaten in die vorgegebene Dimension bringen
        assert train_data.ndim == 3
       
        #Standartwerte festlegen
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
       
        # Berechne, in wie viele Sektionen der Trainingsdatensatz aufgeteilt werden kann, wenn die Länge der Zeitreihe länger als die maximale Trainingslänge ist
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)
  
        # Die Länge der Zeitreihen werden zentriert wenn die am Ende oder am Anfang der Daten NaNs erhalten sind
        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
       
        #Entfernen von NaNs
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        
        ####################################################################################################################################################################
        ###################################### Meine Implementierung der verschiedenen Augmentationsverfahren !!!!!! #######################################################
        ####################################################################################################################################################################
        ####################################################################################################################################################################

        original_data_time = train_data.copy()
        original_data_freq =  torch.fft.fft(torch.tensor(train_data, dtype=torch.float32))
        freq_ts = original_data_freq.clone()
        
        if augment_type == "time":
            original_data_time = train_data.copy()
            if augment_method_time == "jittering": # Fügt einfach Rauschen dazu 
                augmented_data_time = jitter(original_data_time, stärke=augment_strength)
            elif augment_method_time == "scaling": # Skaliert die Amplitude der Zeitreihe
                augmented_data_time = scaling(original_data_time,stärke=augment_strength)
            elif augment_method_time == "permutation": # Mischt die Segmente der Zeitreihe neu
                augmented_data_time = permutation(original_data_time,seg_mode="random",stärke=augment_strength)
            elif augment_method_time == "magnitude_warp": # Verzerrt die Amplitude an zufälligen Punkten
                augmented_data_time =  magnitude_warp(original_data_time,stärke=augment_strength)
            elif augment_method_time == "time_warp": # Verzerrt die Zeitachse der Zeitreihe
                augmented_data_time = time_warp(original_data_time,stärke=augment_strength)
            elif augment_method_time == "window_slice": # Schneidet ein Fenster aus der Zeitreihe und interpoliert es zurück
                augmented_data_time = window_slice(original_data_time,stärke=augment_strength)
            elif augment_method_time == "reverse": # Zeitreihe wird quasi Rückwärts ausgegeben
                augmented_data_time = reverse(original_data_time)
            elif augment_method_time == "kalman": # wendet den Kalman Filter auf die Zeitreihe an
                augmented_data_time = apply_kalman_to_3d(original_data_time)
            else:
                augmented_data_time = original_data_time

            # print(original_data_time)
            # print("----------------")
            # print(augmented_data_time)
            emb_train_time = torch.from_numpy(np.stack(original_data_time)).to(torch.float32)
            emb_train_time_aug = torch.from_numpy(np.stack(augmented_data_time)).to(torch.float32)

            # print(emb_train_time.shape)
            # print(emb_train_time_aug.shape)

            time_dataset = TensorDataset(emb_train_time)
            time_aug_dataset = TensorDataset(emb_train_time_aug)
            time_loader = DataLoader(time_dataset, batch_size=min(self.batch_size, len(time_dataset)), shuffle=True, drop_last=True)
            time_aug_loader = DataLoader(time_aug_dataset, batch_size=min(self.batch_size, len(time_aug_dataset)), shuffle=True, drop_last=True)
            # Initialisieren des Optimizers
            optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
            # Liste in denen die Verluste/Erros pro Epoche abgesichert werden
            loss_log = []
       
            while True:
            # Abrrechen wenn maximal anzahl an Epochen erlangt ist
                if n_epochs is not None and self.n_epochs >= n_epochs:
                    break
            
                cum_loss = 0
                n_epoch_iters = 0
            
                interrupted = False
                for (time_batch, time_aug_batch) in zip(time_loader,time_aug_loader):

                    # Abbrechen wenn maximale Anzahl an Iterationen erlangt ist
                    if n_iters is not None and self.n_iters >= n_iters:
                        interrupted = True
                        break 

                    time_x  = time_batch[0].to(self.device)
                    time_aug_x  = time_aug_batch[0].to(self.device)     

                    if time_aug_x.ndim == 2:
                        time_aug_x = time_aug_x.unsqueeze(-1)
                    
                    # print(time_x.shape)
                    # print(time_aug_x.shape)
                    # Kürzen der Sequenzen
                    if self.max_train_length is not None and time_x.size(1) > self.max_train_length:
                            window_offset_time = np.random.randint(time_x.size(1) - self.max_train_length + 1)
                            time_x = time_x[:, window_offset_time : window_offset_time + self.max_train_length]
                    if self.max_train_length is not None and time_aug_x.size(1) > self.max_train_length:
                            window_offset_time_aug = np.random.randint(time_aug_x.size(1) - self.max_train_length + 1)
                            time_aug_x = time_aug_x[:, window_offset_time_aug : window_offset_time_aug + self.max_train_length]        

                    #------------------------------------------------------------------------------------------------------------------------------------------------------
                    #PRE-TRAIN TASK WIRD HIER REALISIERT
                    # Zufällige Abschnitte der Zeitreihe werden erzeugt, um das Modell auf verschiedene Teile der Daten zu trainieren und die Generalisierung zu verbessern
                    #Anhand der zufälligen Crops wird das Kontrastive Lernen umgesetzt indem es lernt welche Abschnitte zusammen gehören müssen und welche weit von einander liegen
                    
                    # Zeitdaten: Original und augmentierte

                    # Länge der Zeitreihe festlegen
                    ts_l = time_x.size(1)
                    crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
                    crop_left = np.random.randint(ts_l - crop_l + 1)
                    crop_right = crop_left + crop_l
                    crop_eleft = np.random.randint(crop_left + 1)
                    crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                    crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=time_x.size(0))

                    # Crops für Original-Zeitdaten
                    out1_time = self._net(take_per_row(time_x, crop_offset + crop_eleft, crop_right - crop_eleft))
                    out1_time = out1_time[:, -crop_l:]
                    out2_time = self._net(take_per_row(time_x, crop_offset + crop_left, crop_eright - crop_left))
                    out2_time = out2_time[:, :crop_l]

                    # Crops für augmentierte Zeitdaten
                    out1_time_aug = self._net(take_per_row(time_aug_x, crop_offset + crop_eleft, crop_right - crop_eleft))
                    out1_time_aug = out1_time_aug[:, -crop_l:]
                    out2_time_aug = self._net(take_per_row(time_aug_x, crop_offset + crop_left, crop_eright - crop_left))
                    out2_time_aug = out2_time_aug[:, :crop_l]
 
                    #Auf Basis der Ausgaben wird dann der Verlust berechnet
                    # KLEINER LOSS: out1 und out2 haben ähnliche Repräsentationen, was darauf hindeutet, dass sie aus derselben Zeitreihe stammen (positives Paar).
                    # GROßER LOSS: out1 und out2 haben unähnliche Repräsentationen, was darauf hindeutet, dass sie aus verschiedenen Zeitreihen stammen (negatives Paar).
                    loss_time_aug_1 = hierarchical_contrastive_loss(out1_time, out2_time_aug, temporal_unit=self.temporal_unit)
                    loss_time_aug_2 = hierarchical_contrastive_loss(out1_time_aug, out2_time, temporal_unit=self.temporal_unit)

                    loss_time = loss_time_aug_1 + loss_time_aug_2
                    
                    loss = loss_time 
                    loss.backward() #Backpropagation für die Gradientenberechnung
                    torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.)
                    optimizer.step() # Modellparameter werden aktualisiert
                    self.net.update_parameters(self._net)
                    
                    cum_loss += loss.item()
                    n_epoch_iters += 1
                    self.n_iters += 1
                
                    if self.after_iter_callback is not None:
                        self.after_iter_callback(self, loss.item())
            
                if interrupted:
                    break
           
                cum_loss /= n_epoch_iters # Durchschnittlicher Verlust der jeweiligen Epoche wird berechnet
                loss_log.append(cum_loss)
                if verbose:
                    print(f"Epoch #{self.n_epochs}: loss={cum_loss}")

                self.n_epochs += 1
            
                if self.after_epoch_callback is not None:
                    self.after_epoch_callback(self, cum_loss)
           
            return loss_log

        elif augment_type == "frequency":
            #Rauschen hinzufügen (es wird ein Wert drauf addiert)
            if augment_method_freq == "add_frequency":
                augmented_data_freq = add_frequency(freq_ts,stärke=augment_strength)
            #Die Frequence wird mit einem Faktor multipliziert
            elif augment_method_freq == "scaling_frequency":
                augmented_data_freq = scaling_frequency(freq_ts,stärke=augment_strength)
            #Die Frequence wird rückwärts ausgegeben 
            elif augment_method_freq == "reverse_frequency":
                augmented_data_freq = reverse_frequency(freq_ts)
            #Es werden zufällige Frequenzpunkte auf 0 gesetzt
            elif augment_method_freq == "remove_frequency":
                augmented_data_freq = remove_frequency(freq_ts,stärke=augment_strength)
            else:
                augmented_data_freq = freq_ts
            
            augmented_data_freq = inverse_fft(augmented_data_freq)

            emb_train_freq = original_data_freq.to(torch.float32)
            emb_train_freq_aug = augmented_data_freq.to(torch.float32)
            freq_dataset = TensorDataset(emb_train_freq)
            freq_aug_dataset = TensorDataset(emb_train_freq_aug)
            freq_loader = DataLoader(freq_dataset, batch_size=min(self.batch_size, len(freq_dataset)), shuffle=True, drop_last=True)
            freq_aug_loader = DataLoader(freq_aug_dataset,  batch_size=min(self.batch_size, len(freq_aug_dataset)), shuffle=True, drop_last=True)

            # Initialisieren des Optimizers
            optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
            # Liste in denen die Verluste/Erros pro Epoche abgesichert werden
            loss_log = []
       
            while True:
            # Abrrechen wenn maximal anzahl an Epochen erlangt ist
                if n_epochs is not None and self.n_epochs >= n_epochs:
                    break
            
                cum_loss = 0
                n_epoch_iters = 0
            
                interrupted = False
                for (freq_batch, freq_aug_batch) in zip(freq_loader,freq_aug_loader):

                    # Abbrechen wenn maximale Anzahl an Iterationen erlangt ist
                    if n_iters is not None and self.n_iters >= n_iters:
                        interrupted = True
                        break 
                    freq_x  = freq_batch[0].to(self.device)
                    freq_aug_x  = freq_aug_batch[0].to(self.device)
                
                    if self.max_train_length is not None and freq_x.size(1) > self.max_train_length:
                            window_offset_freq = np.random.randint(freq_x.size(1) - self.max_train_length + 1)
                            freq_x = freq_x[:, window_offset_freq : window_offset_freq + self.max_train_length]
                    if self.max_train_length is not None and freq_aug_x.size(1) > self.max_train_length:
                            window_offset_freq_aug = np.random.randint(freq_aug_x.size(1) - self.max_train_length + 1)
                            freq_aug_x = freq_aug_x[:, window_offset_freq_aug : window_offset_freq_aug + self.max_train_length]

                    #------------------------------------------------------------------------------------------------------------------------------------------------------
                    #PRE-TRAIN TASK WIRD HIER REALISIERT
                    # Zufällige Abschnitte der Zeitreihe werden erzeugt, um das Modell auf verschiedene Teile der Daten zu trainieren und die Generalisierung zu verbessern
                    #Anhand der zufälligen Crops wird das Kontrastive Lernen umgesetzt indem es lernt welche Abschnitte zusammen gehören müssen und welche weit von einander liegen
                    
                    # Frequenzdaten: Original und augmentierte
                    # Cropping für Original-Frequenzdaten
                    freq_l = freq_x.size(1)
                    crop_l_freq = np.random.randint(low=2 ** (self.temporal_unit + 1), high=freq_l + 1)
                    crop_left_freq = np.random.randint(freq_l - crop_l_freq + 1)
                    crop_right_freq = crop_left_freq + crop_l_freq
                    crop_eleft_freq = np.random.randint(crop_left_freq + 1)
                    crop_eright_freq = np.random.randint(low=crop_right_freq, high=freq_l + 1)
                    crop_offset_freq = np.random.randint(low=-crop_eleft_freq, high=freq_l - crop_eright_freq + 1, size=freq_x.size(0))

            
                    # Crops für Original-Frequenzdaten
                    out1_freq = self._net(take_per_row(freq_x, crop_offset_freq + crop_eleft_freq, crop_right_freq - crop_eleft_freq))
                    out1_freq = out1_freq[:, -crop_l_freq:]
                    out2_freq = self._net(take_per_row(freq_x, crop_offset_freq + crop_left_freq, crop_eright_freq - crop_left_freq))
                    out2_freq = out2_freq[:, :crop_l_freq]

                    # Crops für augmentierte Frequenzdaten
                    out1_freq_aug = self._net(take_per_row(freq_aug_x, crop_offset_freq + crop_eleft_freq, crop_right_freq - crop_eleft_freq))
                    out1_freq_aug = out1_freq_aug[:, -crop_l_freq:]
                    out2_freq_aug = self._net(take_per_row(freq_aug_x, crop_offset_freq + crop_left_freq, crop_eright_freq - crop_left_freq))
                    out2_freq_aug = out2_freq_aug[:, :crop_l_freq]
                                                            
                    loss_freq_aug_1 = hierarchical_contrastive_loss(out1_freq, out2_freq_aug, temporal_unit=self.temporal_unit)
                    loss_freq_aug_2 = hierarchical_contrastive_loss(out1_freq_aug, out2_freq, temporal_unit=self.temporal_unit)

                    loss_freq = loss_freq_aug_1 + loss_freq_aug_2
                    loss = loss_freq

                    loss.backward() #Backpropagation für die Gradientenberechnung
                    torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.)
                    optimizer.step() # Modellparameter werden aktualisiert
                    self.net.update_parameters(self._net)
                    
                    cum_loss += loss.item()
                    n_epoch_iters += 1
                    self.n_iters += 1
                
                    if self.after_iter_callback is not None:
                        self.after_iter_callback(self, loss.item())
            
                if interrupted:
                    break
           
                cum_loss /= n_epoch_iters # Durchschnittlicher Verlust der jeweiligen Epoche wird berechnet
                loss_log.append(cum_loss)
                if verbose:
                    print(f"Epoch #{self.n_epochs}: loss={cum_loss}")

                self.n_epochs += 1
            
                if self.after_epoch_callback is not None:
                    self.after_epoch_callback(self, cum_loss)
           
            return loss_log
        
        elif augment_type == "combined":
            if augment_method_time == "jittering": # Fügt einfach Rauschen dazu 
                augmented_data_time = jitter(original_data_time,stärke=augment_strength)
            elif augment_method_time == "scaling": # Skaliert die Amplitude der Zeitreihe
                augmented_data_time = scaling(original_data_time,stärke=augment_strength)
            elif augment_method_time == "permutation": # Mischt die Segmente der Zeitreihe neu
                augmented_data_time = permutation(original_data_time, seg_mode="random",stärke=augment_strength)
            elif augment_method_time == "magnitude_warp": # Verzerrt die Amplitude an zufälligen Punkten
                augmented_data_time =  magnitude_warp(original_data_time, stärke=augment_strength)
            elif augment_method_time == "time_warp": # Verzerrt die Zeitachse der Zeitreihe
                augmented_data_time = time_warp(original_data_time, stärke=augment_strength)
            elif augment_method_time == "window_slice": # Schneidet ein Fenster aus der Zeitreihe und interpoliert es zurück
                augmented_data_time = window_slice(original_data_time,stärke=augment_strength)
            elif augment_method_time == "reverse": # Zeitreihe wird quasi Rückwärts ausgegeben
                augmented_data_time = reverse(original_data_time)
            elif augment_method_time == "kalman": # wendet den Kalman Filter auf die Zeitreihe an
                augmented_data_time = apply_kalman_to_3d(original_data_time)
            else:
                augmented_data_time = original_data_time

            if augment_method_freq == "add_frequency":
                augmented_data_freq = add_frequency(freq_ts,stärke=augment_strength)
            #Die Frequence wird mit einem Faktor multipliziert
            elif augment_method_freq == "scaling_frequency":
                augmented_data_freq = scaling_frequency(freq_ts,stärke=augment_strength)
            #Die Frequence wird rückwärts ausgegeben 
            elif augment_method_freq == "reverse_frequency":
                augmented_data_freq = reverse_frequency(freq_ts)
            #Es werden zufällige Frequenzpunkte auf 0 gesetzt
            elif augment_method_freq == "remove_frequency":
                augmented_data_freq = remove_frequency(freq_ts,stärke=augment_strength)
            else:
                augmented_data_freq = freq_ts
            
            augmented_data_freq = inverse_fft(augmented_data_freq)

            emb_train_time = torch.from_numpy(np.stack(original_data_time)).to(torch.float32)
            emb_train_time_aug = torch.from_numpy(np.stack(augmented_data_time)).to(torch.float32)
            emb_train_freq = original_data_freq.to(torch.float32)
            emb_train_freq_aug = augmented_data_freq.to(torch.float32)
        
            time_dataset = TensorDataset(emb_train_time)
            time_aug_dataset = TensorDataset(emb_train_time_aug)
            freq_dataset = TensorDataset(emb_train_freq)
            freq_aug_dataset = TensorDataset(emb_train_freq_aug)

            time_loader = DataLoader(time_dataset, batch_size=min(self.batch_size, len(time_dataset)), shuffle=True, drop_last=True)
            time_aug_loader = DataLoader(time_aug_dataset, batch_size=min(self.batch_size, len(time_aug_dataset)), shuffle=True, drop_last=True)
            freq_loader = DataLoader(freq_dataset, batch_size=min(self.batch_size, len(freq_dataset)), shuffle=True, drop_last=True)
            freq_aug_loader = DataLoader(freq_aug_dataset,  batch_size=min(self.batch_size, len(freq_aug_dataset)), shuffle=True, drop_last=True)

            optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
       
            # Liste in denen die Verluste/Erros pro Epoche abgesichert werden
            loss_log = []
            while True:
            # Abrrechen wenn maximal anzahl an Epochen erlangt ist
                if n_epochs is not None and self.n_epochs >= n_epochs:
                    break
            
                cum_loss = 0
                n_epoch_iters = 0
            
                interrupted = False
                for (time_batch, time_aug_batch, freq_batch, freq_aug_batch) in zip(time_loader,time_aug_loader,freq_loader,freq_aug_loader):

                    # Abbrechen wenn maximale Anzahl an Iterationen erlangt ist
                    if n_iters is not None and self.n_iters >= n_iters:
                        interrupted = True
                        break 

                    time_x  = time_batch[0].to(self.device)
                    time_aug_x  = time_aug_batch[0].to(self.device)
                    freq_x  = freq_batch[0].to(self.device)
                    freq_aug_x  = freq_aug_batch[0].to(self.device)
                
                    if time_aug_x.ndim == 2:
                        time_aug_x = time_aug_x.unsqueeze(-1)

                    # Kürzen der Sequenzen
                    if self.max_train_length is not None and time_x.size(1) > self.max_train_length:
                            window_offset_time = np.random.randint(time_x.size(1) - self.max_train_length + 1)
                            time_x = time_x[:, window_offset_time : window_offset_time + self.max_train_length]
                    if self.max_train_length is not None and freq_x.size(1) > self.max_train_length:
                            window_offset_freq = np.random.randint(freq_x.size(1) - self.max_train_length + 1)
                            freq_x = freq_x[:, window_offset_freq : window_offset_freq + self.max_train_length]
                    if self.max_train_length is not None and time_aug_x.size(1) > self.max_train_length:
                            window_offset_time_aug = np.random.randint(time_aug_x.size(1) - self.max_train_length + 1)
                            time_aug_x = time_aug_x[:, window_offset_time_aug : window_offset_time_aug + self.max_train_length]
                    if self.max_train_length is not None and freq_aug_x.size(1) > self.max_train_length:
                            window_offset_freq_aug = np.random.randint(freq_aug_x.size(1) - self.max_train_length + 1)
                            freq_aug_x = freq_aug_x[:, window_offset_freq_aug : window_offset_freq_aug + self.max_train_length]

                    #------------------------------------------------------------------------------------------------------------------------------------------------------
                    #PRE-TRAIN TASK WIRD HIER REALISIERT
                    # Zufällige Abschnitte der Zeitreihe werden erzeugt, um das Modell auf verschiedene Teile der Daten zu trainieren und die Generalisierung zu verbessern
                    #Anhand der zufälligen Crops wird das Kontrastive Lernen umgesetzt indem es lernt welche Abschnitte zusammen gehören müssen und welche weit von einander liegen
                    
                    # Zeitdaten: Original und augmentierte

                    # Länge der Zeitreihe festlegen
                    ts_l = time_x.size(1)
                    crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
                    crop_left = np.random.randint(ts_l - crop_l + 1)
                    crop_right = crop_left + crop_l
                    crop_eleft = np.random.randint(crop_left + 1)
                    crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                    crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=time_x.size(0))

                    # Cropping für augmentierte Zeitdaten
                    #ts_l_aug = time_aug_x.size(1)
                    # Crops für Original-Zeitdaten
                    out1_time = self._net(take_per_row(time_x, crop_offset + crop_eleft, crop_right - crop_eleft))
                    out1_time = out1_time[:, -crop_l:]
                    out2_time = self._net(take_per_row(time_x, crop_offset + crop_left, crop_eright - crop_left))
                    out2_time = out2_time[:, :crop_l]

                    # Crops für augmentierte Zeitdaten
                    out1_time_aug = self._net(take_per_row(time_aug_x, crop_offset + crop_eleft, crop_right - crop_eleft))
                    out1_time_aug = out1_time_aug[:, -crop_l:]
                    out2_time_aug = self._net(take_per_row(time_aug_x, crop_offset + crop_left, crop_eright - crop_left))
                    out2_time_aug = out2_time_aug[:, :crop_l]

                    # print(out1_time.shape, out2_time.shape)
                    # print(out1_time_aug.shape, out2_time_aug.shape)

                    # Frequenzdaten: Original und augmentierte
                    # Cropping für Original-Frequenzdaten
                    freq_l = freq_x.size(1)
                    crop_l_freq = np.random.randint(low=2 ** (self.temporal_unit + 1), high=freq_l + 1)
                    crop_left_freq = np.random.randint(freq_l - crop_l_freq + 1)
                    crop_right_freq = crop_left_freq + crop_l_freq
                    crop_eleft_freq = np.random.randint(crop_left_freq + 1)
                    crop_eright_freq = np.random.randint(low=crop_right_freq, high=freq_l + 1)
                    crop_offset_freq = np.random.randint(low=-crop_eleft_freq, high=freq_l - crop_eright_freq + 1, size=freq_x.size(0))

        
                    # Crops für Original-Frequenzdaten
                    out1_freq = self._net(take_per_row(freq_x, crop_offset_freq + crop_eleft_freq, crop_right_freq - crop_eleft_freq))
                    out1_freq = out1_freq[:, -crop_l_freq:]
                    out2_freq = self._net(take_per_row(freq_x, crop_offset_freq + crop_left_freq, crop_eright_freq - crop_left_freq))
                    out2_freq = out2_freq[:, :crop_l_freq]

                    # Crops für augmentierte Frequenzdaten
                    out1_freq_aug = self._net(take_per_row(freq_aug_x, crop_offset_freq + crop_eleft_freq, crop_right_freq - crop_eleft_freq))
                    out1_freq_aug = out1_freq_aug[:, -crop_l_freq:]
                    out2_freq_aug = self._net(take_per_row(freq_aug_x, crop_offset_freq + crop_left_freq, crop_eright_freq - crop_left_freq))
                    out2_freq_aug = out2_freq_aug[:, :crop_l_freq]     

                    #Auf Basis der Ausgaben wird dann der Verlust berechnet
                    # KLEINER LOSS: out1 und out2 haben ähnliche Repräsentationen, was darauf hindeutet, dass sie aus derselben Zeitreihe stammen (positives Paar).
                    # GROßER LOSS: out1 und out2 haben unähnliche Repräsentationen, was darauf hindeutet, dass sie aus verschiedenen Zeitreihen stammen (negatives Paar).
                    loss_time_aug_1 = hierarchical_contrastive_loss(out1_time, out2_time_aug, temporal_unit=self.temporal_unit)
                    loss_time_aug_2 = hierarchical_contrastive_loss(out1_time_aug, out2_time, temporal_unit=self.temporal_unit)

                    loss_time = loss_time_aug_1 + loss_time_aug_2
                                                            
                    loss_freq_aug_1 = hierarchical_contrastive_loss(out1_freq, out2_freq_aug, temporal_unit=self.temporal_unit)
                    loss_freq_aug_2 = hierarchical_contrastive_loss(out1_freq_aug, out2_freq, temporal_unit=self.temporal_unit)

                    loss_freq = loss_freq_aug_1 + loss_freq_aug_2
                    loss = loss_time + loss_freq

                    loss.backward() #Backpropagation für die Gradientenberechnung
                    torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.)
                    optimizer.step() # Modellparameter werden aktualisiert
                    self.net.update_parameters(self._net)
                    
                    cum_loss += loss.item()
                    n_epoch_iters += 1
                    self.n_iters += 1
                
                    if self.after_iter_callback is not None:
                        self.after_iter_callback(self, loss.item())
            
                if interrupted:
                    break
            
                cum_loss /= n_epoch_iters # Durchschnittlicher Verlust der jeweiligen Epoche wird berechnet
                loss_log.append(cum_loss)
                if verbose:
                    print(f"Epoch #{self.n_epochs}: loss={cum_loss}")

                self.n_epochs += 1
            
                if self.after_epoch_callback is not None:
                    self.after_epoch_callback(self, cum_loss)
           
            return loss_log

        ####################################################################################################################################################################
        ###################################### Meine Implementierung der verschiedenen Augmentationsverfahren !!!!!! #######################################################
        ####################################################################################################################################################################
        ####################################################################################################################################################################

            #------------------------------------------------------------------------------------------------------------------------------------------------------
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        # Modell wird auf den Input X angewandt mit einer Maskierung
        out = self.net(x.to(self.device, non_blocking=True), mask)
  
        #Full series -> gesamte Zeitreihe wird beachtet
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            #Max-Pooling wird auf die gesamte Zeitreihe angwandt
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
        # Wenn encoding_window ein Integer ist, wird ein Max-Pooling mit dieser Fenstergröße durchgeführt    
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
  
        # Wenn encoding_window 'multiscale' ist, werden mehrere Pooling-Operationen mit verschiedenen Fenstergrößen durchgeführt  
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
           
        else:
            if slicing is not None:
                out = out[:, slicing]
           
        return out.cpu()
   
    def encode(self, data, mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations using the model.
       
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
           
        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape
  
        org_training = self.net.training
        self.net.eval()
       
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
       
        #Hier werden die Repräsentationen für die eingegebenen Daten berechnet, indem das Modell ohne Gradientenberechnung auf Batches von Daten angewendet wird
        #Je nach Paranetern werden die Daten in gleitenden Fenstern oder als Ganzes verarbeitet und die Repräsentationen werden am Ende zusammengeführt und zurückgegeben
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)
  
                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                   
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                       
                output.append(out)
               
            output = torch.cat(output, dim=0)
           
        self.net.train(org_training)
        return output.numpy()
   
    def save(self, fn):
        ''' Save the model to a file.
       
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
   
    def load(self, fn):
        ''' Load the model from a file.
       
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
   
