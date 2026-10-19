"""Desafio NLP: deteccion de incoherencias en evaluacion por pares"""

from collections import Counter
import re
import copy
import unicodedata
from pathlib import Path
from typing import Callable

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence # optimiza lstm para que no procese los 0

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------------------------
CSV_SOURCE = "https://www.dlsi.ua.es/~juanra/UA/dataset/dcadep/dcadep_melt_grades.csv.gz"
RESULTS_DIR = Path("results")

ACTIVITIES = [1, 2]
COURSES = [2015, 2016]
SECTIONS = ['1', '2', '3', '4', '5', '6', '7'] 
TARGET = "value"

SEED = 42
N_SPLITS = 10

EMPTY_TOKEN = "<empty>" 

# Stopwords que se conservan
KEEP_STOPWORDS = {
    "no",
    "muy",
    "todo",
    "mas",
    "pero",
    "poco",
    "hay",
    "tiene",
    "nada",
    "ni",
    "si",
    "mucho",
    "sin",
    "todos",
    "tanto",
    "algo"
}


# ----------------------------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------------------------
def make_strata(values: pd.Series) -> pd.Series:
    """Estratos para balancear pliegues: <1, [1,2), [2,3) y 3 exacto"""
    conditions = [values < 1, values < 2, values < 3]
    return pd.Series(np.select(conditions, [0, 1, 2], default=3), index=values.index) # asigna 0, 1, 2, 3 depende la condicion, por defecto 3.


def load_activity(activity: int, source: str = CSV_SOURCE) -> pd.DataFrame:
    """Filas de una actividad con estratos y flag de feedback vacio"""
    df = pd.read_csv(source, sep="\t", decimal=",")
    df["feedback"] = df["feedback"].fillna("")
    df["section"] = df["section"].astype(str)

    mask = (df["activity"] == activity) & df["section"].isin(SECTIONS) & df["year"].isin(COURSES)
    out = df.loc[mask, ["year", "evaluator", "work", "section", TARGET, "feedback"]].copy()

    # Aseguraciones de los datos
    assert out[TARGET].notna().all(), "hay NaN en value"
    assert out[TARGET].between(0, 3).all(), "value fuera de [0, 3]"

    out["is_empty"] = out["feedback"].str.strip() == ""
    out["stratum"] = make_strata(out[TARGET])
    return out.reset_index(drop=True)



# Preprocesado de texto

# Quita accentos
def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)  # separa cada letra de su marca de acento
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")  # Mn = marca: se descarta

# Limpia y deja solo letras y numeros
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", strip_accents(text.lower()))

# Lista de NLTK normalizada igual que el texto (sin acentos) y sin las palabras a conservar
def build_stopwords(keep: set[str] = KEEP_STOPWORDS) -> set[str]:
    nltk.download("stopwords", quiet=True)
    normalized = {strip_accents(w) for w in stopwords.words("spanish")}
    return normalized - {strip_accents(w) for w in keep}  # la lista a conservar se normaliza

# minusculas -> sin acentos -> tokens -> sin stopwords -> stemming, si no queda nada = <empty>
def preprocess_text(text: str, stop_set: set[str], stemmer: SnowballStemmer) -> list[str]:
    tokens = [t for t in tokenize(text) if t not in stop_set]
    stems = [stemmer.stem(t) for t in tokens]
    return stems if stems else [EMPTY_TOKEN]

# Aplica todo el pipeline a los textos y devuelve lista de tokens
def preprocess_corpus(texts: pd.Series, keep: set[str] = KEEP_STOPWORDS) -> pd.Series:
    stop_set = build_stopwords(keep)
    stemmer = SnowballStemmer("spanish")
    return texts.map(lambda text: preprocess_text(text, stop_set, stemmer))



# Validacion cruzada de 10 pliegues 
# Estratificada, misma semilla para todos los modelos

# Etiquetar cada fila de los datos con un número de grupo (0 al 9)
# Indica en que pliegue esa fila específica actuará como el conjunto de examen (test) 
# Para garantizar que todos los modelos que se entrenen mas adelante usen las mismas divisiones de datos
def assign_folds(data: pd.DataFrame, seed: int = SEED, n_splits: int = N_SPLITS) -> np.ndarray:
    fold_id = np.empty(len(data), dtype=int) # crea array nuevo sin inicializar valores
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) # mantiene la misma proporcion en cada pligue
    for k, (_, test_idx) in enumerate(skf.split(data, data["stratum"])):
        fold_id[test_idx] = k
    return fold_id

# Devuelve las posiciones de train y test del pliegue k
def fold_indices(fold_id: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.where(fold_id != k)[0]
    test_idx = np.where(fold_id == k)[0] # [0] para que devuelva directamente el array interno
    return train_idx, test_idx


# ----------------------------------------------------------------------------------------------
# Modelos PyTorch
# ----------------------------------------------------------------------------------------------

# Perceptron multicapa basico para representaciones Bag of Words
class MLPBoW(nn.Module):
    # vocab_size, palabras de la capa de entrada, hidden_size cuantas neuronas tendra la capa oculta
    def __init__(self, vocab_size: int, hidden_size: int = 64):
        super(MLPBoW, self).__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1) # Salida lineal, el recorte se hace post-prediccion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out.squeeze(-1) # Transforma (batch_size, 1) en (batch_size,)

# Red recurrente LSTM simple para secuencias de texto con padding
class SeqRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_size: int = 64):
        super(SeqRNN, self).__init__()

        # Se reserva padding_idx=0 para que su vector sea siempre de ceros y no aprenda nada de el
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # nn.Embedding transforma los IDs numéricos de las palabras en vectores densos de números continuos
        # Procesara las palabras una a una en orden, manteniendo una memoria de lo que ha leido antes
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True) # LSTM (Long Short-Term Memory)
        self.fc = nn.Linear(hidden_size, 1) # Coge el resumen final de la LSTM y lo transforma en un solo numero de salida

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Convierte los ID de palabras en vectores
        embedded = self.embedding(x)

        # Empaqueta la secuenciapara ignorar el padding
        packed_embedded = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pasa la secuencia empaquetada por la LSTM
        _, (hidden, _) = self.lstm(packed_embedded) # devuelve el resultado de cada palabra y los estados, oculto y de cada celda

        # El estado oculto final de la ultima capa, hidden[-1] resume toda la frase
        final_hidden = hidden[-1]

        # Capa lineal de salida, sin recorte ya que se ahace en el post procesado
        out = self.fc(final_hidden)

        return out.squeeze(-1) # elimina una dimensione xtra innecesaria
    


# Envoltorio para instanciar, entrenar y evaluar el MLP con el arnes actual
class BoWMLPWrapper:
    def __init__(self, min_freq: int = 2, hidden_size: int = 64, batch_size: int = 32, max_epochs: int = 50, patience: int = 5):
        self.min_freq = min_freq
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        
        self.vectorizer = None
        self.model = None
        # Uso de GPU/MPS si esta disponible, sino CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")


    def fit(self, train_df: pd.DataFrame) -> None:
        torch.manual_seed(SEED)
        # Vectorizacion estricta en entrenamiento
         # Convierte texto en numeros (vectores de caracteristicas) para eliminar "basura"
         # que son palabras con erratas o que solo se ven 1 vez

        X_text = train_df["tokens"].apply(lambda x: " ".join(x)).values
        y_full = train_df[TARGET].values.astype(np.float32)

        # Divide primero en train (90%) y val interna (10%) estratificando por stratum
        # train_test_split divide los datos de forma aleatoria
        X_train_text, X_val_text, y_train, y_val = train_test_split(
            X_text, y_full, test_size=0.10, random_state=SEED, stratify=train_df["stratum"]
        )

        # 2. Ajustar el vocabulario SOLO con el 90% de entrenamiento
        self.vectorizer = CountVectorizer(min_df=self.min_freq, token_pattern=r"\S+", lowercase=False) # token_pattern evita perder tokens de un solo caracter como "x"
        # fit_transform lee los textos, aprende que palabras existen y aplica el min_df, 
        # luego convierte cada texto en una fila de numeros en base a ese vocabulario aprendido
        X_train = self.vectorizer.fit_transform(X_train_text).toarray().astype(np.float32)
        X_val = self.vectorizer.transform(X_val_text).toarray().astype(np.float32)
        vocab_size = X_train.shape[1] # Numero total de columnas


        # Dataloader se encarga de organizar y dividr la entrga de datos en partes mas pequeñas
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), # Empareja los datos
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), # Empareja los datos
            batch_size=self.batch_size,
        )


        # Init del modelo, funcion de perdida MSE y optimizador
        self.model = MLPBoW(vocab_size, self.hidden_size).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)


        # BUCLE DE ENTRENAMIENTO CON EARLY STOPPING
        best_val_loss = float("inf") # guarda la menor perdida de val
        best_weights = None # guard los mejores pesos del modelo
        patience_counter = 0 # contador para saber cuanto lleva sin mejorar

        for epoch in range(self.max_epochs):
            
            ###### TRAIN
            self.model.train() # activa modo train
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad() # borra los gradientes de los erroes anteriores
                preds = self.model(batch_x) # forward, el modelo recibe los textos en numeros, y devuelve las predicciones
                loss = criterion(preds, batch_y) # MSE compara las preddciones con las respuestas correctas y calcula loss
                # backpropagation, calcula el error hacia atras, y step hace que adam mueva los pesos de las neuronas
                loss.backward() 
                optimizer.step()


            ####### VAL
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad(): # MODO AHORRO, Esta linea avisa a PyTorch que SOLO quiere hacer predicciones, esto desactiva el rastreo de gradientes y derivadas
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                    preds = self.model(batch_x)
                     # calcula el error promedio lo convierte en un numero y lo multiplica por la cantidad de textos del lote (esto se hace porque el ultimo lote suele ser mas pequeño que el resto)
                    val_loss += criterion(preds, batch_y).item() * batch_x.size(0)
            
            val_loss /= len(X_val) # Toda la perdida acumulada entre el total de textos de val

            ###### STOP
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict()) # hace una copia en memoria con todos los paramentros actuales de las neuronas
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Restaurar los mejores pesos enconrados
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        # Transformar datos de test ignorarndo palabras nuevas
        X_text = test_df["tokens"].apply(lambda x: " ".join(x)).values # junta los tokens en una frase
        X_bow = self.vectorizer.transform(X_text).toarray().astype(np.float32) # usa solo las palabras que aprendió en el entrenamiento

        loader = DataLoader(
            TensorDataset(torch.tensor(X_bow)),
            batch_size=self.batch_size,
            shuffle=False
        )

        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                # Si el batch tiene tamaño 1, squeeze() puede eliminar ambas dimensiones
                # Asegura al menos 1D con view(-1)
                preds = self.model(batch_x).view(-1)
                all_preds.append(preds.cpu().numpy()) # cpu().numpy() si los calculos fueron en la GPU, los trae a CPU y los convierte en arrays de NP

        final_preds = np.concatenate(all_preds) # junta los arrays de todos los lotes en un unico vector

        # Garantizar rango entre 0 y 3
        return np.clip(final_preds, 0, 3)

# Envoltorio para instanciar, entrenar y evaluar la red secuencial RNN
class SeqRNNWrapper:
    """Envoltorio para instanciar, entrenar y evaluar la red secuencial RNN"""
    def __init__(self, min_freq: int = 2, max_len: int = 23, embedding_dim: int = 64, 
                 hidden_size: int = 64, batch_size: int = 32, max_epochs: int = 50, patience: int = 5):

        # Filtrar y rellenar con ceros en SequenceVectorizer
        self.min_freq = min_freq 
        self.max_len = max_len
        # Crear tabla de vectores y la memoria de LSTM
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        # Se guardaran las isntanacias de SequenceVEctorizer y SeqRNN
        self.vectorizer = None
        self.model = None

        # Elige el componente mas rapido del pc para hacera hi los calculos
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def fit(self, train_df: pd.DataFrame) -> None:
        torch.manual_seed(SEED)

        # En la RNN se empieza de las listas de tokens
        tokens_full = train_df["tokens"].values
        y_full = train_df[TARGET].values.astype(np.float32)

        # Validacion interna del 10% estratificada antes de contruir el vocab
        tokens_train, tokens_val, y_train, y_val = train_test_split(
            tokens_full, y_full, test_size=0.10, random_state=SEED, stratify=train_df["stratum"] # garantiza que train y val, tengan la misma proporcion de categorias
        )

        # Vectorizacion del 90% de train
        # El vectorizador aprende solo las palabras de train, y luego transforma las palabras en matrices de IDs con el padding
        self.vectorizer = SequenceVectorizer(min_freq=self.min_freq, max_len=self.max_len)
        self.vectorizer.fit(tokens_train)
        X_train = self.vectorizer.transform(tokens_train)
        X_val = self.vectorizer.transform(tokens_val)

        # Calcular las longitudes reales sin los 0 de relleno para pack_padded_sequence, con maximo de max_len
        # Si media 3, guarda un 3, si media 50, como el tope es 23, guarda un 23
        len_train = np.array([min(len(t), self.max_len) for t in tokens_train], dtype=np.int64)
        len_val = np.array([min(len(t), self.max_len) for t in tokens_val], dtype=np.int64)

        vocab_size = self.vectorizer.vocab_size

        # empaqueta la matriz de texto traducido, las longitudes reales y la respuesta correcta a predecir
        # dataloader reparte el trabajo en lotes (32, 23, 64)
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(len_train), torch.tensor(y_train)),
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(len_val), torch.tensor(y_val)),
            batch_size=self.batch_size,
            shuffle=False
        )

        # Crea la RNN con el tamaño de vocab y la envia a la gpu o cpu
        self.model = SeqRNN(vocab_size, self.embedding_dim, self.hidden_size).to(self.device)

        criterion = nn.MSELoss() # crieterio para regresion
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001) # el optimizador que mueve los numeros segun los erroes

        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            
            ###### TRAIN
            self.model.train() # activa modo train
            for batch_x, batch_len, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_len = batch_len.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad() # borra los gradientes de los erroes anteriores
                preds = self.model(batch_x, batch_len) # forward, el modelo recibe los textos en numeros, y devuelve las predicciones
                loss = criterion(preds, batch_y)# MSE compara las preddciones con las respuestas correctas y calcula loss
                # backpropagation, calcula el error hacia atras, y step hace que adam mueva los pesos de las neuronas
                loss.backward()
                optimizer.step()

            ####### VAL
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad(): # Desactiva el rastreo de gradientes y derivadas
                for batch_x, batch_len, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_len = batch_len.to(self.device)
                    batch_y = batch_y.to(self.device)

                    preds = self.model(batch_x, batch_len)
                    # calcula el error promedio lo convierte en un numero y lo multiplica por la cantidad de textos del lote
                    val_loss += criterion(preds, batch_y).item() * batch_x.size(0)

            val_loss /= len(X_val) # La perdida acumulada entre el total de textos

            ###### STOP
            if val_loss < best_val_loss: # si el error es el mas bajo ,se guarda el recors y se reinicia el contador
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict()) # hace una copia con todos los parametros de las neuronas
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Restaurar los mejores pesos encontrados
        if best_weights is not None:
            self.model.load_state_dict(best_weights)


    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        tokens_test = test_df["tokens"].values
        X_test = self.vectorizer.transform(tokens_test)
        len_test = np.array([min(len(t), self.max_len) for t in tokens_test], dtype=np.int64)

        loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(len_test)),
            batch_size=self.batch_size,
            shuffle=False
        )

        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch_x, batch_len in loader:
                batch_x, batch_len = batch_x.to(self.device), batch_len.to(self.device)
                # Si el batch tiene tamaño 1, squeeze() puede eliminar ambas dimensiones
                # Asegura al menos 1D con view(-1)
                preds = self.model(batch_x, batch_len).view(-1)
                all_preds.append(preds.cpu().numpy()) # cpu().numpy() si los calculos fueron en la GPU, los trae a CPU y los convierte en arrays de NP

        final_preds = np.concatenate(all_preds) # junta los arrays de todos los lotes en un unico vector
        
        # Garantizar rango entre 0 y 3
        return np.clip(final_preds, 0, 3)
                


# Esta clase reserva el ID 0 para el relleno (padding) y el ID 1 para las palabras desconocidas
# asignando un ID único a cada palabra que alcance la frecuencia mínima
# Convierte listas de tokens en secuencias numericas de longitud fija
class SequenceVectorizer:
    def __init__(self, min_freq: int = 2, max_len: int =23):
        self.min_freq = min_freq
        self.max_len = max_len  # maximo de largo de los textos
        self.word2idx = {}  # diccionario {"hola": 2, "adios": 3}
        self.pad_idx = 0  # padding con 0
        self.unk_idx = 1  # unknown con 1
        self.vocab_size = 2 # contador del tamaño empieza en 2

    def fit(self, X: np.ndarray) -> None:
        """Ajusta el vocabulario contando las frecuencias en el conjunto de train"""
        counts = Counter()
        # recorre el conjunto X y cuenta cuantas veces aparece cada palabra
        for tokens in X:
            counts.update(set(tokens))

        for word, freq in counts.items():
            if freq >= self.min_freq: # si la freq es menor a 2, se trata como desconocida
                # gaurda la palabra con el numero actual, y lo incrementa
                self.word2idx[word] = self.vocab_size
                self.vocab_size += 1

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforma las listas de tokens a IDs rellenando hasta max len"""
        sequences = []
        for tokens in X:

            # Palabra a palabra de tokens, la busca en el diccionario
            # Si la palabra existe devuelve su ID, sino existe devuelve por defecto nuk_idx
            seq = [self.word2idx.get(word, self.unk_idx) for word in tokens] # seq es el resultado temporal de la traduccion [2, 4, 1]

            if len(seq) > self.max_len:
                seq = seq[:self.max_len] # si se pasa de 23, corta la lista y se queda con los primeros 23
            else:
                seq = seq + [self.pad_idx] * (self.max_len - len(seq)) # si faltan hasta 23, lo rellena con 0

            sequences.append(seq)

        return np.array(sequences, dtype=np.int64) # el embedding de pytorch necesita enteros de 64b


# ----------------------------------------------------------------------------------------------
# Arnes de evaluacion
# Se crea uno NUEVO en cada pliegue
# ----------------------------------------------------------------------------------------------
# Predice siempre la misma constante, calculada con el entrenamiento (media o mediana)
class ConstantBaseline:
    def __init__(self, statistic: str = "median"):
        assert statistic in ("mean", "median"), "statistic debe ser 'mean' o 'median'"
        self.statistic = statistic
        self.constant = None

    def fit(self, train_df: pd.DataFrame) -> None:
        y = train_df[TARGET].to_numpy()
        self.constant = float(np.mean(y) if self.statistic == "mean" else np.median(y))

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(test_df), self.constant) # se copia el valor de constant en cada osicion del array



# MAE y RMSE de unas predicciones
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {"mae": mae, "rmse": rmse}


# Validacion cruzada, para probar el rendimiento de los modelos dividiendo los datos en folds para asegurarse de que lo datos no son con suerte
# Tambien calcula las preddicion Out of Fold, que son las preddiciones que hace el modelo sobre datos que nunca ha visto en el entrenamiento
# Entrena y evalua un modelo en cada pliegue. Devuelve metricas agregadas y predicciones OOF
def evaluate_cv(
        model_name: str, 
        model_factory: Callable, # se le pasa la "fabrica de modelos" para que el modelo este limpio
        data: pd.DataFrame, 
        fold_id: np.ndarray,
        return_oof: bool = False    
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    oof_rows = []
    for k in range(int(fold_id.max()) + 1):
        train_idx, test_idx = fold_indices(fold_id, k) # indica que posiciones son de train y cual de test para el fold k
        train_df, test_df = data.iloc[train_idx], data.iloc[test_idx]

        model = model_factory()  # modelo nuevo en cada pliegue, asi nada se arrastra del anterior
        model.fit(train_df) # el modelo se entrena con los datos reservados a train

        y_true = test_df[TARGET].to_numpy()
        y_pred = model.predict(test_df) # hace las preddciones 

        row = {"model": model_name, "fold": k, "n_test": len(test_df), **compute_metrics(y_true, y_pred)} # ** desempaquetan el diccionario que devuelve, se evita que se aniden diccionarios
        for year in sorted(data["year"].unique()):  # error por curso: comprueba que mezclar cursos no sesga
            mask = (test_df["year"] == year).to_numpy() # aisla las notas de ese año en concreto
            row[f"mae_{year}"] = compute_metrics(y_true[mask], y_pred[mask])["mae"]
        rows.append(row)

        if return_oof:
            fold_oof = test_df[["year", "evaluator", "work", "section", TARGET]].copy()
            fold_oof["fold"] = k
            fold_oof["pred"] = y_pred
            fold_oof["error"] = y_pred - y_true
            fold_oof["abs_error"] = np.abs(fold_oof["error"])
            oof_rows.append(fold_oof)

    metrics_df = pd.DataFrame(rows)
    if return_oof:
        oof_df = pd.concat(oof_rows, ignore_index=True)
        return metrics_df, oof_df
    return metrics_df


# Media y desviacion tipica entre pliegues de MAE y RMSE, por modelo
def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby("model").agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
    )
    return summary.round(4)


# Wilcoxon de rangos con signo sobre las diferencias pareadas por pliegue entre dos modelos
def wilcoxon_compare(results: pd.DataFrame, model_a: str, model_b: str, metric: str = "mae") -> dict:
    a = results[results["model"] == model_a].sort_values("fold")[metric].to_numpy()
    b = results[results["model"] == model_b].sort_values("fold")[metric].to_numpy()
    assert len(a) == len(b), "los dos modelos deben tener los mismos pliegues"
    _, p_value = wilcoxon(a, b)  # dos colas, con 10 pares el p minimo posible es 2/2^10 = 0.00195
    return {
        "metric": metric, 
        "a": model_a, 
        "b": model_b, 
        "median_diff_a_minus_b": float(np.median(a - b)),    # diferencia entre la mediana del error a y b
        "folds_a_better": int(np.sum(a < b)),  # En algun pligue a ha sido mejor que b?
        "p_value": float(p_value)  # si el p_value es <0.05, significa que es improbable que el modelo haya ganado por suerte.
        }


# Funcion generada por Gemini Flash 3.8, y supervisada por Pablo De La Cruz
# Genera diagrama de dispersion con jitter y recta y=x para cada seccion evaluable
def plot_predictions_by_section(oof_df: pd.DataFrame, activity: int) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Obtener unicamente las secciones reales presentes en los datos
    real_sections = sorted(oof_df["section"].unique(), key=lambda x: int(x) if x.isdigit() else x)
    np.random.seed(SEED)
    
    for i, sec in enumerate(real_sections):
        ax = axes[i]
        sec_df = oof_df[oof_df["section"] == sec]
        
        y_true = sec_df[TARGET].to_numpy()
        y_pred = sec_df["pred"].to_numpy()
        
        jitter = np.random.normal(0, 0.04, size=len(y_true))
        x_jittered = np.clip(y_true + jitter, 0, 3)
        
        ax.scatter(x_jittered, y_pred, alpha=0.15, s=12, edgecolors="none", color="#1f77b4")
        ax.plot([0, 3], [0, 3], color="red", linestyle="--", linewidth=1.5, label="y = x (ideal)")
        
        sec_mae = np.mean(np.abs(y_pred - y_true))
        sec_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        
        ax.set_title(f"Seccion {sec} (n={len(sec_df):,})", fontsize=11, fontweight="bold")
        ax.text(0.05, 0.85, f"MAE: {sec_mae:.3f}\nRMSE: {sec_rmse:.3f}", 
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#cccccc"))
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlim(-0.15, 3.15)
        ax.set_ylim(-0.15, 3.15)
        if i >= 4:
            ax.set_xlabel("Valor real")
        if i % 4 == 0:
            ax.set_ylabel("Prediccion")

    # Panel resumen global en la ultima posicion (indice 7)
    ax_summary = axes[7]
    y_true_all = oof_df[TARGET].to_numpy()
    y_pred_all = oof_df["pred"].to_numpy()
    jitter_all = np.random.normal(0, 0.04, size=len(y_true_all))
    
    ax_summary.scatter(np.clip(y_true_all + jitter_all, 0, 3), y_pred_all, alpha=0.08, s=8, edgecolors="none", color="#2ca02c")
    ax_summary.plot([0, 3], [0, 3], color="red", linestyle="--", linewidth=1.5)
    
    tot_mae = np.mean(np.abs(y_pred_all - y_true_all))
    tot_rmse = np.sqrt(np.mean((y_pred_all - y_true_all) ** 2))
    ax_summary.set_title(f"Global Actividad {activity} (n={len(oof_df):,})", fontsize=11, fontweight="bold")
    ax_summary.text(0.05, 0.85, f"MAE: {tot_mae:.3f}\nRMSE: {tot_rmse:.3f}", 
                    transform=ax_summary.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#cccccc"))
    ax_summary.grid(True, linestyle=":", alpha=0.6)
    ax_summary.set_xlabel("Valor real")

    # Apagar los subplots vacios intermedios si hay menos de 7 secciones
    for j in range(len(real_sections), 7):
        axes[j].axis("off")

    plt.suptitle(f"Actividad {activity}: Predicciones seq_rnn frente a valores reales por seccion", fontsize=14, y=0.98)
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"activity{activity}_predictions_by_section.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Grafica guardada en: {plot_path}")


# ----------------------------------------------------------------------------------------------
# Ejecucion
# ----------------------------------------------------------------------------------------------
# Carga, preprocesa, parte en pliegues y evalua los baselines de una actividad"""
def run_activity(activity: int, source: str = CSV_SOURCE) -> pd.DataFrame:
    data = load_activity(activity, source)
    data["tokens"] = preprocess_corpus(data["feedback"])
    fold_id = assign_folds(data)

    vocab_size = len({token for tokens in data["tokens"] for token in tokens})
    print(f"\n=== Actividad {activity}: {len(data):,} filas | vocabulario tras el pipeline: {vocab_size:,} ===")
    print("filas de test por pliegue:", np.bincount(fold_id).tolist())

    res_seq_rnn, oof_seq_rnn = evaluate_cv(
        "seq_rnn", lambda: SeqRNNWrapper(min_freq=2, max_len=23), data, fold_id, return_oof=True
    )

    results = pd.concat([
        evaluate_cv("const_mean", lambda: ConstantBaseline("mean"), data, fold_id),
        evaluate_cv("const_median", lambda: ConstantBaseline("median"), data, fold_id),
        evaluate_cv("bow_mlp", lambda: BoWMLPWrapper(min_freq=2), data, fold_id),
        #evaluate_cv("seq_rnn", lambda: SeqRNNWrapper(min_freq=2, max_len=23), data, fold_id),
        res_seq_rnn,
    ], ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / f"activity{activity}_cv_results.csv", index=False)

    print(summarize_results(results).to_string())

    oof_path = RESULTS_DIR / f"activity{activity}_seq_rnn_oof.csv"
    oof_seq_rnn.to_csv(oof_path, index=False)
    print(f"\n\nPredicciones OOF guardadas en: {oof_path} ({len(oof_seq_rnn):,} filas)")

    plot_predictions_by_section(oof_seq_rnn, activity)

    print("\n\n--- Pruebas de Wilcoxon ---")
    print("Objetivo RMSE (vs const_mean):", wilcoxon_compare(results, "const_mean", "bow_mlp", metric="rmse"))
    print("\nObjetivo MAE (vs const_median):", wilcoxon_compare(results, "const_median", "bow_mlp", metric="mae"))
    print("Objetivo RMSE (seq_rnn vs const_mean):", wilcoxon_compare(results, "const_mean", "seq_rnn", metric="rmse"))
    print("Objetivo MAE (seq_rnn vs const_median):", wilcoxon_compare(results, "const_median", "seq_rnn", metric="mae"))
    
    # Compara el nuevo RNN secuencial contra el BoW
    print("\n\nSecuencial vs BoW (MAE):", wilcoxon_compare(results, "bow_mlp", "seq_rnn", metric="mae"))
    print("\nSecuencial vs BoW (RMSE):", wilcoxon_compare(results, "bow_mlp", "seq_rnn", metric="rmse"))


    return results


def main() -> None:
    for activity in ACTIVITIES:
        run_activity(activity)


if __name__ == "__main__":
    main()