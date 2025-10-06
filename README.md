# 🤖 Comparación de Modelos de Deep Learning

Este proyecto evalúa diferentes arquitecturas de redes neuronales (MLP, GRU y Bidirectional LSTM) para resolver una **tarea de clasificación binaria**.  
Se implementa un pipeline completo de *entrenamiento, evaluación y despliegue* con interfaz web y API.

---

## 📂 Estructura del proyecto

```

TF_Machine-main/
│
├── code/
│   ├── api/                 # API REST con FastAPI
│   │   ├── main.py
│   │   └── source/process.py
│   ├── app/                 # Aplicación web (frontend)
│   │   ├── index.html
│   │   └── public/          # Imágenes de resultados
│   └── notebooks/           # Notebooks de entrenamiento y EDA
│       ├── eda.ipynb
│       └── training.ipynb
│
├── data/
│   ├── merged_data.csv
│   └── response_data.csv
│
├── models/
│   └── model.keras          # Modelo entrenado
│
├── results.md               # Métricas detalladas de cada modelo
└── README.md

````

---

## ⚙️ Requisitos

- **Python 3.10+**
- Librerías principales:
  ```bash
  pip install tensorflow scikit-learn pandas numpy matplotlib seaborn fastapi uvicorn
  ````

---

## 🚀 Ejecución

### 1. Análisis exploratorio (EDA)

Abre y ejecuta el notebook:

```bash
notebooks/eda.ipynb
```

Analiza correlaciones y limpieza del dataset (`merged_data.csv`).

### 2. Entrenamiento de modelos

```bash
notebooks/training.ipynb
```

Entrena los modelos **MLP**, **GRU** y **Bidirectional LSTM** y guarda los resultados en `results.md`.

### 3. API con FastAPI

Desde `/code/api/`:

```bash
uvicorn main:app --reload
```

Endpoint principal:

* `POST /predict` → recibe datos de entrada y devuelve la clase predicha.

### 4. Visualización

Abre `code/app/index.html` en tu navegador para ver el panel de resultados y predicciones.

---

## 🧠 Modelos evaluados

| Modelo      | Exactitud | Comentario                                          |
| ----------- | --------- | --------------------------------------------------- |
| **MLP**     | 0.74      | Mejor rendimiento general, balanceado entre clases. |
| **GRU**     | 0.51      | Bajo desempeño, recall alto solo para una clase.    |
| **Bi-LSTM** | 0.50      | Sesgo hacia la clase 1, rendimiento desigual.       |

> 📈 El MLP fue el modelo más estable, logrando F1-score promedio de **0.74**.

---

## 📊 Resultados destacados

* **MLP:** Mejor balance entre precisión y recall.
* **GRU:** Detecta mejor la clase positiva pero con menor exactitud global.
* **Bidirectional LSTM:** Sobresaturado hacia una clase, requiere reentrenamiento o regularización.

---

## 🏁 Próximos pasos

* Integrar almacenamiento en cloud (S3 o Firestore).
* Optimizar hiperparámetros del MLP.
* Desplegar API en un entorno serverless (AWS Lambda / Render).

```


