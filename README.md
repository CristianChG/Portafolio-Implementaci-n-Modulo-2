# Clasificador de personas con y sin lentes

**Objetivo:** Detectar automáticamente si una persona **usa lentes** o **no** a partir de una imagen facial, con un pipeline reproducible de entrenamiento.

---

## 1) Contexto y motivación

* **Problema:** automatizar una tarea binaria (con/sin lentes) útil como etapa previa en flujos de verificación/registro.
* **Caso de uso:** equipos que etiquetan imágenes o aplicaciones que necesitan filtros/ajustes en función de si el sujeto porta lentes.
* **Criterio de éxito:** alcanzar **≥ 90%** de accuracy en validación/test con un modelo compacto y fácil de desplegar.

## 2) Objetivo del trabajo

* Construir un **baseline CNN** simple, entrenado con data augmentation ligero y **early stopping / checkpoint** al mejor `val_loss`.
* Entregar **dos notebooks** (training + predicción) y dejar listo el guardado/carga del modelo (`.keras`).

## 3) Dataset

* **Fuente:** Kaggle — *facecropped-glasses-vs-noglasses-dataset*.
* **Split provisto:** `train/`, `val/`, `test/` (carpetas por clase `with_glasses/` y `without_glasses/`).
* **Tamaño (detectado en notebook):**

  * `train`: **76,049** imágenes (con lentes: **27,513** · sin lentes: **48,536**)
  * `test`: **3,803** imágenes (con lentes: **1,376** · sin lentes: **2,427**)
  * `val`: incluido en el dataset (conteo no impreso en la corrida final)
* **Balance:** ligera **desproporción** hacia *sin lentes*; se compensa parcialmente con **stratify** (en el split ya provisto) y **data augmentation**.

## 4) ETL / Preparación de datos

* **Carga:** `tf.keras.utils.image_dataset_from_directory` sobre cada split.
* **Tamaño de imagen:** `64×64` RGB.
* **Batching:** `BATCH_SIZE = 32` · `AUTOTUNE` para `prefetch`.
* **Normalización:** `layers.Rescaling(1./255)` (y/o mapeo previo en el dataset, según celda ejecutada).
* **Data augmentation:** `RandomFlip("horizontal")` + `RandomRotation(0.05)`.
* **Etiquetas:** binarias `0/1` inferidas por subcarpeta.

> Archivos/Notebooks: `Portafolio_Implementación_Modulo_2_Training.ipynb` (training) y `prediccion.ipynb` (inferencia).

## 5) Modelo base (CNN ligera)

**Arquitectura (resumen):**

```
Input (64,64,3)
→ DataAugmentation (flip/rotation)
→ Conv2D(16, 3×3, relu) + MaxPool
→ Conv2D(32, 3×3, relu) + MaxPool
→ Conv2D(64, 3×3, relu) + MaxPool
→ GlobalAveragePooling2D
→ Dropout(0.2)
→ Dense(32, relu)
→ Dense(1, sigmoid)
```

* **Épocas:** se probaron corridas con tope de 30; la corrida final registrada establece `EPOCHS = 10` y depende de *early stopping*.

## 6) Métricas

**Validación (mejores épocas observadas en logs):** `val_accuracy ≈ 0.988` · `val_loss ≈ 0.0375`.

**Prueba (test):**

| Métrica      |      Valor |
| ------------ | ---------: |
| **Accuracy** | **0.9916** |
| **Loss**     |     0.0264 |


## 7) Resultados y hallazgos

* Con una **CNN compacta** y **aumentos ligeros**, se logra **>99%** en test, lo cual es **excelente** para un baseline.
* El **input reducido (64×64)** y el **número de params (~25k)** facilitan el despliegue en ambientes con recursos limitados (CPU o dispositivos modestos).
* El **desbalance** leve hacia *sin lentes* no impidió que el modelo alcance alta precisión; para análisis por clase se recomienda añadir *reportes por clase* en la siguiente iteración.

## 8) Cómo reproducir

### Requisitos

* Python `>=3.10`, TensorFlow 2.x (en Colab funciona con `2.19.0`).

### Entrenamiento (Colab)

1. Ejecutar `Portafolio_Implementación_Modulo_2_Training.ipynb`.
2. El notebook descarga el dataset vía `kagglehub`, prepara *datasets* (`train/val/test`), construye y entrena la CNN con callbacks, y **guarda** `best_glasses_classifier.keras` (por `ModelCheckpoint`).

### Inferencia (Colab)

1. Coloca el archivo `best_glasses_classifier.keras` en tu Drive (ruta usada en el notebook: `/content/drive/MyDrive/glasses_classifier/`).
2. Ejecuta `prediccion.ipynb`, que **monta Drive**, **carga el modelo** y permite realizar `model.predict(...)` sobre lotes de imágenes.