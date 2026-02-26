# Ciencia de Datos — Trabajos Prácticos

<div align="center">

**Universidad de Buenos Aires - FIUBA**  
*Cátedra Martinelli - 2do Cuatrimestre 2025*

---

</div>

## Sobre este repositorio

Este repositorio contiene los trabajos prácticos realizados durante el curso de **Ciencia de Datos** en la cátedra Martinelli de la Facultad de Ingeniería de la Universidad de Buenos Aires (FIUBA). Cada trabajo práctico explora diferentes técnicas, tecnologías y metodologías fundamentales en el campo de la ciencia de datos.

## Estructura del Repositorio

```
2025-2c-tps-CienciadeDatos/
│
├── tp1/
│   ├── tp1.ipynb
│   └── tp1-informe.pdf
│
├── tp2/
│   └── tp2.ipynb
│
├── tp3/
│   ├── tp3_parteI.ipynb
│   ├── tp3_parteII.ipynb
│   ├── tp3_parteIII.ipynb
│   └── tp3_parteIV.ipynb
│
├── consigna.txt
└── README.md
```

---

## Contenidos

### Trabajo Práctico 1: Análisis de datos con Pandas

Análisis exploratorio de datos de ventas utilizando **Pandas**. Se trabaja con múltiples datasets para investigar patrones en órdenes, descuentos por estado, tendencias de ventas y visualización de datos utilizando librerías como Matplotlib, Seaborn y Plotly.

**Tecnologías:** Pandas, Matplotlib, Seaborn, Plotly

---

### Trabajo Práctico 2: Análisis de datos con Spark

Análisis de datos a gran escala utilizando **Apache Spark**. Se procesan y analizan datasets mediante RDDs y DataFrames de Spark, aplicando operaciones distribuidas, transformaciones y agregaciones para extraer insights de grandes volúmenes de información.

**Tecnologías:** Apache Spark (PySpark), RDDs, DataFrames

---

### Trabajo Práctico 3: Clasificación de desastres

Proyecto de Machine Learning dividido en 4 partes para clasificar mensajes relacionados con desastres naturales. El objetivo es desarrollar modelos que identifiquen y categoricen mensajes de emergencia.

**Dataset:** [Natural Language Processing with Disaster Tweets - Kaggle](https://www.kaggle.com/competitions/nlp-getting-started)

**Parte I: Análisis Exploratorio**  
Exploración inicial del dataset de mensajes de desastres. Se realiza preprocesamiento de texto, análisis de frecuencias, generación de wordclouds, tokenización, eliminación de stopwords y análisis de sentimientos usando VADER. Se visualizan patrones y características del dataset para comprender la distribución de categorías.

**Parte II: Machine Learning Baseline**  
Establecimiento de un modelo baseline de clasificación utilizando **Regresión Logística**. Se implementan técnicas de encoding (One-Hot Encoding, Binary Encoding, Target Encoding), vectorización de texto con TF-IDF, normalización de features con StandardScaler, y se entrena el modelo baseline para establecer una métrica de referencia inicial.

**Parte III: Machine Learning Avanzado**  
Mejora del modelo baseline utilizando técnicas más sofisticadas. Se implementan embeddings de texto con **Sentence-BERT**, se entrenan modelos avanzados como Random Forest y XGBoost, y se aplican técnicas de feature engineering y optimización de hiperparámetros para mejorar el rendimiento de clasificación.

**Parte IV: Consignas Adicionales**  
Visualización y análisis de embeddings utilizando **Word2Vec**. Se entrena un modelo Word2Vec con el corpus de tweets, y se utiliza reducción de dimensionalidad (TSNE, UMAP) para visualizar y explorar las relaciones semánticas entre palabras en el espacio vectorial, permitiendo identificar agrupaciones y similaridades entre términos.

**Tecnologías:** Pandas, NumPy, NLTK, scikit-learn (Regresión Logística, Random Forest, StandardScaler), Sentence-BERT (Sentence Transformers), Word2Vec (Gensim), XGBoost, TF-IDF, VADER Sentiment, Category Encoders, TSNE, UMAP

---

