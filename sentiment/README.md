# Trabajo Práctico 2 - Twitter sentyment analysis 
## Ejercicio 1: Corpus de Tweets: Estadísticas Básicas

Se programó el script stats.py para que devuelva, la distribución para cada dataset de acuerdo a la polaridad de sus tweets.
Los datasets y su distribucion se muestran a continuacion

|Dataset |Train |P  | NEU | N | NONE |
|--------|------|---|------|----|----|
|PE | 1000 |231 | 166 | 242| 361|
|ES | 1008 |139 | 418 | 318| 133|
|CR | 800|  165 | 94 | 311 |230 |

## Ejercicio 2: Mejoras al Clasificador Básico de Polaridad

A continuacion se propusieron tres clasificadores, Logistic Regression, Support Vector Machine y Multinomial Naive Bayes.

|Modelo:SVM |ES | CR | PE|
|----------|----|----|----|
|Accuracy:| 51.58% | 46.67%  | 39.20%  |
|Macro-Precision:| 39.53% | 37.31% | 33.04% |
|Macro-Recall:| 38.75% | 38.79% | 34.43% |
|Macro-F1:| 39.14% | 38.04% | 33.72% |


|Modelo: LR |ES | CR | PE |
|----------|----|----|----|
|Accuracy:| 53.16% | 46.67% | 42.00% |
|Macro-Precision:| 37.13% | 40.01%| 33.86% |
|Macro-Recall:| 37.34% | 37.86% | 34.70% |
|Macro-F1:| 37.23% | 38.91% | 34.28% |


|Modelo: MNB |ES | CR | PE |
|----------|----|----|----|
|Accuracy:| **54.55%**  | **49.00%** | **45.00%** |
|Macro-Precision:| 54.19% | 66.93%| 30.16% |
|Macro-Recall:| 34.96% | 36.58% | 31.51% |
|Macro-F1:| 42.50% | 47.31% | 30.82% |


Con el objetivo de mejorar la clasificacion, se realizaron diferentes modificaciones al preprocesamiento de los tweets.
Se realizaron las siguientes mejoras.
- Cambio de Tokenizer: dado que los tweets están escritos en leguaje coloquial, se utilizó TweetTokenizer de NLTK
- Normalización básica de tweets: Se reemplazaron todas las menciones de usuarios por el token @USER, del mismo modo las url fueron reemplazadas por el token URL. La repetición consecutiva de mas de 3 caracteres fueron contraidas a 3 caracteres.
- Filtrado de Stopwords: Se uso el conjunto de stopwords de NLTK, ignorando el token "no" ya que el mismo tiene alta cognotación semántica para los tweets negativos.
- Lemmatización: se utilizó WordNetLemmatizer para Lemmatizar los tokens
- Binarización de las palabra: el Tokenizador ignora múltiples ocurrencias del mismo token en el tweet. 
- Se eliminaron los acentos de las palabras (prueba extra)
- En vez de usar CountVectorizer se uso TfIdfvectorizer

A continuación se muestran los resultados de estas mejoras, sobre el svm para cada uno de los conjunto de datos.



|SVM con mejoras| ES | CR | PE |
|------|------|------|------|
|Accuracy:| 51.98% |52.67% | 42.80%|
|Macro-Precision:| 40.27% | 45.92% | 37.21% |
|Macro-Recall:| 39.90% | 44.26% | 38.52% |
|Macro-F1:| 40.09% | 45.07% | 37.86% |



|MNB con mejoras| ES | CR | PE |
|------|------|------|------|
|Accuracy:| 54.74% | 52.33% | 50.20%  |
|Macro-Precision:| 78.37% | 69.57% | 36.03% |
|Macro-Recall:| 35.54% | 39.27% | 34.01% |
|Macro-F1:| 48.90% | 50.20% | 34.99% |



|LR con mejoras| ES | CR | PE |
|------|------|------|------|
|Accuracy:| **55.53%**  | **52.67%**  | **49.60%**  |
|Macro-Precision:| 38.38% | 66.79% | 38.51% |
|Macro-Recall:| 36.99% | 40.72% | 37.44% |
|Macro-F1:| 37.67% | 50.60% | 37.97% |

EN general la mejora de performance ronda entre un 1~6% de accuracy.

## Ejercicio 3: Exploración de Parámetros ("Grid Search")

Luego de haber introducido mejoras en el preprocesamiento, con el objetivo de seguir mejorando la accuracy del modelo. 
Para ello trabajamos sobre los modelos de SVM y Logistic regression. Mediante una busqueda de grid search donde se busco parametros para C y Penalty donde los posibles valores fueron C = [.1, .01, .001, .00001, 1., 5., 10., 100.] y penalty = [l1, l2]



|SVM|C|Accuracy|
|------|-----|-----|
|ES|0.1|0.538|
|CR|0.1|0.497|
|PE|0.1|0.430|



|LR|C|Penalty|Accuracy|
|------|-----|-----|-----|
|ES|1.0|l1|0.540|
|CR|1.0|l2|0.495|
|PE|1.0|l2|0.428|

Para poder decidir cual es la mejor combinacion, se reporto la accuracy para cada modelo y en base a eso se seleccionario los parametros que se muestran en las tablas anteriores. 

## Ejercicio 4: Inspección de Modelos
Features que mas aportan por clase

### Negativo
no 2.0879
triste 1.8400
odio 1.3438
estan 1.2939
peor 1.2710
feo 1.1198
mismo 1.1140
puto 1.0598
puta 1.0269
cosa 1.0261

### Positivo

### NEU

### NONE

To be continue.. 
