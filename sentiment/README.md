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
|Accuracy:| **55.34%** | **53.00%** | 49.60%|
|Macro-Precision:| 40.76% | 67.03% | 38.20% |
|Macro-Recall:| 36.78% | 40.91% | 37.92% |
|Macro-F1:| 38.67% | 50.81% | 38.06% |


|MNB con mejoras| ES | CR | PE |
|------|------|------|------|
|Accuracy:| 54.74% | 52.33% | **50.20%**  |
|Macro-Precision:| 78.37% | 69.57% | 36.03% |
|Macro-Recall:| 35.54% | 39.27% | 34.01% |
|Macro-F1:| 48.90% | 50.20% | 34.99% |



|LR con mejoras| ES | CR | PE |
|------|------|------|------|
|Accuracy:| 55.23%  | 52.67%  | 49.60%  |
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
10 features que mas aportan a la clase negativa

|Feature| Peso |
|-------|------|
| no | 2.0879 |
| triste | 1.8400 |
| odio | 1.3438 |
| estan | 1.2939 |
| peor | 1.2710 |
| feo | 1.1198 |
| mismo | 1.1140 |
| puto | 1.0598 |
| puta | 1.0269 |
| cosa | 1.0261 |

10 features que menos aportan a la clase negativa

|Feature| Peso |
|-------|------|
| @USER | -0.9955 |
| bonito | -1.0925 |
| buen | -1.1068 |
| primer | -1.1182 |
| mejor | -1.1562 |
| buena | -1.2147 |
| genial | -1.2426 |
| gracias | -1.5045 |
| URL | -1.6904 |
| ! | -2.6488 |


### Positivo
10 features que mas aportan a la clase positiva

|Feature| Peso |
|-------|------|
| ! | 2.9962 | 
| gracias | 2.1531 |
| buen | 2.0653 |
| genial | 1.6863 |
| mejor | 1.5717 |
| guapa | 1.5389 |
| feliz | 1.2997 |
| buenos | 1.1467 |
| bonito | 1.1370 |
| URL | 1.0888 |


10 features que menos aportan a la clase positiva

|Feature| Peso |
|-------|------|
|peor | -0.6631 |
|alguna | -0.6660 |
|si | -0.6843 |
|odio | -0.7109 |
|alguien | -0.7488 |
|pues | -0.7636 |
|pone | -0.7720 |
|? | -0.9156 |
|triste | -1.1995 |
|no | -2.0758 |


### NEU
10 features que mas aportan a la clase NEU

|Feature| Peso |
|-------|------|
| si  | 1.3556 |
| nerviosa  | 1.0524 |
| casa  | 0.9947 |
| aunque  | 0.9659 |
| claro  | 0.8959 |
| pasa  | 0.8855 |
| hacen  | 0.8828 |
| vez  | 0.8765 |
| serio  | 0.8586 |
| dice  | 0.8273 |


10 features que menos aportan a la clase NEU

|Feature| Peso |
|-------|------|
|hacer | -0.5426 |
|creo | -0.5640 |
|buen | -0.5706 |
|quiero | -0.5991 |
|buenos | -0.6231 |
|cosas | -0.7406 |
|? | -0.7619 |
|ano | -0.7683 |
|hoy | -0.9171 |
|gracias | -1.0973 |


### NONE
10 features que mas aportan a la clase NONE

|Feature| Peso |
|-------|------|
| ? | 2.2908 |
| " | 1.2808 |
| semana | 1.2582 |
| alguna | 1.0930 |
| video | 1.0264 |
| jugar | 0.9578 |
| primer | 0.9196 |
| ahora | 0.8912 |
| URL | 0.8493 |
| fechas | 0.8416 |


10 features que menos aportan a la clase NONE

|Feature| Peso |
|-------|------|
|hoy | -0.6634  |
|triste |-0.6715 |
|ser |-0.6731 |
|siempre| -0.6951 |
|vida |-0.6983 |
|buen |-0.7027 |
|estan| -0.8463 |
|ma |-0.8681 |
|mal| -0.9625 |
|...| -1.0747 |


Los features que mas aportan por clases tienen sentido en general de acuerdo a cada clase. 
Resulta extraño que el token de URL aparezca como el decimo mejor token de la clase positiva. 
El signo de exclamación ! que sea el primero token que mas aporta, da la idea que tweets que usan ! es para hacer enfasis en sentimientos positivos.
Como trabajo a realizar seria sacar los signos de puntuación de los features, para ver si se agregan token mas representativos por clase.

## Ejercicio 5: Análisis de Error.

Agregar Tabla con instancias mal clasificadas

|Tweet | Prediccion | Etiqueta | Distancia|
|------|------------|----------|----------|
|@otonashi_saya Pero la verdad es que en general no voy mal de motivación... casi al revés, me sobra motivación (y ego  |N |P |0.4240|
|@Nadieelosabe Vale vuelvo a preguntar. No sabia no que te siguiera  |N |NONE |0.4159|
|Bueno, pues vamos a recuperar el tiempo perdido. Empezare por descargarme el Manchester City - West Ham, a ver que tal  |N |P |0.4141|
|Cuando no puedo dormir, escribo todo lo que preocupa en una libreta que alguien me regaló y es como un somnífero instantáneo  |N |P |0.4110|
|En realidad no soy tan blanca  |N |NEU |0.4106|
|A mí nunca me podrán hacer una broma porque no cojo llamadas y menos cuando son ocultas  |N |NONE |0.4060|
|No puedo evitar ir a beber una lata de refresco y acordarme de @LourdesBiurrun family y terminar sirviéndome en vaso...  |N |NEU |0.3983|
|Mi madre me deja ponerme rubia pero no el pelo morado  |N |NEU |0.3958|
|@SergioRevolS Pero este verano tampoco ha llegado a hacer calor, sobre todo si lo comparamos con el pasado  |N |P |0.3879|
|@albertbru una especie de Titanic pero en versión cutre no?  |NONE |N |0.3669|



Instancia mal clasificada
'Cuando no puedo dormir, escribo todo lo que preocupa en una libreta que alguien me regaló y es como un somnífero instantáneo '


|N | NEU |NONE |P|
|---|----|----|---|
|0.57506126 |0.12920719| 0.13164904 |0.1640825 |

Distancia entre el tag y clase predicha: 0.410979


|Feature|Peso|
|-----|------|
|no | 2.0879 |
|puedo | 0.6061 |
|dormir | 0.4247 |
|alguien | 0.0520 |
|escribo | -0.3563 |
|, | -0.5794 |

Podemos ver que el token "no", aporta significativamente a la clase negativa. Pero semanticamente no es una negacion, expresa una situacion.

Reemplazamos el token "no" por la frase tengo insomnio.
'Cuando tengo insomnio, escribo todo lo que preocupa en una libreta que alguien me regaló y es como un somnífero instantáneo '

|N | NEU |NONE |P|
|---|----|----|---|
|0.36225665 |0.15590209| 0.21313609| 0.26870516|


|Feature|Peso|
|-----|------|
|alguien | 0.0520 |
|escribo | -0.3563 |
|, | -0.5794 |

Agrego ahora tokens positivos con el objetivo de ver cuantos tokens positivos cambia la prediccion
'Cuando tengo insomnio, escribo todo lo que preocupa en una libreta que alguien me regaló y es como el mejor somnífero instantáneo '

|N | NEU |NONE |P|
|---|----|----|---|
|0.25377219| 0.12917566 | 0.18663001 | 0.43042214|

|Feature|Peso|
|-----|------|
|alguien | 0.0520 |
|escribo | -0.3563 |
|, | -0.5794 |
|mejor | -1.1562 |


## Ejercicio 6: Análisis Final.

El mejor clasificar obtenido fue un svm, con el preprocesamiento realizado en la seccion previa
Ahora se puso a prueba el mejor clasificador obteenido en el conjunto de datos de test final de interTASS Español.
Los resultados obtenidos son los siguientes:

- Accuracy: 53.92% 
- Macro-Precision: 41.75%
- Macro-Recall: 36.38%
- Macro-F1: 38.88%
