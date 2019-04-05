# Trabajo Práctico 1 - Modelado de Lenguaje

## Ejercicio 1, Corpus.
Para este trabajo seleccione un corpus de comentarios de noticias del sitio [https://www.project-syndicate.org/](https://www.project-syndicate.org/). Este conjunto de datos fue utilizado como datos de entrenamiento para [First Conference on Machine Translation (WMT16 - Berlin, Alemania)](http://www.statmt.org/wmt16/). Obtenido gracias al trabajo de [Cristian Cardelino](https://github.com/crscardellino/sbwce).

El dataset cuenta con un total de **288771 comentarios**(62 Mb en total) de opinión escritos en español, el corpus cuenta con **99698 tokens unicos** con un largo **promedio de 30 tokens** por comentario. La carga del corpus se realizo con PlaintextCorpusReader, el dataset está realmente limpio, libre de emojis algo habitual en los comentarios, las palabras están bien escritas sin repetición de vocales. Esto hace que la tokenización sea realmente buena usando el tokenizador de NLTK, por lo que no fue necesario modificar o agregar reglas de tokenizacion.

## Ejercicio 2, Modelos N-Gramas.
Para este ejercicio se definió una función auxiliar, la cual agrega los delimitadores de las oraciones, al inicio n-1 veces \<s> y un \</s>. Agregar al inicio de cada oracion , n-1 veces el delimitador de inicio nos permite librarnos de chequear el tamaño de los ngramas.
Esta función fue utilizada para la inicialización del modelo, que toma como entrada el tamaño de los n-gramas junto con los comentarios tokenizados. Y con ello se construye un diccionario contado las ocurrencias de los n-gramas y los (n-1)-gramas. Luego estas ocurrencias son usadas para calcular las probabilidades de ocurrencias.

Luego de haber construido el diccionario con las ocurrencias de los n-gramas y (n-1)-gramas, nos permite calcular con que probabilidad ocurrira una palabra dado que ocurrieron previamente las n-1. Usando la suposición de Markov, para calcular esto lo que se hace es dividir las ocurrencias de los n-gramas sobre el (n-1)-grama, se repite este proceso para todos los n-gramas que componen la oración y se multiplican estos valores. Como estas probabilidad son son muy pequeños puede producir underflow. Para ello a esa probabilidad la calculamos de otra forma, en vez de multiplicarlas, se le aplica logaritmo en base 2 a la suma de las probabilidades. Además de evitar el problema de underflow, tenemos una mejora en cuestión de eficiencia ya que sumar es más "barato" que multiplicar.

## Ejercicio 3, Generación de Texto.

En el ejercicio anterior, construimos una clase que nos permite construir modelos de n-gramas con el n que seleccionemos. La catedra proveyo scripts para entrenar el model y evaluarlo. En esta actividad el objetivo fue, a partir de los modelos que se pueden construir con la implementación de n-gramas generar texto, para ello en el script *generate.py* a partir de un modelo entrenado mediante el script *train.py* se implementó la generación de oraciones en base al modelo entrenado. Para ello en el archivo *ngram_generator.py*, en base al ejemplo visto en clases, se construyó la clase **NGramGenerator**. Esta clase cuenta con dos métodos principales, uno que genera tokens (generate_token) y otro que genera oraciones (generate_sent). Para generar tokens usamos el método de la transformada inversa.
Para generar oraciones partimos del token de inicio \<s> y luego se llama sucesivamente a generate token, que genera un token en base a los previos, de acuerdo a las probabilidades calculadas cuando se inicializa la clase en base al modelo de n-gramas entrenado previamente.

### Ejemplos de oraciones generadas.

A continuación se muestran las oraciones generadas con los modelos de ngramas con n = 1,2,3,4

| Modelo N-grama| Oraciones generadas |
| ------------- | ------------- |
| Unigramas  | - de a en de la una bancos tatuada menos a la los al Presidente las no El compromiso , los una 18 seriedad que ya un por global como aunque actualidad de complicados alucinante aprobado menos mejoradas esfuerzo <br> - trazas que cultivada leyes 2006 para las que en definen los Acemoglu empleadas y de patente imperio demócratas ese educación Toda : historia una en 1996 contra Unidos infiltración serio las luchar formación hogares como económico Los una y las méritos sufrir de momento Las gobierno <br> - de actualidad Al externa estos Guerra como en el que capacidades complementado viene Ahora , críticos y hizo <br>- en puede , el . momento militar los fronterizas la determinar|
| Bigramas  | - La franja media anual del SBS hoy consideran miembros más largo plazo . <br> - y de la Administración europea , el gasto público definido .<br> - Una falta de que es la población que deben tener más racional . <br> -Y qué quería reformar la Asamblea General de la banca en el enorme diferencia entre ellos .  |
| Trigramas  | - Algunos de los gobiernos restrinjan el gasto a medida que las heterodoxas dependen del apoyo popular dentro de ellos o sus vecinos - se basan en la Constitución de Estados Unidos , al menos el diez por ciento de dicho crecimiento se puede compensar cualquier empeoramiento de su país es capaz de defender una política migratoria para atraer una inversión , y clonar animales . <br> -Por el momento , el VIH / sida . <br> - El ejemplo más notable fue el tercer trimestre de 2008 - 2009 será cerca del 90 por ciento actualmente y las relaciones sinojaponesas . <br> - Si bien una estabilización continua persistió en los de independencia . |
| Cuatrigramas  | - El nuevo presidente de Francia , que combinaba un gaullismo dominante y un Partido Comunista que el país era menor a un desembolso masivo de recursos financieros y de gestión idónea de los asuntos públicos de las sociedades como la de México o Chile . <br> - Si no se contiene pronto el brote , es además un llamado a las armas . <br> - La última elección que produjo un cambio cuando una coalición de unos vencedores que profesaban valores compartidos .<br> - Requerirá recursos importantes y el proceso principal que contribuye a este malestar : la importancia de tomar medidas ni actuar como si tuviéramos una alternativa de poder , las perspectivas de empleo , pero el prolongado conflicto salarial con los empleados . |

Lo que podemos observar acá es que cuando usamos modelos de unigramas, las oraciones no tienen mucho sentido, a medida que el modelo de n-gramas es más grande, las oraciones empiezan a cobrar mayor sentido. Más aún, en el modelo de bigramas ya podemos ver que finaliza cada oración con un punto. Un detalle que se puede ver, son los signos de puntuación, el tokenizador los está tomando como un token y eso hace que los puntos finales de las oraciones esten con un espacio luego de la última palabra. Pasa lo mismo con las comas, dos puntos, etc, se podría mejorar esto para que la generación de oraciones sea más "natural".

## Ejercicio 4.
Este ejercicio constó de programar el suavizado Add-One para ello se definió una nueva clase AddOneNgram, la misma hereda de NGram. La inicialización de la clase es similar a la de NGram pero se agrega el tamaño del vocabulario . Además se modifica el calculo de las probabilidades usando la siguiente formula

![img](http://latex.codecogs.com/svg.latex?P%28word%29%3D%5Cfrac%7Bwordcount%2B1%7D%7Bnumberofwords%2BV%7D)

Además se agregó en la interfaz de train.py la opción de elegir este modelo.

## Ejercicio 5.
Primero para este ejercicio se dividió el dataset en %90 para entrenamiento y %10 reservado para test. Y se implementaron tres métricas vistas en clase, log-probability, cross-entropy y perplexity. Se agregó a la interfaz de eval el llamado al cálculo de estas tres métricas. Por lo tanto se entrena el modelo con el %90 del dataset y luego se evalua con el %10 restante.

Log-probability (más alta mejor)

![img](http://latex.codecogs.com/svg.latex?L%3D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%7B%5Clog_%7B2%7D%28p%28x%5E%7Bi%7D%29%29%7D)

Cross-entropy (más baja mejor)

![img](http://latex.codecogs.com/svg.latex?H%3D%5Cfrac%7B-1%7D%7BM%7D%2AL)

Perplexity (más baja mejor)

![img](http://latex.codecogs.com/svg.latex?2%5E%7BH%7D)


Se entrenaron modelos Add-One con n = 1,2,3,4 y se calculo el valor de perplexity para cada modelo los resultados se muestran a continuación.

|Ngrama|Perplexity|
|:----:|:--------:|
|1| 973.23 |
|2| 1561.06 |
|3| 13810.55 |
|4| 37666.34 |


En la tabla podemos ver que a medida que crece el n, crece la medida de perplexity lo cual no es lo que deseeamos. El objetivo es minimizar la perplexity.

## Ejercicio 6.
En este ejericio se implemento el suavizado por interpolacion, creando una nueva clase llamada InterpolatedNGram. La cual incluye la estimación del gamma por barrido de valores. En caso que no se pase el parámetro gamma, se usa el 10% del conjunto de entrenamiento como conjunto de Held-Out para estimar este parametro.
Se agregó a la interfaz de train.py la opción para utilizar este modelo.
Se entrenaron modelos con diferentes valores de n = 1, 2, 3, 4. Primero se reporta el barrido de gamma para encontrar el "mejor" gamma para cada modelo

|Gamma| log prob (unigramas)| log prob (bigramas)|log prob (trigramas)|log prob (cuatrigramas)|
|:---:|---------------------|--------------------|--------------------|-----------------------|
|10   |**-8521199.334863318** | -6084102.56629145  | -6096795.83087352  | -6239268.536773331  |
|60   |-8521199.334863318   | -5977344.166649291 | -5790151.842994055 | -5800524.714881214  |
|110  |-8521199.334863318   | **-5967872.554864988** | -5748454.090111691 | -5738518.196349467  |
|160  |-8521199.334863318   | -5970449.028596952 | **-5738869.232825182** | -5720562.604646363  |
|210  |-8521199.334863318   | -5976672.607309182 | -5739609.617010944 | **-5716837.49348453** |
|260  |-8521199.334863318   | -5984238.239187142 | -5744673.13759559  | -5719197.964933377  |

En la tabla podemos ver que el valor de Gamma varia segun el tamaño de los ngramas.
para unigramas, el mejor gamma es 10. bigramas 110, trigramas 160 y cuatrigramas 210. Luego de haber fijado los gamma para cada modelo se calculo la perplexity de cada modelo. A continuación se reporta la medidad de perplexity para estos modelos.

|Ngrama|Perplexity|
|:----:|:--------:|
|1| 1991.95 |
|2| 207.42 |
|3| 169.81 |
|4| 166.73 |

Como podemos ver en la tabla a medida que crece el n, la medida de perplexity disminuye. 

## Ejercicio 7.
En este ejercicio se implementó el suavizado Back-Off. Para ello construimos la clase BackOffNGram. Al igual que en el anterior ejercicio se implementó un metodo para estimar el valor de Beta en este caso mediante un barrido de posibilidades. Sabemos a priori que el valor de Beta será entre 0 y 1 entonces partimos el intervalo en diez partes y el beta seleccionado será el que maximize la log prob luego se fija ese Beta para el modelo. Para cada uno de los modelos se reporta a continuación el barrido para conseguir el valor de Beta. 

|Beta | Unigramas | Bigramas | Trigramas | Cuatrigramas|
|:---:|-----------|----------|-----------|-------------|
| 0.0 | **-7718025.352601002** | -inf | -inf |  -inf |
| 0.1 | -7718025.352601002 | -5880806.400129013 | -6046083.138597587 | -6446313.496414582  |
| 0.2 | -7718025.352601002 | -5804416.503966612 | -5773996.051445964 |  -6005103.339433806 |
| 0.3 | -7718025.352601002 | -5765067.515546679 | -5628665.9717234755 |  -5767994.1890726965 |
| 0.4 | -7718025.352601002 | -5741661.351914602 | -5537104.245458065 | -5616769.084029178  |
| 0.5 | -7718025.352601002 | -5727882.912910021 | -5477271.90146417 | -5515674.741378643  |
| 0.6 | -7718025.352601002 | **-5721369.88085302** | -5440572.9186013825 | -5450577.389939791  |
| 0.7 | -7718025.352601002 | -5721675.4039853215 | **-5424607.53588324** | **-5417100.014850051** |
| 0.8 | -7718025.352601002 | -5730389.661861994 | -5432926.706137322 | -5419814.746792188  |
| 0.9 | -7718025.352601002 | -5754942.180876019 | -5485009.447407748 | -5486436.636362652  |

Se entrenaron modelos con diferentes n = 1, 2, 3, 4. A continuacion se reporta la perplexity para cada uno de ellos

|Ngrama|Perplexity|
|:----:|:--------:|
|1| 1.0070686524364396 |
|2| 1.0052459310986406 |
|3| 1.0049781914824645 |
|4| 1.0049733867654602 |


## Comparacion entre modelos de suavizado
A continuacion se muestran los valores de perplexity para cada tamaño de ngrama y modelo de suavizado. El objetivo de esta tabla es tener una comparación de los modelos.

|Modelo Suavizado| Unigramas | Bigramas | Trigramas | Cuatrigramas |
|:--------------:|--------|---------|----------|----------|
| Add One        | 973.23 | 1561.06 | 13810.55 | 37666.34 |
| Interpolate    | 1991.95 | 207.42 | 169.81 | 166.73 |
| Back-Off       | 973.71 | 165.90 | 127.29 | 127.88 |
