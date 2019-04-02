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


## Ejercicio 4.

## Ejercicio 5.

## Ejercicio 6.

## Ejercicio 7.