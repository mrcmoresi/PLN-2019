# Trabajo Práctico 1 - Modelado de Lenguaje

## Ejercicio 1, Corpus.
Para este trabajo seleccione un corpus de comentarios de noticias del sitio [https://www.project-syndicate.org/](https://www.project-syndicate.org/). Este conjunto de datos fue utilizado como datos de entrenamiento para [First Conference on Machine Translation (WMT16 - Berlin, Alemania)](http://www.statmt.org/wmt16/). Obtenido gracias al trabajo de [Cristian Cardelino](https://github.com/crscardellino/sbwce).

El dataset cuenta con un total de **288771 comentarios**(62 Mb en total) de opinión escritos en español, el corpus cuenta con **99698 tokens unicos** con un largo **promedio de 30 tokens** por comentario. La carga del corpus se realizo con PlaintextCorpusReader, el dataset está realmente limpio, libre de emojis algo habitual en los comentarios, las palabras están bien escritas sin repetición de vocales. Esto hace que la tokenización sea realmente buena usando el tokenizador de NLTK, por lo que no fue necesario modificar o agregar reglas de tokenizacion.

## Ejercicio 2, Modelos N-Gramas.
Para este ejercicio se definió una función auxiliar, la cual agrega los delimitadores de las oraciones, al inicio n-1 veces <s> y un </s>. Agregar al inicio de cada oracion , n-1 veces el delimitador de inicio nos permite librarnos de chequear el tamaño de los ngramas.
Esta función fue utilizada para la inicialización del modelo, que toma como entrada el tamaño de los n-gramas junto con los comentarios tokenizados. Y con ello se construye un diccionario contado las ocurrencias de los n-gramas y los (n-1)-gramas. Luego estas ocurrencias son usadas para calcular las probabilidades de ocurrencias.

Luego de haber construido el diccionario con las ocurrencias de los n-gramas y (n-1)-gramas, nos permite calcular con que probabilidad ocurrira una palabra dado que ocurrieron previamente las n-1. Usando la suposición de Markov, para calcular esto lo que se hace es dividir las ocurrencias de los n-gramas sobre el (n-1)-grama, se repite este proceso para todos los n-gramas que componen la oración y se multiplican estos valores. Como estas probabilidad son son muy pequeños puede producir underflow. Para ello a esa probabilidad la calculamos de otra forma, en vez de multiplicarlas, se le aplica logaritmo en base 2 a la suma de las probabilidades. Además de evitar el problema de underflow, tenemos una mejora en cuestión de eficiencia ya que sumar es más "barato" que multiplicar.

## Ejercicio 3.

## Ejercicio 4.

## Ejercicio 5.

## Ejercicio 6.

## Ejercicio 7.