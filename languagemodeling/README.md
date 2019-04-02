# Trabajo Práctico 1 - Modelado de Lenguaje

## Ejercicio 1, Corpus.
Para este trabajo seleccione un corpus de comentarios de noticias del sitio [https://www.project-syndicate.org/](https://www.project-syndicate.org/). Este conjunto de datos fue utilizado como datos de entrenamiento para [First Conference on Machine Translation (WMT16 - Berlin, Alemania)](http://www.statmt.org/wmt16/). Obtenido gracias al trabajo de [Cristian Cardelino](https://github.com/crscardellino/sbwce).

El dataset cuenta con un total de **288771 comentarios**(62 Mb en total) de opinión escritos en español con un largo **promedio de 30 tokens** por comentario. La carga del corpus se realizo con PlaintextCorpusReader, el dataset está realmente limpio, libre de emojis algo habitual en los comentarios, las palabras están bien escritas sin repetición de vocales. Esto hace que la tokenización sea realmente buena usando el tokenizador de NLTK. 


