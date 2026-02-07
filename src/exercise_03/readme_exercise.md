
# Exercise 3: Learn a non-linear function with PyTorch

## Objective

El objetivo de este ejercicio es estimar una función desconocida a partir de datos observados mediante un modelo de machine learning.

La función que se pretende modelar es una función no lineal sinusoidal
$$
 y = 100 \sin\left(\frac{8\pi x}{100}\right) + 2
$$ 
 
 El objetivo del ejercicio no es descubrir explícitamente la expresión analítica de la función, sino entrenar un modelo que sea capaz de reproducir su comportamiento a partir de datos, incluso si el modelo actúa como una caja negra.


## Task Formalization

La tarea que se aborda en este ejercicio puede formalizarse en dos etapas. En primer lugar, se define el objetivo del problema. En segundo lugar, se describe el enfoque utilizado para resolverlo mediante entrenamiento supervisado.

### Task Formalization (Inference)

Existe una función desconocida $f$ de la cual se dispone de un conjunto de datos que relacionan valores de entrada $𝑥$ con valores de salida $y$.

$$
y = f(x)
$$

El objetivo es construir un modelo de Machine Learning que aproxime dicha función a partir de los datos disponibles. El modelo aprende un conjunto de parámetros $W$ que permiten expresar la relación entre la entrada y la salida:

$$
y = f(W,x)
$$

Desde el punto de vista gráfico, el proceso de inferencia puede representarse como:


```mermaid
graph TD
    A((x)) --> B["f(W,x)"]
    B --> C((y))
    
```
    


### Task Formalization (Training)

Durante el entrenamiento, el modelo recibe pares de datos 
$(x,y)$. A partir de la entrada $x$, el modelo produce una predicción $y$, que se compara con el valor real $y$ mediante una función de pérdida. Esta pérdida se utiliza para actualizar los parámetros del modelo mediante backpropagation del error y optimización basada en gradiente descendente.

El proceso puede representarse gráficamente de la siguiente manera:


```mermaid
graph TD
    A((x))
    B((y))
    C((y'))
    M["f(W,x)"]
    L(Loss)

    A --> M
    M --> C
    C --> L
    B --> L
    L --> W
    W --> M

```

## Evaluation metrics

Dado que se trata de un problema de regresión, se utilizan las siguientes métricas de evaluación:

Mean Squared Error (MSE): mide el error cuadrático medio.

Mean Absolute Error (MAE): mide el error absoluto medio.

R-squared (R²): indica la proporción de varianza explicada por el modelo.

Estas métricas permiten evaluar tanto la precisión como la capacidad de generalización del modelo.
## Data Considerations

### Dataset description

El dataset es sintético y está compuesto por 10.000 puntos generados a partir de una función cuadrática no lineal. Los valores de entrada $x$ se generan de forma uniforme en el rango [0,100]. La salida $y$ se obtiene aplicando la función real y añadiendo ruido gaussiano con desviación estándar fija para simular datos reales.

### Data preparation and preprocessing

Los datos se convierten a tensores de PyTorch y se reestructuran para cumplir con el formato requerido por el modelo. Posteriormente, el dataset se divide en tres subconjuntos independientes:

70% para entrenamiento

15% para validación

15% para test

### Data augmentation

No se realiza aumento de datos, ya que el dataset es sintético y suficientemente grande para el objetivo del ejercicio.

## Model Considerations




### Suitable Loss Functions

Dado que el problema planteado es un problema de regresión, la función de pérdida debe medir la diferencia entre valores continuos reales y valores continuos predichos por el modelo. Entre las funciones de pérdida más adecuadas para este tipo de tareas se encuentran MSE y MAE.

- Mean Squared Error (MSE): penaliza fuertemente errores grandes.
- Mean Absolute Error (MAE): es más robusta frente a valores irregulares.

### Selected Loss Function

Dado que el dataset incluye ruido gaussiano, una función que penalice más los errores grandes resulta más apropiada, es decir, la función de pérdida MSE.

### Possible architectures

Para aproximar una función no lineal sinusoidal, un modelo lineal como el del primer ejercicio no sería suficiente. Por tanto, hace falta un modelo más complejo.

En este caso se ha diseñado una red neuronal multicapa (MLP) con una capa de entrada de una dimensión, dos capas ocultas de 128 neuronas cada una y una capa de salida de una dimensión. Las capas ocultas utilizan la función de activación ReLU, mientras que la capa de salida no emplea ninguna activación.

Este tipo de arquitectura es capaz de modelar relaciones no lineales complejas, como la función sinusoidal del problema. A diferencia de modelos lineales, una MLP puede aproximar funciones altamente no lineales gracias a la combinación de capas ocultas y activaciones no lineales.

### Last layer activation

Como ya se ha comentado, la capa de salida del modelo no utiliza ninguna función de activación. Esto se debe a que en problemas de regresión la salida puede tomar cualquier valor real, positivo o negativo.

El uso de activaciones como ReLU o Sigmoid en la última capa limitaría ese rango de salida y dificultaría la aproximación de la función objetivo.

### Other Considerations

Los valores de entrada se normalizan al rango [0,1], lo cual mejora la estabilidad del entrenamiento, ya que al escalar los valores de entrada a un rango reducido, se evitan activaciones y gradientes de gran magnitud que pueden dificultar la convergencia del modelo.

## Training

El proceso de entrenamiento se realiza mediante aprendizaje supervisado, utilizando mini-batches y validación en cada época. El modelo se entrena para minimizar la función de pérdida MSE sobre el conjunto de entrenamiento, mientras que el conjunto de validación se emplea para seleccionar el mejor modelo.

Se guarda el modelo que obtiene el menor error de validación, lo que permite evitar sobreajuste.

### Training hyperparameters

- Tamaño del dataset: 10.000 muestras
- Batch size: 10
- Número de épocas: 400
- Learning rate: 0.001
- Optimizador: AdamW
- Función de pérdida: MSE
- Arquitectura: MLP con dos capas ocultas de 128 neuronas

### Loss function graph

![image](../../outs/exercise_03/loss_plot.png)

### Discussion of the training process

Durante el entrenamiento se observa una disminución progresiva de la pérdida tanto en el conjunto de entrenamiento como en el de validación. Las curvas de ambas pérdidas presentan un comportamiento similar, lo que indica que el modelo aprende de manera estable y no memoriza los datos de entrenamiento.

Las oscilaciones que aparecen en la señal de la pérdida de validación se deben al ruido en los datos.

## Evaluation

### Evaluation metrics

Las métricas empleadas para evaluar el modelo son:

- R²: mide la proporción de varianza explicada por el modelo.

- MAE: mide el error medio absoluto.

- MSE: mide el error cuadrático medio.

Estas métricas se calculan de forma independiente sobre los conjuntos de entrenamiento, validación y test.

Las gráficas de regresión que se observan a continuación muestran bastante proximidad entre los valores reales y los valores predichos.

![image](../../outs/exercise_03/train_regression_plot.png)

![image](../../outs/exercise_03/validation_regression_plot.png)

![image](../../outs/exercise_03/test_regression_plot.png)

Metrics for each dataset is depicted: 

![image](../../outs/exercise_03/metrics.png)

### Evaluation results

El valor de R² es cercano a 0.89 en train, validation y test, lo que indica que el modelo explica aproximadamente el 89% de la variabilidad de los datos.

Los valores de MAE se sitúan alrededor de 19, lo cual tiene sentido por la desviación estándar del ruido introducido en el dataset.

Los valores de MSE se mantienen estables entre los distintos conjuntos, lo que demuestra una buena capacidad de generalización.

Las gráficas de puntos muestran que el modelo reproduce correctamente la forma sinusoidal del problema.

Example for train set:

![image](../../outs/exercise_03/train_data_points_plot.png)


Example for validation set:

![image](../../outs/exercise_03/validation_data_points_plot.png)


Example for test set:

![image](../../outs/exercise_03/test_data_points_plot.png)


### Discussion of the results

#### How the model solves the problem?
El modelo aprende la relación entre la entrada y la salida mediante capas ocultas con activaciones ReLU. Esto le permite aproximar correctamente la forma sinusoidal de la función objetivo, a pesar del ruido presente en los datos.

#### Is there overfitting, underfitting or any other issues? 
No se observa ni overfitting ni underfitting. Las métricas de entrenamiento, validación y test son muy similares, lo que demuestra que hay un equilibrio adecuado entre capacidad del modelo y cantidad de datos.
#### How can we improve the model?
Se podría probar a utilizar otras funciones de activación que más adecuadas para este tipo de señal, ajustar el tamaño de batch, probar diferentes parámetros de entrenmiento etc.
#### How this model will generalize to new data?
Dado que las métricas en el conjunto de test son similares a las de entrenamiento y validación, se puede concluir que el modelo generaliza correctamente a datos no vistos previamente.
## Design Feedback loops

Primero se intentó con un modelo simple que estaba formado por una capa lineal sin capas ocultas ni funciones de activación no lineales. Este modelo solo puede aprender relaciones lineales, por lo que no era capaz de aprender la nueva señal y salían muy malos resultados.

Por ello, se diseñó una arquitectura inicial de red neuronal multicapa con una profundidad suficiente para capturar la no linealidad del problema, utilizando capas ocultas con 64 neuronas. Además, se ajustaron los hiperparámetros principales, estableciendo un learning rate moderado y un número de 300 épocas, con el objetivo de permitir una convergencia progresiva del modelo sin incurrir en sobreajuste.

Tras entrenar esta primera configuración, se observó que el valor de la función de pérdida era demasiado alto y que el modelo no lograba aproximar adecuadamente la salida predicha a la salida deseada. Este comportamiento indicaba que el modelo no tenía la capacidad suficiente para modelar la función sinusoidal. Además, la falta de normalización en los datos de entrada provocaba inestabilidades en el entrenamiento y dificultaba la optimización de los parámetros.

Para solucionarlo, se normalizó la variable de entrada, escalando los valores al rango [0,1]. En segundo lugar, se aumentó la capacidad del modelo incrementando el número de neuronas en las capas ocultas de 64 a 128. De esta forma, la red obtuvo más flexibilidad para aprender la forma de la función, ya que con menos neuronas no conseguía capturar correctamente las oscilaciones pronunciadas de la señal sinusoidal. Finalmente, se amplió el número de épocas a 400, permitiendo que el modelo tuviera más iteraciones para converger hacia una solución óptima.

Gracias a estas modificaciones, se logró una reducción significativa del error de entrenamiento y validación, así como una mejora clara en la aproximación de la función real.

## Questions

Pleaser answer the following questions. Include graphs if necessary. Store the graphs in the `outs/exercise_03` folder.

### Which are the differences you found between previous model and this one?
Este modelo permite abordar un problema más complejo que el anterior. Mientras que el modelo previo se utilizaba para aproximar una función cuadrática relativamente sencilla, el nuevo modelo debe aprender una función sinusoidal con oscilaciones más pronunciadas. Para ello, se ha normalizado la entrada, se ha aumentado el número de neuronas de 64 a 128 y se han ajustado las épocas de entrenamiento. Estos cambios permiten que el modelo tenga mayor capacidad para capturar la no linealidad y entrenar de forma más estable, logrando un mejor ajuste y una mejor generalización a nuevos datos.

### Does the model generalizes well to new data?
Sí. La similitud entre las métricas de entrenamiento, validación y test demuestra que el modelo generaliza correctamente. Además, el valor elevado de R² en el conjunto de test confirma que el modelo captura la estructura de la función y no simplemente el ruido de los datos.




