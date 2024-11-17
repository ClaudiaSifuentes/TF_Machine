# Modelos y Resultados

## MLP (Perceptrón Multicapa)

|           | Precisión | Recall | F1-score | Soporte |
|-----------|-----------|--------|----------|---------|
| Clase 0   | 0.71      | 0.83   | 0.76     | 1781    |
| Clase 1   | 0.80      | 0.66   | 0.72     | 1786    |
| **Exactitud** |         |        | 0.74     | 3567    |
| **Promedio macro** | 0.75  | 0.74   | 0.74     | 3567    |
| **Promedio ponderado** | 0.75  | 0.74   | 0.74     | 3567    |

### Interpretación

- El modelo MLP está funcionando bien en general, con una precisión y recall balanceados para ambas clases.
- La clase 0 tiene mayor precisión y recall, lo que indica que el modelo es mejor para identificar esta clase.
- La clase 1 tiene un recall ligeramente más bajo, pero su precisión sigue siendo fuerte. El recall más bajo sugiere que el modelo podría estar perdiendo algunos verdaderos positivos de esta clase, pero evita los falsos positivos.
- Los F1-scores de ambas clases son bastante altos (por encima de 0.7), lo que indica un buen balance entre precisión y recall.
- La exactitud del modelo (0.74) refleja un rendimiento general razonable.

## GRU (Unidad Recurrente Gated)

|           | Precisión | Recall | F1-score | Soporte |
|-----------|-----------|--------|----------|---------|
| Clase 0   | 0.51      | 0.31   | 0.39     | 1778    |
| Clase 1   | 0.51      | 0.70   | 0.59     | 1784    |
| **Exactitud** |         |        | 0.51     | 3562    |
| **Promedio macro** | 0.51  | 0.51   | 0.49     | 3562    |
| **Promedio ponderado** | 0.51  | 0.51   | 0.49     | 3562    |

### Interpretación

- GRU tiene una precisión relativamente baja para ambas clases (alrededor de 0.51), lo que significa que tiene dificultades para identificar correctamente los verdaderos positivos para ambas clases.
- El recall es notablemente mejor para la clase 1 (0.70), lo que significa que el modelo identifica más de los verdaderos positivos de la clase 1, pero pierde muchas instancias de la clase 0 (recall 0.31).
- Los F1-scores reflejan este desequilibrio, con la clase 1 desempeñándose mejor que la clase 0.
- La exactitud es mucho más baja, con 0.51, lo que se debe principalmente al bajo recall de la clase 0 y a un rendimiento desequilibrado entre las dos clases.

## Bidirectional LSTM

|           | Precisión | Recall | F1-score | Soporte |
|-----------|-----------|--------|----------|---------|
| Clase 0   | 0.58      | 0.01   | 0.03     | 1778    |
| Clase 1   | 0.50      | 0.99   | 0.67     | 1784    |
| **Exactitud** |         |        | 0.50     | 3562    |
| **Promedio macro** | 0.54  | 0.50   | 0.35     | 3562    |
| **Promedio ponderado** | 0.54  | 0.50   | 0.35     | 3562    |

### Interpretación

- El modelo Bidirectional LSTM está altamente desequilibrado, con un recall casi perfecto para la clase 1 (0.99) pero un recall extremadamente bajo para la clase 0 (0.01). Esto indica que el modelo está prediciendo mayormente la clase 1 y no identifica en absoluto la clase 0.
- La precisión es relativamente baja para ambas clases (0.58 para clase 0 y 0.50 para clase 1), lo que sugiere que el modelo no es muy preciso al identificar los verdaderos positivos.
- El F1-score para la clase 1 (0.67) refleja el alto recall pero baja precisión, mientras que para la clase 0 (0.03), el F1-score es muy bajo, indicando un mal rendimiento del modelo para predecir la clase 0.
- La exactitud de 0.50 está cerca de la adivinación aleatoria, probablemente porque el modelo está sesgado hacia la predicción de la clase 1.

## Resumen de Resultados

- **MLP:** Es el modelo de mejor rendimiento entre los tres, con precisión, recall y F1-scores balanceados. Logra una exactitud decente y clasifica razonablemente bien ambas clases.
- **GRU:** Este modelo tiene problemas con la precisión, mostrando un mejor recall para la clase 1 pero un recall muy bajo para la clase 0. Su exactitud es significativamente más baja que la del MLP.
- **Bidirectional LSTM:** Este modelo sufre de un desequilibrio extremo entre clases, con un recall casi perfecto para la clase 1 pero un recall cercano a cero para la clase 0. Su rendimiento en precisión y F1-score es deficiente, lo que se traduce en una baja exactitud.

## Cómo Mejorar

### Para MLP:
El modelo ya está funcionando bien, pero podrías intentar:
- Ajuste de hiperparámetros (por ejemplo, capas, nodos o funciones de activación).
- Ingeniería de características adicional, como agregar más indicadores técnicos o usar características de retardo (datos de días anteriores).

### Para GRU:
- Técnicas de balanceo de clases, como usar pesos de clase o sobremuestreo (por ejemplo, SMOTE) para mejorar el recall de la clase 0.
- Ajustes en la arquitectura del modelo: aumentar la complejidad del modelo o experimentar con otras arquitecturas como LSTM o modelos basados en atención.

### Para Bidirectional LSTM:
- Balanceo de clases: Similar al GRU, ajustar la distribución de clases utilizando pesos de clase o sobremuestreo.
- Preprocesamiento de datos: Considerar agregar más características diversas o ajustar los hiperparámetros para evitar que el modelo se sobreajuste a la clase 1.
