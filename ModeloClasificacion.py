import pandas as pd
import numpy as np

df = pd.read_csv('M2_ML Uresti\Proyecto\StudentP_clean.csv')

# Definir la función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Definir la función de costo
def cost_function(X, y, theta):
    h = sigmoid(np.dot(X, theta))
    J = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    return J

# Definir la función de gradiente
def gradient(X, y, theta):
    h = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, (h - y)) / len(y)
    return grad

# Inicializar los parámetros
n_features = df.shape[1] - 1
theta = np.zeros((n_features + 1, 1))

# Añadir una columna de unos a X para el término de intersección
X = np.hstack((np.ones((df.shape[0], 1)), df.drop('Passed', axis=1)))
y = df['Passed'].values.reshape(-1, 1)

# Entrenar el modelo
alpha = 0.05
num_iter = 1500
for i in range(num_iter):
    grad = gradient(X, y, theta)
    theta -= alpha * grad

# Hacer predicciones
y_pred = np.round(sigmoid(np.dot(X, theta)))


# Convertir y y y_pred a arrays unidimensionales para facilitar el cálculo
y = y.flatten()
y_pred = y_pred.flatten()

# Calcular la matriz de confusión
TP = np.sum((y_pred == 1) & (y == 1))  # Verdaderos positivos
TN = np.sum((y_pred == 0) & (y == 0))  # Verdaderos negativos
FP = np.sum((y_pred == 1) & (y == 0))  # Falsos positivos
FN = np.sum((y_pred == 0) & (y == 1))  # Falsos negativos

# Calcular precisión
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# Calcular recall
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calcular F1 Score
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Calcular precisión general
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Imprimir resultados
print("Precisión del modelo:", accuracy)
print("Precisión (precision):", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Matriz de Confusión:")
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

# Evaluar el modelo
print("Accuracy:", accuracy)