# 🏎️ Auto RL: Entrenamiento de Movilidad con PyTorch

Este proyecto utiliza **Deep Reinforcement Learning (DQN)** para entrenar a un vehículo autónomo en un simulador 2D desarrollado con **Pygame**. El objetivo es que el auto aprenda a navegar por una pista sin colisionar, basándose únicamente en la lectura de sus sensores.

## 🚀 Inicio Rápido

Para los usuarios de Windows, he incluido un script de automatización que configura todo el entorno por ti:

1. Ejecuta `launch_app.sh`.
2. El script creará un entorno virtual, instalará las dependencias y lanzará la simulación.

### Instalación Manual
Si prefieres hacerlo manualmente o estás en otro sistema operativo:

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
python src/main.py
```

## 🧠 Arquitectura del Proyecto

- **`src/`**: Carpeta que contiene el código fuente principal.
  - **`model.py`**: Implementación de la red neuronal DQN (Deep Q-Network) y el agente de aprendizaje.
  - **`environment.py`**: Motor de simulación, física del auto y lógica de sensores (Raycasting).
  - **`main.py`**: Bucle principal que integra el entrenamiento con la visualización gráfica.
- **`requirements.txt`**: Lista de dependencias necesarias (PyTorch, Pygame, NumPy).
- **`launch_app.bat`**: Script de arranque de un solo clic para Windows.

## 🛠️ Cómo Funciona la IA

- **Sensores**: El auto tiene 5 sensores de proximidad distribuidos en abanico frontal que detectan la distancia a los obstáculos.
- **Estado**: La entrada de la red neuronal es un vector con las distancias captadas por los sensores y la velocidad actual.
- **Acciones**: El agente puede elegir entre: *Nada, Acelerar, Frenar, Girar a la Izquierda o Girar a la Derecha*.
- **Recompensa**: El sistema premia al auto por la velocidad y la distancia recorrida, penalizando fuertemente las colisiones.

## 📊 Visualización
Durante la ejecución verás:
- **Líneas Rojas**: Los rayos de los sensores en tiempo real.
- **Dashboard**: Información sobre el número de episodio, la velocidad actual y la distancia total recorrida.

---
*Desarrollado con PyTorch y Pygame.*
