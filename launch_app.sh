#!/bin/bash
echo "Creando entorno virtual..."
python3 -m venv .venv

echo "Activando entorno virtual..."
source .venv/Scripts/activate

echo "Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Iniciando aplicacion..."
python3 src/main.py
