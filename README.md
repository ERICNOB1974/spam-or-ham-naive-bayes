# Clasificación de mensajes spam/ham con técnicas de machine learning (Naive Bayes).
---
## Introducción

El objetivo de este proyecto es desarrollar un clasificador capaz de distinguir entre mensajes de tipo “ham” (deseados) y mensajes de tipo “spam” (no deseados), utilizando como base un modelo de machine learning llamado Naive Bayes. Para el desarrollo se tienen en cuenta únicamente mensajes escritos en español provenientes de un conjunto de datos previamente recopilados. El fin es construir una herramienta sencilla y eficiente que permita entrenar al modelo con ejemplos y luego utilizarlo para predecir a qué categoría pertenece un nuevo mensaje.

## Instalación
Para poder ejecutar el programa es necesario tener instalado Python 3 y algunas librerías adicionales.
1. Clonar el proyecto en tu computadora.
    ```
    git clone https://github.com/ERICNOB1974/spam-or-ham-naive-bayes.git
    cd spam-or-ham-naive-bayes
    ```
2. Instalar Python 3
    ```
    sudo apt install python3 (Ubuntu)
    sudo pacman -S Python (ArchLinux)
    ```
3. Crear un entorno virtual:
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
4. Instalar las dependencias necesarias:
    ```
    pip install -r requeriments.txt
    ```
5. Ejecutar:
    ```
    python antispam_naive_bayes.py
    ```

## Uso
Una vez que el programa se ejecuta correctamente, se desarrollan dos fases:

1. Entrenamiento y resultados del modelo

    El sistema carga el dataset ds_llm.csv, entrena un clasificador Naive Bayes y muestra en pantalla:

    - Accuracy
    - Reporte de métricas (precision, recall, f1-score)
    - Palabras más asociadas a mensajes spam
    - Palabras más asociadas a mensajes ham

    Esta parte se muestra automáticamente sin que el usuario tenga que hacer nada.

2. Interacción con el usuario

- El programa queda esperando que el usuario escriba un mensaje en la consola.

- El usuario puede escribir cualquier texto y el sistema responderá si lo clasifica como spam o ham.

- Para finalizar, se debe escribir la palabra:
    ```
    salir
    ```
El programa mostrará "Saliendo..." y terminará.


## Características
- Entrenamiento con Naive Bayes Multinomial usando scikit-learn.  
- Representación de texto con TF-IDF.  
- Eliminación de stopwords en español con NLTK.  
- Evaluación automática del modelo (accuracy, precision, recall, f1-score).  
- Identificación de palabras más asociadas a spam y a ham.  
- Interfaz por consola para que el usuario ingrese mensajes y obtenga la clasificación.  
