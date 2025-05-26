# Detección y Reconocimiento Facial de Vacas Holstein

Este repositorio contiene la implementación de una arquitectura basada en YOLOv11 y redes siamesas para la detección y reconocimiento facial de vacas Holstein en fincas lecheras colombianas. La solución busca automatizar la identificación individual del ganado, facilitando la trazabilidad y reduciendo la dependencia en métodos manuales.

----------

## Propósito General

Este proyecto tiene como objetivo proporcionar un sistema automatizado robusto que permite la detección precisa del rostro de vacas Holstein utilizando YOLOv11, y su posterior reconocimiento individual mediante una red siamesa entrenada con imágenes faciales. La implementación incluye una aplicación web interactiva desarrollada en Flask para facilitar su uso por parte del usuario final.

## Estructura del Repositorio

El repositorio está organizado en cuatro directorios principales:

### 1. `siamese_network`

-   `entrenamiento_red_siamesa.ipynb`: Contiene el código completo para entrenar la red siamesa, encargada de generar embeddings para la identificación individual.
    
-   Modelo entrenado (`.pt`): El archivo del mejor modelo siames obtenido tras el entrenamiento.
    

### 2. `yolo_model`

-   `cow_face_detection_with_YOLOv11.ipynb`: Código de entrenamiento para el modelo YOLOv11, incluyendo métricas y resultados.
    
-   `dataset_creation_yolo_format.ipynb`: Script utilizado para preparar y extraer imágenes faciales del dataset original en formato compatible con YOLO.
    
-   `parameters_YOLOv5.yaml`: Archivo YAML con parámetros específicos utilizados durante el entrenamiento del modelo YOLOv11.
    
-   `test_model.py`: Ejemplo práctico para probar rápidamente el funcionamiento del modelo YOLOv11 entrenado.
    

### 3. `final_app`

Contiene todos los archivos relacionados con la aplicación Flask diseñada para el usuario final:

-   Scripts Python, plantillas HTML, y otros recursos necesarios para ejecutar la aplicación.
    


### 4. `datasets`

Este directorio incluye los datasets usados para el entrenamiento, organizados en las siguientes subcarpetas:

-   `Vacas`: Contiene las imágenes originales de las vacas de una finca lechera colombiana, junto con las etiquetas específicas asociadas a la cara del animal.
    
-   `Cara`: Incluye el nuevo dataset generado mediante el script `dataset_creation_yolo_format.ipynb`, enfocado exclusivamente en imágenes faciales.
    
-   `Cattely-Cattle-Face-Images-Dataset`: Dataset adicional obtenido del repositorio [Cattely-Cattle-Face-Images-Dataset](https://github.com/aideep1400/Cattely-Cattle-Face-Images-Dataset). Se agradece a los autores originales por proporcionar estos datos para la investigación y desarrollo.

 
## Probar la Aplicación Flask Desplegada

Se puede acceder y probar la aplicación desplegada en la siguiente URL:

[http://34.69.200.14:8080](http://34.69.200.14:8080)

## Instalación y Ejecución Aplicación Flask

A continuación se detallan las instrucciones para configurar el ambiente virtual e iniciar la aplicación Flask:

### Paso 1: Crear y activar ambiente virtual

```
python -m venv venv
source venv/bin/activate  # Linux o Mac
venv\Scripts\activate     # Windows
```

### Paso 2: Instalar dependencias

```
pip install -r requirements.txt
```

### Paso 3: Ejecutar la aplicación Flask

```
flask run
```


----------
