# Carpeta de datos

Aquí queda MNIST la primera vez que se ejecuta `experimento.py`: `util.py`
lo descarga a `datos/MNIST/raw/` (unos 11 MB) y lo reutiliza en las
ejecuciones siguientes.

Si el computador no tiene red o no tiene `torchvision`, la tarea sigue
funcionando: `util.digitos_sinteticos` genera localmente dígitos de siete
segmentos con inclinación, desplazamiento, difuminado y ruido, con la misma
forma (28x28, diez clases). El programa avisa por consola cuál de las dos
fuentes está usando.

Esta carpeta no se entrega.
