# Tarea: Perceptrón Multicapa

Ciencia de Datos — Universidad Católica del Norte

## Archivos

| Archivo | Qué es |
|---|---|
| `enunciado.pdf` | El enunciado completo. Léelo primero. |
| `submission.py` | **El único archivo que debes editar.** |
| `util.py` | Datos, cargadores, semilla y escritura del CSV. No se modifica. |
| `experimento.py` | Cierre de la tarea: entrena y arma la tabla de ablación. |
| `grader.py` | Autograder con las pruebas visibles. Corre sin red. |
| `datos/` | Donde queda MNIST tras la primera descarga. No se entrega. |

## Preparación

Basta con Python 3, numpy y PyTorch:

```
pip install -r requirements.txt
```

Todo corre en el procesador: no hace falta tarjeta gráfica ni cuenta en
ningún servicio en línea.

## Antes de empezar

`COMO-TRABAJAR.pdf` trae las instrucciones generales: qué instalar, cómo abrir
una terminal en esta carpeta, qué significa el puntaje del corrector y los
errores más frecuentes. Sirve para todas las tareas del curso.

Un detalle que conviene saber de entrada: en Windows el intérprete se llama
`python` (o `py`), y en Linux y macOS suele ser `python3`. Los comandos de acá
usan `python`: si tu sistema responde «command not found», prueba con
`python3`.

## Cómo trabajar

Cada vez que completes una parte, corre el autograder desde esta carpeta:

```
python grader.py
```

Cuando las cuatro partes estén listas, ejecuta el cierre:

```
python experimento.py
```

Entrena las dos arquitecturas, corre el estudio de ablación y deja la tabla en
`ablacion.csv` (un archivo de trabajo, que no se entrega) y además impresa como
la constante `TABLA_ABLACION`, lista para copiar en la sección «Tus respuestas»
de `submission.py`. Demora menos de un minuto con los datos ya descargados; a
la primera ejecución hay que sumarle la descarga de MNIST (unos 11 MB).

## Sobre los datos

La primera ejecución de `experimento.py` descarga MNIST (unos 11 MB) a
`datos/`. Si el computador no tiene red o no tiene `torchvision`, `util.py`
genera dígitos sintéticos localmente y avisa por consola: la tarea funciona
igual y las conclusiones del estudio de ablación son las mismas.

## Entrega

Un solo archivo, por Campus Virtual: `submission.py`. No se entrega nada más:
ni `ablacion.csv`, ni la carpeta `datos/`, ni los demás archivos de la tarea.

El corrector que viene con la tarea comprueba una parte del puntaje: llegar a
su total completo es condición necesaria y no suficiente, porque la nota la
pone el corrector del curso, que ejecuta esas mismas pruebas más otras sobre lo
mismo. Los 100 puntos son automáticos.

`TABLA_ABLACION` vale 5 de esos 100 puntos y solo sale de correr
`experimento.py`: si la dejas vacía, esos 5 puntos se pierden.
