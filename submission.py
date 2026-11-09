"""Perceptrón multicapa: arquitectura, entrenamiento y ablación.

Este es el **único archivo que debes editar y entregar**. Busca los bloques
marcados con ``# BEGIN_TU_CODIGO`` y reemplázalos por tu implementación. El
primero de esos bloques está en la sección «Tus respuestas», acá arriba, y no
lleva código: lleva la tabla que produce tu propio experimento.

Todo corre en el procesador (CPU): no hay tarjeta gráfica ni servicios en la
nube involucrados. Los datos los prepara ``util.py``.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import util

# ======================================================================
# TUS RESPUESTAS
# ======================================================================
# Esta sección es la hoja de respuestas de la tarea, no código. Se completa al
# final, cuando las cuatro partes ya estén resueltas y hayas corrido
# `python experimento.py`.
#
# TABLA_ABLACION: la tabla completa del estudio de ablación, una tupla por
# configuración y en el mismo orden en que las devuelve
# `configuraciones_ablacion()`. Dentro de cada tupla van los trece valores en
# el orden de `util.COLUMNAS_CSV`:
#
#     (nombre, forma, ancho, profundidad, activacion, tasa, lote, epocas,
#      parametros, perdida_entrenamiento, exactitud_entrenamiento,
#      perdida_validacion, exactitud_validacion)
#
# Los números salen de tu propia corrida: `experimento.py` imprime la tabla y,
# debajo, esta misma constante ya escrita, lista para copiar y pegar acá.
# Cuatro decimales bastan.
# BEGIN_TU_CODIGO
TABLA_ABLACION = [
    ('base', 'cilindrica', 128, 2, 'relu', 0.1000, 64, 2, 118282, 0.3513, 0.8930, 0.3033, 0.9030),
    ('tasa-baja', 'cilindrica', 128, 2, 'relu', 0.0010, 64, 2, 118282, 2.2776, 0.1788, 2.2636, 0.2135),
    ('tasa-alta', 'cilindrica', 128, 2, 'relu', 1.5000, 64, 2, 118282, 2.3122, 0.1095, 2.3045, 0.1140),
    ('lote-pequeño', 'cilindrica', 128, 2, 'relu', 0.1000, 16, 2, 118282, 0.2444, 0.9180, 0.2305, 0.9295),
    ('lote-grande', 'cilindrica', 128, 2, 'relu', 0.1000, 256, 2, 118282, 0.7486, 0.7818, 0.5612, 0.8190),
    ('activacion-sigmoide', 'cilindrica', 128, 2, 'sigmoide', 0.1000, 64, 2, 118282, 2.1348, 0.3202, 1.9403, 0.3795),
    ('activacion-tanh', 'cilindrica', 128, 2, 'tanh', 0.1000, 64, 2, 118282, 0.3533, 0.9040, 0.2899, 0.9145),
    ('profundidad-1', 'cilindrica', 128, 1, 'relu', 0.1000, 64, 2, 101770, 0.3141, 0.9085, 0.2739, 0.9230),
    ('profundidad-4', 'cilindrica', 128, 4, 'relu', 0.1000, 64, 2, 151306, 0.7123, 0.7660, 0.5270, 0.8355),
    ('forma-piramidal', 'piramidal', 128, 2, 'relu', 0.1000, 64, 2, 109386, 0.3655, 0.8912, 0.2978, 0.9065),
]
# END_TU_CODIGO


# ======================================================================
# EL CÓDIGO DE LA TAREA
# ======================================================================
# Nombres de activación admitidos y la clase de torch que les corresponde.
# La entrada del diccionario es lo que aparece en la columna "activacion" del
# CSV de ablaciones.
ACTIVACIONES = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoide": nn.Sigmoid,
}

# Formas de arquitectura que estudiamos en esta tarea.
FORMAS = ("cilindrica", "piramidal")


# ======================================================================
# Problema 1: la arquitectura
# ======================================================================
def tamanos_ocultos(forma: str, ancho: int, profundidad: int) -> list[int]:
    """Devuelve la lista con el número de neuronas de cada capa oculta.

    Args:
        forma: ``"cilindrica"`` (todas las capas ocultas del mismo ancho) o
            ``"piramidal"`` (cada capa tiene la mitad de neuronas que la
            anterior, con un mínimo de una neurona).
        ancho: número de neuronas de la primera capa oculta.
        profundidad: cantidad de capas ocultas.

    Returns:
        Lista de largo ``profundidad`` con enteros positivos. Con
        ``profundidad = 0`` la lista es vacía (el modelo queda lineal).

    Ejemplos:
        ``tamanos_ocultos("cilindrica", 128, 3) -> [128, 128, 128]``
        ``tamanos_ocultos("piramidal", 128, 3)  -> [128, 64, 32]``

    Raises:
        ValueError: si ``forma`` no está en :data:`FORMAS`.
    """
    # BEGIN_TU_CODIGO
    if forma not in FORMAS:
        raise ValueError(f"Forma desconocida: {forma!r}. Debe ser una de {FORMAS}.")
    if forma == "cilindrica":
        return [ancho] * profundidad
    ocultas = []
    actual = max(1, ancho)
    for _ in range(profundidad):
        ocultas.append(actual)
        actual = max(1, actual // 2)
    return ocultas
    # END_TU_CODIGO


def crear_mlp(
    tamano_entrada: int,
    ocultas: list[int],
    tamano_salida: int,
    activacion: str = "relu",
) -> nn.Sequential:
    """Construye el perceptrón multicapa con ``torch.nn``.

    El modelo recibe imágenes de forma ``(n, 1, 28, 28)`` y devuelve
    ``(n, tamano_salida)`` con los **puntajes sin normalizar** (los
    *logits*): la ``softmax`` no va aquí, porque
    ``torch.nn.functional.cross_entropy`` la aplica internamente.

    El orden de las capas es:

    1. ``nn.Flatten()``, que estira cada imagen a un vector de
       ``tamano_entrada`` componentes;
    2. por cada tamaño ``h`` en ``ocultas``: una ``nn.Linear`` hacia ``h``
       seguida de la activación pedida;
    3. una ``nn.Linear`` final hacia ``tamano_salida``, **sin** activación.

    Args:
        tamano_entrada: número de píxeles por imagen (784).
        ocultas: tamaños de las capas ocultas, tal como los devuelve
            :func:`tamanos_ocultos`.
        tamano_salida: número de clases (10).
        activacion: clave de :data:`ACTIVACIONES`.

    Returns:
        Un ``nn.Sequential`` listo para entrenar.

    Raises:
        ValueError: si ``activacion`` no está en :data:`ACTIVACIONES`.
    """
    # BEGIN_TU_CODIGO
    if activacion not in ACTIVACIONES:
        raise ValueError(
            f"Activación desconocida: {activacion!r}. Debe estar en {tuple(ACTIVACIONES)}."
        )
    clase_act = ACTIVACIONES[activacion]
    capas: list[nn.Module] = [nn.Flatten()]
    dim_actual = tamano_entrada
    for h in ocultas:
        capas.append(nn.Linear(dim_actual, h))
        capas.append(clase_act())
        dim_actual = h
    capas.append(nn.Linear(dim_actual, tamano_salida))
    return nn.Sequential(*capas)
    # END_TU_CODIGO


def contar_parametros(modelo: nn.Module) -> int:
    """Cuenta los parámetros entrenables del modelo (pesos y sesgos).

    Es el número que hay que mirar cuando se compara la capacidad de dos
    arquitecturas: una red piramidal y una cilíndrica con el mismo ancho
    inicial no tienen la misma cantidad de parámetros.
    """
    # BEGIN_TU_CODIGO
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    # END_TU_CODIGO


# ======================================================================
# Problema 2: el bucle de entrenamiento y la evaluación
# ======================================================================
def entrenar_una_epoca(
    modelo: nn.Module,
    cargador: torch.utils.data.DataLoader,
    optimizador: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Recorre una vez el conjunto de entrenamiento actualizando los pesos.

    Para cada lote ``(x, y)`` del cargador hay que hacer, en este orden:

    1. pasar ``x`` por el modelo para obtener los puntajes;
    2. calcular la entropía cruzada con ``F.cross_entropy(puntajes, y)``;
    3. borrar los gradientes acumulados con ``optimizador.zero_grad()``;
    4. propagar hacia atrás con ``perdida.backward()``;
    5. dar el paso con ``optimizador.step()``.

    Antes de empezar, deja el modelo en modo entrenamiento con
    ``modelo.train()``.

    Returns:
        ``(perdida_media, exactitud)`` sobre toda la época. La pérdida media
        es el promedio **ponderado por el tamaño de cada lote**, no el
        promedio simple de las pérdidas por lote: el último lote suele ser
        más pequeño. La exactitud es la fracción de ejemplos cuyo puntaje más
        alto corresponde a la clase correcta.
    """
    # BEGIN_TU_CODIGO
    modelo.train()
    suma_perdida = 0.0
    aciertos = 0
    total = 0
    for x, y in cargador:
        puntajes = modelo(x)
        perdida = F.cross_entropy(puntajes, y)
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

        n = x.size(0)
        suma_perdida += perdida.item() * n
        aciertos += int((puntajes.argmax(dim=1) == y).sum().item())
        total += n
    return suma_perdida / total, aciertos / total
    # END_TU_CODIGO


def evaluar(
    modelo: nn.Module,
    cargador: torch.utils.data.DataLoader,
) -> tuple[float, float]:
    """Mide pérdida y exactitud sin modificar el modelo.

    Dos diferencias con :func:`entrenar_una_epoca`, y las dos importan:

    * ``modelo.eval()`` deja al modelo en modo evaluación;
    * todo el recorrido va dentro de ``with torch.no_grad():``, así torch no
      construye el grafo de computación (más rápido y sin gastar memoria).

    Aquí **no** se llama a ``backward()`` ni a ``step()``: medir no es
    aprender.

    Returns:
        ``(perdida_media, exactitud)`` con el mismo promedio ponderado por
        tamaño de lote de :func:`entrenar_una_epoca`.
    """
    # BEGIN_TU_CODIGO
    modelo.eval()
    suma_perdida = 0.0
    aciertos = 0
    total = 0
    with torch.no_grad():
        for x, y in cargador:
            puntajes = modelo(x)
            perdida = F.cross_entropy(puntajes, y)
            n = x.size(0)
            suma_perdida += perdida.item() * n
            aciertos += int((puntajes.argmax(dim=1) == y).sum().item())
            total += n
    return suma_perdida / total, aciertos / total
    # END_TU_CODIGO


# ======================================================================
# Problema 3: entrenar de punta a punta
# ======================================================================
def entrenar_modelo(
    modelo: nn.Module,
    cargador_entrenamiento: torch.utils.data.DataLoader,
    cargador_validacion: torch.utils.data.DataLoader,
    epocas: int = 3,
    tasa: float = 0.1,
) -> list[dict]:
    """Entrena el modelo durante ``epocas`` y devuelve el historial.

    Crea **un solo** optimizador antes del bucle:
    ``torch.optim.SGD(modelo.parameters(), lr=tasa)``. Usamos descenso por
    gradiente estocástico puro, sin momento, para que el efecto de la tasa de
    aprendizaje se vea sin nada que lo amortigüe.

    En cada época: entrena con :func:`entrenar_una_epoca`, mide con
    :func:`evaluar` sobre el cargador de validación, y agrega al historial un
    diccionario con exactamente estas claves::

        {
            "epoca": 1,                       # int, empieza en 1
            "perdida_entrenamiento": float,
            "exactitud_entrenamiento": float,
            "perdida_validacion": float,
            "exactitud_validacion": float,
        }

    Returns:
        Lista de largo ``epocas``, una entrada por época y en orden.
    """
    # BEGIN_TU_CODIGO
    optimizador = torch.optim.SGD(modelo.parameters(), lr=tasa)
    historial = []
    for epoca in range(1, epocas + 1):
        perdida_ent, exactitud_ent = entrenar_una_epoca(
            modelo, cargador_entrenamiento, optimizador
        )
        perdida_val, exactitud_val = evaluar(modelo, cargador_validacion)
        historial.append(
            {
                "epoca": epoca,
                "perdida_entrenamiento": perdida_ent,
                "exactitud_entrenamiento": exactitud_ent,
                "perdida_validacion": perdida_val,
                "exactitud_validacion": exactitud_val,
            }
        )
    return historial
    # END_TU_CODIGO


# ======================================================================
# Problema 4: el estudio de ablación
# ======================================================================
# Configuración de referencia. Todas las demás salen de cambiarle un solo
# valor: así cualquier diferencia observada se le puede atribuir a ese
# cambio y no a otro.
CONFIGURACION_BASE = {
    "nombre": "base",
    "forma": "cilindrica",
    "ancho": 128,
    "profundidad": 2,
    "activacion": "relu",
    "tasa": 0.1,
    "lote": 64,
    "epocas": 2,
}

# Las variaciones del estudio: (nombre de la fila, clave a cambiar, valor).
VARIACIONES_ABLACION = [
    ("tasa-baja", "tasa", 0.001),
    ("tasa-alta", "tasa", 1.5),
    ("lote-pequeño", "lote", 16),
    ("lote-grande", "lote", 256),
    ("activacion-sigmoide", "activacion", "sigmoide"),
    ("activacion-tanh", "activacion", "tanh"),
    ("profundidad-1", "profundidad", 1),
    ("profundidad-4", "profundidad", 4),
    ("forma-piramidal", "forma", "piramidal"),
]


def configuraciones_ablacion() -> list[dict]:
    """Arma la lista de configuraciones del estudio de ablación.

    La primera entrada es una copia de :data:`CONFIGURACION_BASE`. Después
    viene una entrada por cada elemento de :data:`VARIACIONES_ABLACION`, en
    el mismo orden: una copia de la base con ``"nombre"`` reemplazado por el
    nombre de la variación y ``clave`` reemplazada por ``valor``.

    Returns:
        Lista de ``1 + len(VARIACIONES_ABLACION)`` diccionarios
        independientes: modificar uno no debe afectar a los demás ni a
        :data:`CONFIGURACION_BASE`.
    """
    # BEGIN_TU_CODIGO
    configuraciones = [dict(CONFIGURACION_BASE)]
    for nombre, clave, valor in VARIACIONES_ABLACION:
        cfg = dict(CONFIGURACION_BASE)
        cfg["nombre"] = nombre
        cfg[clave] = valor
        configuraciones.append(cfg)
    return configuraciones
    # END_TU_CODIGO


def ejecutar_ablacion(
    configuraciones: list[dict],
    datos: util.Datos,
    ruta_csv: Path | str = "ablacion.csv",
    semilla: int = 0,
) -> list[dict]:
    """Entrena una red por configuración y escribe la tabla de ablación.

    Para cada configuración, en este orden:

    1. ``util.fijar_semilla(semilla)`` — así todas las corridas parten de la
       misma inicialización de los pesos;
    2. ``util.crear_cargadores(datos, configuracion["lote"], semilla=semilla)``
       — la semilla también va aquí, para que el orden del barajado sea el
       mismo en todas las filas. Con las dos cosas fijas, la única diferencia
       entre dos filas es el hiperparámetro que cambió;
    3. :func:`tamanos_ocultos` y :func:`crear_mlp` con lo que dice la
       configuración (``forma``, ``ancho``, ``profundidad``, ``activacion``),
       usando ``util.TAMANO_ENTRADA`` y ``util.TAMANO_SALIDA``;
    4. :func:`entrenar_modelo` con ``epocas`` y ``tasa``;
    5. una fila con **todas** las columnas de ``util.COLUMNAS_CSV``: los ocho
       valores de la configuración, ``parametros`` desde
       :func:`contar_parametros`, y las cuatro métricas de la **última**
       época del historial.

    Al final escribe todas las filas con ``util.escribir_csv``.

    Returns:
        La lista de filas, en el mismo orden que ``configuraciones``.
    """
    # BEGIN_TU_CODIGO
    filas = []
    for configuracion in configuraciones:
        util.fijar_semilla(semilla)
        cargador_ent, cargador_val = util.crear_cargadores(
            datos, configuracion["lote"], semilla=semilla
        )
        ocultas = tamanos_ocultos(
            configuracion["forma"],
            configuracion["ancho"],
            configuracion["profundidad"],
        )
        modelo = crear_mlp(
            util.TAMANO_ENTRADA,
            ocultas,
            util.TAMANO_SALIDA,
            configuracion["activacion"],
        )
        historial = entrenar_modelo(
            modelo,
            cargador_ent,
            cargador_val,
            epocas=configuracion["epocas"],
            tasa=configuracion["tasa"],
        )
        ultima = historial[-1]
        fila = {
            "nombre": configuracion["nombre"],
            "forma": configuracion["forma"],
            "ancho": configuracion["ancho"],
            "profundidad": configuracion["profundidad"],
            "activacion": configuracion["activacion"],
            "tasa": configuracion["tasa"],
            "lote": configuracion["lote"],
            "epocas": configuracion["epocas"],
            "parametros": contar_parametros(modelo),
            "perdida_entrenamiento": ultima["perdida_entrenamiento"],
            "exactitud_entrenamiento": ultima["exactitud_entrenamiento"],
            "perdida_validacion": ultima["perdida_validacion"],
            "exactitud_validacion": ultima["exactitud_validacion"],
        }
        filas.append(fila)
    util.escribir_csv(filas, ruta_csv)
    return filas
    # END_TU_CODIGO
