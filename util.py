"""Utilidades de apoyo de la tarea. No hay que modificar este archivo.

Aquí vive todo lo que no es parte del ejercicio: la carga de los datos, la
construcción de los cargadores por lotes, la semilla y la escritura del CSV
de ablaciones.

Sobre los datos
---------------
La tarea clasifica dígitos manuscritos de 28x28 píxeles. Los datos se
obtienen en tres etapas, de la mejor a la peor:

1. **MNIST con torchvision**, descargado una sola vez a la carpeta
   ``datos/`` de la tarea (unos 11 MB). Es la opción por defecto.
2. **MNIST leído a mano** desde ``datos/MNIST/raw/``, en el formato IDX
   original, si torchvision no está instalado o falla. Si falta algún archivo
   y hay red, se descarga con la biblioteca estándar.
3. **Dígitos sintéticos** generados localmente por :func:`digitos_sinteticos`
   si no hay red ni datos en disco. Son dígitos de siete segmentos con
   inclinación, desplazamiento, difuminado y ruido aleatorios: un problema de
   diez clases con la misma forma (28x28, etiquetas 0--9) y una dificultad
   parecida, suficiente para que el estudio de ablación muestre las mismas
   tendencias.

En los tres casos el resultado es idéntico en forma y tipo, así que el resto
de la tarea no necesita saber de dónde vinieron los datos.
"""
from __future__ import annotations

import csv
import gzip
import random
import struct
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Cada imagen es de 28x28 y hay diez dígitos.
TAMANO_ENTRADA = 28 * 28
TAMANO_SALIDA = 10

CARPETA = Path(__file__).resolve().parent
CARPETA_DATOS = CARPETA / "datos"

# Espejo público de MNIST, el mismo que usa torchvision.
_URL_MNIST = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_ARCHIVOS_MNIST = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

# Orden de las columnas del CSV de ablaciones. Fijarlo aquí evita que cada
# entrega salga con un orden distinto.
COLUMNAS_CSV = [
    "nombre",
    "forma",
    "ancho",
    "profundidad",
    "activacion",
    "tasa",
    "lote",
    "epocas",
    "parametros",
    "perdida_entrenamiento",
    "exactitud_entrenamiento",
    "perdida_validacion",
    "exactitud_validacion",
]


class Datos(NamedTuple):
    """Conjunto de datos listo para entrenar.

    Atributos:
        X_ent: imágenes de entrenamiento, ``(n, 1, 28, 28)`` de tipo float32.
        y_ent: etiquetas de entrenamiento, ``(n,)`` de tipo int64.
        X_val: imágenes de validación.
        y_val: etiquetas de validación.
        origen: ``"mnist"`` o ``"sintetico"``, solo informativo.
    """

    X_ent: torch.Tensor
    y_ent: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    origen: str


# ----------------------------------------------------------------------
# Semilla
# ----------------------------------------------------------------------
def fijar_semilla(semilla: int = 0) -> None:
    """Fija la semilla de ``random``, numpy y torch.

    Con esto dos ejecuciones del mismo experimento entregan los mismos
    números. Sin eso, una tabla de ablación no permite comparar nada: las
    diferencias entre filas podrían venir solo de la inicialización.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


# ----------------------------------------------------------------------
# Dígitos sintéticos (plan de respaldo, sin red)
# ----------------------------------------------------------------------
# Cada dígito se dibuja como una pantalla de siete segmentos sobre un lienzo
# de 28x28. Los rangos son (fila_inicial, fila_final, columna_inicial,
# columna_final), con el índice final excluido.
_SEGMENTOS = {
    "a": (4, 7, 9, 20),    # barra superior
    "g": (12, 15, 9, 20),  # barra del medio
    "d": (21, 24, 9, 20),  # barra inferior
    "f": (4, 15, 9, 12),   # vertical superior izquierda
    "b": (4, 15, 17, 20),  # vertical superior derecha
    "e": (13, 24, 9, 12),  # vertical inferior izquierda
    "c": (13, 24, 17, 20),  # vertical inferior derecha
}

_DIGITOS = {
    0: "abcdef",
    1: "bc",
    2: "abged",
    3: "abgcd",
    4: "fgbc",
    5: "afgcd",
    6: "afgecd",
    7: "abc",
    8: "abcdefg",
    9: "abcdfg",
}


def _mascaras_segmentos() -> dict[str, np.ndarray]:
    """Devuelve, para cada segmento, su máscara 28x28 de ceros y unos."""
    mascaras = {}
    for nombre, (f0, f1, c0, c1) in _SEGMENTOS.items():
        m = np.zeros((28, 28), dtype=np.float32)
        m[f0:f1, c0:c1] = 1.0
        mascaras[nombre] = m
    return mascaras


def _difuminar(imagen: np.ndarray) -> np.ndarray:
    """Promedio 3x3: suaviza los bordes duros de los segmentos."""
    p = np.pad(imagen, 1)
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    ) / 9.0


def digitos_sinteticos(n: int, semilla: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Genera ``n`` dígitos sintéticos de 28x28 con sus etiquetas.

    Devuelve ``(X, y)`` con ``X`` de forma ``(n, 1, 28, 28)`` en float32 y
    valores en ``[0, 1]``, e ``y`` de forma ``(n,)`` en int64 con valores
    ``0..9``.

    Cada muestra parte de la plantilla de siete segmentos del dígito y recibe
    intensidad distinta por segmento, inclinación, desplazamiento, difuminado
    opcional y ruido gaussiano. Nada de esto usa la red: la tarea funciona
    igual en un computador aislado.
    """
    rng = np.random.default_rng(semilla)
    mascaras = _mascaras_segmentos()
    filas = np.arange(28)

    X = np.zeros((n, 1, 28, 28), dtype=np.float32)
    y = rng.integers(0, 10, size=n)

    for i in range(n):
        imagen = np.zeros((28, 28), dtype=np.float32)
        for segmento in _DIGITOS[int(y[i])]:
            imagen += float(rng.uniform(0.55, 1.0)) * mascaras[segmento]
        imagen = np.clip(imagen, 0.0, 1.0)

        # Inclinación: cada fila se corre horizontalmente según su altura,
        # igual que la letra manuscrita inclinada.
        pendiente = rng.uniform(-0.32, 0.32)
        desplazamiento = np.round(pendiente * (filas - 13.5)).astype(int)
        columnas = (np.arange(28)[None, :] - desplazamiento[:, None]) % 28
        imagen = imagen[filas[:, None], columnas]

        # Traslación entera dentro del lienzo.
        imagen = np.roll(imagen, (int(rng.integers(-2, 3)), int(rng.integers(-2, 3))),
                         axis=(0, 1))

        if rng.random() < 0.5:
            imagen = _difuminar(imagen)

        imagen = imagen + rng.normal(0.0, 0.16, size=(28, 28))
        X[i, 0] = np.clip(imagen, 0.0, 1.0)

    return X, y.astype(np.int64)


# ----------------------------------------------------------------------
# MNIST
# ----------------------------------------------------------------------
def _leer_idx(ruta: Path) -> np.ndarray:
    """Lee un archivo IDX comprimido (el formato original de MNIST)."""
    with gzip.open(ruta, "rb") as f:
        contenido = f.read()
    magico, cantidad = struct.unpack(">II", contenido[:8])
    tipo = (magico >> 8) & 0xFF
    dimensiones = magico & 0xFF
    if tipo != 0x08:
        raise ValueError(f"{ruta.name}: solo se admiten datos de tipo byte")
    forma = [cantidad]
    for k in range(1, dimensiones):
        forma.append(struct.unpack(">I", contenido[4 + 4 * k: 8 + 4 * k])[0])
    inicio = 4 + 4 * dimensiones
    return np.frombuffer(contenido, dtype=np.uint8, offset=inicio).reshape(forma)


def _mnist_con_torchvision(directorio: Path, permitir_descarga: bool):
    """Intenta cargar MNIST con torchvision. Devuelve None si no se puede."""
    try:
        from torchvision import datasets  # noqa: PLC0415 — importación opcional
    except Exception:  # noqa: BLE001 — torchvision ausente o mal instalado
        return None
    try:
        ent = datasets.MNIST(root=str(directorio), train=True,
                             download=permitir_descarga)
        val = datasets.MNIST(root=str(directorio), train=False,
                             download=permitir_descarga)
    except Exception:  # noqa: BLE001 — sin red o descarga incompleta
        return None
    return (
        ent.data.numpy(), ent.targets.numpy(),
        val.data.numpy(), val.targets.numpy(),
    )


def _mnist_a_mano(directorio: Path, permitir_descarga: bool):
    """Lee los archivos IDX de MNIST; los descarga si faltan y hay red."""
    crudos = directorio / "MNIST" / "raw"
    faltantes = [n for n in _ARCHIVOS_MNIST if not (crudos / n).exists()]
    if faltantes:
        if not permitir_descarga:
            return None
        crudos.mkdir(parents=True, exist_ok=True)
        for nombre in faltantes:
            try:
                with urllib.request.urlopen(_URL_MNIST + nombre, timeout=30) as r:
                    (crudos / nombre).write_bytes(r.read())
            except Exception:  # noqa: BLE001 — sin red o espejo caído
                return None
    try:
        return tuple(_leer_idx(crudos / nombre) for nombre in _ARCHIVOS_MNIST)
    except Exception:  # noqa: BLE001 — archivo truncado o corrupto
        return None


# ----------------------------------------------------------------------
# Carga y preparación
# ----------------------------------------------------------------------
def _estandarizar(X_ent: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centra y escala ambos conjuntos con la media y la desviación del
    conjunto de **entrenamiento**.

    Usar estadísticas del conjunto de validación sería filtrar información
    que el modelo no debería ver.
    """
    media = float(X_ent.mean())
    desviacion = float(X_ent.std()) or 1.0
    return ((X_ent - media) / desviacion, (X_val - media) / desviacion)


def cargar_datos(
    n_entrenamiento: int = 6000,
    n_validacion: int = 2000,
    directorio: Path | str = CARPETA_DATOS,
    semilla: int = 0,
    permitir_descarga: bool = True,
    silencioso: bool = False,
) -> Datos:
    """Carga un subconjunto de dígitos listo para entrenar en pocos minutos.

    Se usa un subconjunto (y no los 60.000 dígitos de MNIST) a propósito: con
    6.000 imágenes de entrenamiento y 2.000 de validación, cada época demora
    un par de segundos en el procesador de un computador de laboratorio, y el
    estudio de ablación completo cabe en menos de un minuto.
    """
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    crudos = (
        _mnist_con_torchvision(directorio, permitir_descarga)
        or _mnist_a_mano(directorio, permitir_descarga)
    )

    if crudos is not None:
        origen = "mnist"
        X_ent, y_ent, X_val, y_val = (np.asarray(a) for a in crudos)
        X_ent = X_ent.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
        X_val = X_val.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
        y_ent = y_ent.astype(np.int64)
        y_val = y_val.astype(np.int64)
    else:
        origen = "sintetico"
        if not silencioso:
            print("AVISO: no se pudo obtener MNIST (sin torchvision o sin red).")
            print("       Se usan dígitos sintéticos generados localmente.")
        X_ent, y_ent = digitos_sinteticos(n_entrenamiento, semilla=semilla)
        X_val, y_val = digitos_sinteticos(n_validacion, semilla=semilla + 1000)

    # Submuestreo reproducible.
    rng = np.random.default_rng(semilla)
    ind_ent = rng.permutation(len(X_ent))[:n_entrenamiento]
    ind_val = rng.permutation(len(X_val))[:n_validacion]
    X_ent, y_ent = X_ent[ind_ent], y_ent[ind_ent]
    X_val, y_val = X_val[ind_val], y_val[ind_val]

    X_ent, X_val = _estandarizar(X_ent, X_val)

    return Datos(
        X_ent=torch.from_numpy(np.ascontiguousarray(X_ent, dtype=np.float32)),
        y_ent=torch.from_numpy(np.ascontiguousarray(y_ent)),
        X_val=torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32)),
        y_val=torch.from_numpy(np.ascontiguousarray(y_val)),
        origen=origen,
    )


def datos_sinteticos(
    n_entrenamiento: int = 600,
    n_validacion: int = 200,
    semilla: int = 0,
) -> Datos:
    """Versión pequeña y siempre disponible de :func:`cargar_datos`.

    No toca el disco ni la red: la usa el autograder.
    """
    X_ent, y_ent = digitos_sinteticos(n_entrenamiento, semilla=semilla)
    X_val, y_val = digitos_sinteticos(n_validacion, semilla=semilla + 1000)
    X_ent, X_val = _estandarizar(X_ent, X_val)
    return Datos(
        X_ent=torch.from_numpy(X_ent),
        y_ent=torch.from_numpy(y_ent),
        X_val=torch.from_numpy(X_val),
        y_val=torch.from_numpy(y_val),
        origen="sintetico",
    )


def crear_cargadores(
    datos: Datos, lote: int, lote_evaluacion: int = 512, semilla: int = 0
) -> tuple[DataLoader, DataLoader]:
    """Arma los ``DataLoader`` de entrenamiento y validación.

    El de entrenamiento baraja en cada época (eso es lo que convierte al
    descenso por gradiente en *estocástico*); el de validación no baraja,
    porque solo se recorre para medir.

    El barajado usa un generador **propio**, sembrado con ``semilla``, y no el
    generador global de torch. La diferencia importa para el estudio de
    ablación: si el barajado saliera del generador global, su punto de partida
    dependería de cuántos números al azar consumió la construcción del modelo,
    que es distinta para cada arquitectura. Dos filas que solo querían cambiar
    la profundidad terminarían además viendo los lotes en otro orden, y esa
    segunda diferencia arruinaría la comparación.

    Args:
        datos: el conjunto ya preparado.
        lote: tamaño de lote del cargador de entrenamiento.
        lote_evaluacion: tamaño de lote del de validación; solo afecta la
            velocidad de la medición, no su resultado.
        semilla: semilla del generador que decide el orden del barajado.
    """
    entrenamiento = TensorDataset(datos.X_ent, datos.y_ent)
    validacion = TensorDataset(datos.X_val, datos.y_val)
    generador = torch.Generator()
    generador.manual_seed(int(semilla))
    return (
        DataLoader(entrenamiento, batch_size=int(lote), shuffle=True,
                   generator=generador),
        DataLoader(validacion, batch_size=int(lote_evaluacion), shuffle=False),
    )


# ----------------------------------------------------------------------
# Salida
# ----------------------------------------------------------------------
def escribir_csv(filas: list[dict], ruta: Path | str) -> None:
    """Escribe las filas de la ablación en ``ruta`` con las columnas de
    :data:`COLUMNAS_CSV`.

    Los números con decimales se guardan con cuatro cifras, que es toda la
    precisión que tiene sentido comparar entre corridas.
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        escritor.writeheader()
        for fila in filas:
            limpia = {}
            for columna in COLUMNAS_CSV:
                valor = fila.get(columna, "")
                limpia[columna] = f"{valor:.4f}" if isinstance(valor, float) else valor
            escritor.writerow(limpia)


def imprimir_tabla(filas: list[dict]) -> None:
    """Muestra la tabla de ablación en la consola, con las mismas columnas."""
    anchos = {c: max(len(c), *(len(_texto(f.get(c, ""))) for f in filas))
              for c in COLUMNAS_CSV} if filas else {c: len(c) for c in COLUMNAS_CSV}
    encabezado = "  ".join(c.ljust(anchos[c]) for c in COLUMNAS_CSV)
    print(encabezado)
    print("-" * len(encabezado))
    for fila in filas:
        print("  ".join(_texto(fila.get(c, "")).ljust(anchos[c]) for c in COLUMNAS_CSV))


def _texto(valor) -> str:
    return f"{valor:.4f}" if isinstance(valor, float) else str(valor)
