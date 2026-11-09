"""Cierre de la tarea: entrena las dos arquitecturas y corre la ablación.

Se ejecuta con::

    python experimento.py

Hace tres cosas:

1. carga los dígitos (MNIST si está disponible, sintéticos si no) e informa
   de dónde salieron;
2. entrena la arquitectura **cilíndrica** y la **piramidal** con la misma
   semilla y muestra sus curvas época a época;
3. corre el estudio de ablación completo, deja la tabla en ``ablacion.csv``
   (un archivo de trabajo, que no se entrega) y la imprime además como la
   constante ``TABLA_ABLACION`` que va copiada en ``submission.py``.

Todo corre en el procesador y demora menos de un minuto con los datos ya
descargados; a la primera ejecución hay que sumarle la descarga de MNIST
(unos 11 MB). No hay que modificar este archivo.
"""
from __future__ import annotations

import sys

# Nada de .pyc en la carpeta de una entrega: el bytecode de submission
# es la solución de referencia en forma decompilable.
sys.dont_write_bytecode = True

import math
import time
from pathlib import Path

import util
from submission import (
    configuraciones_ablacion,
    contar_parametros,
    crear_mlp,
    ejecutar_ablacion,
    entrenar_modelo,
    tamanos_ocultos,
)

CARPETA = Path(__file__).resolve().parent
RUTA_CSV = CARPETA / "ablacion.csv"

EPOCAS_DEMOSTRACION = 3
TASA_DEMOSTRACION = 0.1
LOTE_DEMOSTRACION = 64
SEMILLA = 1


def demostrar(nombre: str, forma: str, datos: util.Datos) -> None:
    """Entrena una arquitectura y muestra su historial época a época."""
    util.fijar_semilla(SEMILLA)
    cargador_ent, cargador_val = util.crear_cargadores(
        datos, LOTE_DEMOSTRACION, semilla=SEMILLA
    )
    ocultas = tamanos_ocultos(forma, ancho=128, profundidad=2)
    modelo = crear_mlp(util.TAMANO_ENTRADA, ocultas, util.TAMANO_SALIDA, "relu")

    print(f"\n{nombre}  (capas ocultas: {ocultas})")
    print(f"  parámetros entrenables: {contar_parametros(modelo):,}")
    inicio = time.time()
    historial = entrenar_modelo(
        modelo,
        cargador_ent,
        cargador_val,
        epocas=EPOCAS_DEMOSTRACION,
        tasa=TASA_DEMOSTRACION,
    )
    for fila in historial:
        print(
            f"  época {fila['epoca']}:"
            f"  entrenamiento pérdida={fila['perdida_entrenamiento']:.4f}"
            f" exactitud={fila['exactitud_entrenamiento']:.4f}"
            f"  |  validación pérdida={fila['perdida_validacion']:.4f}"
            f" exactitud={fila['exactitud_validacion']:.4f}"
        )
    print(f"  tiempo: {time.time() - inicio:.1f} s")


def imprimir_constante(filas: list[dict]) -> None:
    """Muestra la tabla escrita como la constante que va en submission.py.

    Así la entrega se arma copiando y pegando, sin transcribir números a mano.
    """
    print("TABLA_ABLACION = [")
    for fila in filas:
        celdas = []
        for columna in util.COLUMNAS_CSV:
            valor = fila[columna]
            if isinstance(valor, float):
                celdas.append(
                    f"{valor:.4f}" if math.isfinite(valor)
                    else f'float("{valor}")'
                )
            else:
                celdas.append(repr(valor))
        print("    (" + ", ".join(celdas) + "),")
    print("]")


def main() -> int:
    print("=" * 78)
    print(" Perceptrón multicapa")
    print("=" * 78)

    datos = util.cargar_datos()
    print(f"  origen de los datos: {datos.origen}")
    print(f"  entrenamiento: {tuple(datos.X_ent.shape)}"
          f"   validación: {tuple(datos.X_val.shape)}")

    print("\n" + "-" * 78)
    print(" Parte A: dos arquitecturas, mismos datos y misma semilla")
    print("-" * 78)
    demostrar("Arquitectura cilíndrica", "cilindrica", datos)
    demostrar("Arquitectura piramidal", "piramidal", datos)

    print("\n" + "-" * 78)
    print(" Parte B: estudio de ablación")
    print("-" * 78)
    configuraciones = configuraciones_ablacion()
    print(f"  {len(configuraciones)} configuraciones; una sola diferencia con"
          " la base en cada una.")
    inicio = time.time()
    filas = ejecutar_ablacion(configuraciones, datos, RUTA_CSV, semilla=SEMILLA)
    print(f"  tiempo total: {time.time() - inicio:.1f} s\n")
    util.imprimir_tabla(filas)
    print(f"\n  Tabla escrita también en {RUTA_CSV.name}, que es un archivo de"
          " trabajo y no se entrega.")
    print("  Lo que se entrega es esta misma tabla como constante: copia el"
          " bloque de abajo")
    print("  en la sección «Tus respuestas» de submission.py, reemplazando"
          " TABLA_ABLACION.\n")
    imprimir_constante(filas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
