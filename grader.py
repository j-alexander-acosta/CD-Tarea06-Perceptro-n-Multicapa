#!/usr/bin/env python3
"""Autograder de la tarea «Perceptrón Multicapa».

Se ejecuta sin red y sin torchvision, con datos sintéticos pequeños generados
en el momento::

    python grader.py

Imprime el detalle de cada prueba y el subtotal por parte. Este corrector
comprueba una parte de los 100 puntos de la tarea: llegar a su total completo
es condición necesaria y no suficiente. La nota la pone el corrector del
curso, que ejecuta estas mismas pruebas más otras sobre lo mismo.

Nota sobre los tiempos: el autograder entrena redes diminutas sobre unos
pocos cientos de dígitos, así que su ejecución completa demora segundos, no
minutos.
"""
from __future__ import annotations

import contextlib
import math
import os
import signal
import sys

# Nada de .pyc en la carpeta de una entrega: el bytecode de submission
# es la solución de referencia en forma decompilable.
sys.dont_write_bytecode = True
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import submission as sub  # noqa: E402
import util  # noqa: E402


# ----------------------------------------------------------------------
# Infraestructura de pruebas
# ----------------------------------------------------------------------
# (número, título, puntos del problema que corresponden al autograder).
PARTES = [
    (1, "La arquitectura del MLP", 20),
    (2, "El bucle de entrenamiento y la evaluación", 26),
    (3, "Entrenar de punta a punta", 24),
    (4, "El estudio de ablación", 30),
]
PUNTAJE_TAREA = sum(puntos for _, _, puntos in PARTES)

# Ninguna prueba de esta tarea necesita más de unos pocos segundos. El
# límite existe para que un ciclo que no termina se reporte como falla en vez
# de dejar el autograder colgado sin decir nada.
LIMITE_SEGUNDOS = 60

PRUEBAS = []


class TiempoAgotado(Exception):
    """Se levanta cuando una prueba supera su límite de tiempo."""


def _alarma(signum, marco):  # pragma: no cover — manejador de señal
    raise TiempoAgotado()


@contextlib.contextmanager
def limite_de_tiempo(segundos):
    """Corta la prueba si demora más de ``segundos``, donde el SO lo permita.

    Windows no implementa ``signal.SIGALRM``: ahí el temporizador simplemente
    no se arma y las pruebas corren sin límite, en vez de hacer caer el
    autograder completo.
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    anterior = signal.signal(signal.SIGALRM, _alarma)
    signal.setitimer(signal.ITIMER_REAL, segundos)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, anterior)


def prueba(parte, descripcion, puntos, extra=False):
    """Decorador que registra una función de prueba."""

    def decorador(funcion):
        PRUEBAS.append(
            {
                "parte": parte,
                "descripcion": descripcion,
                "puntos": puntos,
                "funcion": funcion,
                "extra": extra,
            }
        )
        return funcion

    return decorador


def cerca(obtenido, esperado, tol=1e-6):
    """Compara dos números o tensores con tolerancia absoluta."""
    return torch.allclose(
        torch.as_tensor(obtenido, dtype=torch.float64),
        torch.as_tensor(esperado, dtype=torch.float64),
        rtol=0.0,
        atol=tol,
    )


def afirmar(condicion, mensaje):
    if not condicion:
        raise AssertionError(mensaje)


# ----------------------------------------------------------------------
# Instrumentos: modelos y optimizadores que registran cómo se los usó
# ----------------------------------------------------------------------
class CapaEspia(nn.Module):
    """Capa lineal que anota, en cada pasada, en qué condiciones se la usó.

    Guarda si el modelo estaba en modo entrenamiento y si los gradientes
    estaban habilitados. Sirve para revisar `modelo.train()`, `modelo.eval()`
    y `torch.no_grad()` sin mirar el código del estudiante.
    """

    def __init__(self, entrada, salida):
        super().__init__()
        self.lineal = nn.Linear(entrada, salida)
        self.modos = []
        self.gradientes = []

    def forward(self, x):
        self.modos.append(self.training)
        self.gradientes.append(torch.is_grad_enabled())
        return self.lineal(x)


class OptimizadorEspia(torch.optim.SGD):
    """SGD que cuenta cuántas veces se llamó a `zero_grad` y a `step`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llamadas_zero_grad = 0
        self.llamadas_step = 0

    def zero_grad(self, *args, **kwargs):
        self.llamadas_zero_grad += 1
        return super().zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.llamadas_step += 1
        return super().step(*args, **kwargs)


def modelo_espia(semilla=0, entrada=784, salida=10):
    """Modelo mínimo (Flatten + CapaEspia) con pesos reproducibles."""
    torch.manual_seed(semilla)
    return nn.Sequential(nn.Flatten(), CapaEspia(entrada, salida))


def espia_de(modelo):
    for modulo in modelo.modules():
        if isinstance(modulo, CapaEspia):
            return modulo
    raise AssertionError("no se encontró la capa instrumentada")


def datos_de_prueba(n=120, semilla=3):
    """Tensores pequeños, ya estandarizados, con la forma de los dígitos."""
    datos = util.datos_sinteticos(n, 8, semilla=semilla)
    return datos.X_ent, datos.y_ent


def metricas_directas(modelo, X, y):
    """Pérdida y exactitud del modelo sobre todo el conjunto, de una vez.

    Es la referencia contra la que se compara `evaluar`: el promedio
    ponderado por lotes tiene que dar exactamente esto.
    """
    with torch.no_grad():
        puntajes = modelo(X)
        perdida = float(F.cross_entropy(puntajes, y).item())
        aciertos = int((puntajes.argmax(dim=1) == y).sum().item())
    return perdida, aciertos / len(y)


def cargador(X, y, lote, barajar=False):
    return DataLoader(TensorDataset(X, y), batch_size=lote, shuffle=barajar)


# ----------------------------------------------------------------------
# Parte 1: la arquitectura
# ----------------------------------------------------------------------
@prueba(1, "tamanos_ocultos: cilíndrica y piramidal", 5)
def p1_tamanos():
    afirmar(
        list(sub.tamanos_ocultos("cilindrica", 128, 3)) == [128, 128, 128],
        "la forma cilíndrica repite el mismo ancho en cada capa oculta",
    )
    afirmar(
        list(sub.tamanos_ocultos("piramidal", 128, 3)) == [128, 64, 32],
        "la forma piramidal parte en `ancho` y va dividiendo por dos",
    )
    afirmar(
        list(sub.tamanos_ocultos("cilindrica", 64, 0)) == [],
        "con profundidad 0 la lista de capas ocultas debe quedar vacía",
    )


@prueba(1, "crear_mlp: orden y tamaños de las capas", 6)
def p1_estructura():
    modelo = sub.crear_mlp(784, [128, 64], 10, "relu")
    afirmar(
        isinstance(modelo, nn.Sequential),
        f"crear_mlp debe devolver un nn.Sequential y devolvió {type(modelo)}",
    )
    capas = list(modelo)
    tipos = [type(c).__name__ for c in capas]
    afirmar(
        tipos == ["Flatten", "Linear", "ReLU", "Linear", "ReLU", "Linear"],
        f"el orden de las capas no calza; obtuve {tipos}",
    )
    dimensiones = [(c.in_features, c.out_features)
                   for c in capas if isinstance(c, nn.Linear)]
    afirmar(
        dimensiones == [(784, 128), (128, 64), (64, 10)],
        f"las dimensiones encadenadas no calzan; obtuve {dimensiones}",
    )


@prueba(1, "contar_parametros", 2)
def p1_parametros():
    modelo = sub.crear_mlp(784, [128, 64], 10, "relu")
    esperado = 784 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10
    obtenido = sub.contar_parametros(modelo)
    afirmar(
        obtenido == esperado,
        f"la red 784-128-64-10 tiene {esperado} parámetros y contaste {obtenido}",
    )


# ----------------------------------------------------------------------
# Parte 2: el bucle de entrenamiento y la evaluación
# ----------------------------------------------------------------------
@prueba(2, "entrenar_una_epoca: métricas y pesos actualizados", 6)
def p2_metricas():
    X, y = datos_de_prueba(120)
    modelo = modelo_espia()
    antes = [p.detach().clone() for p in modelo.parameters()]
    optimizador = torch.optim.SGD(modelo.parameters(), lr=0.05)
    salida = sub.entrenar_una_epoca(modelo, cargador(X, y, 30), optimizador)
    afirmar(
        isinstance(salida, tuple) and len(salida) == 2,
        f"debe devolver la tupla (perdida, exactitud) y devolvió {salida!r}",
    )
    perdida, exactitud = float(salida[0]), float(salida[1])
    afirmar(math.isfinite(perdida) and perdida > 0.0,
            f"la pérdida media debe ser un número positivo y fue {perdida}")
    afirmar(0.0 <= exactitud <= 1.0,
            f"la exactitud es una fracción entre 0 y 1 y fue {exactitud}")
    despues = [p.detach().clone() for p in modelo.parameters()]
    afirmar(
        any(not torch.equal(a, b) for a, b in zip(antes, despues)),
        "después de una época los pesos deberían haber cambiado: "
        "¿se llamó a optimizador.step()?",
    )
    # Con más épocas la pérdida tiene que bajar.
    primera = perdida
    for _ in range(4):
        ultima, _ = sub.entrenar_una_epoca(modelo, cargador(X, y, 30), optimizador)
    afirmar(
        ultima < primera,
        f"tras cinco épocas la pérdida subió ({primera:.4f} -> {ultima:.4f}); "
        "revisa el orden de zero_grad / backward / step",
    )


@prueba(2, "evaluar: pérdida y exactitud correctas", 5)
def p2_evaluar_valores():
    X, y = datos_de_prueba(120, semilla=11)
    modelo = modelo_espia(semilla=5)
    perdida_esperada, exactitud_esperada = metricas_directas(modelo, X, y)
    # 120 ejemplos en lotes de 50 dejan un último lote de 20. Con lotes
    # iguales el promedio simple y el ponderado coinciden, así que una
    # implementación sin ponderar pasaría esta prueba y el error recién
    # aparecería en la corrección docente.
    perdida, exactitud = sub.evaluar(modelo, cargador(X, y, 50))
    afirmar(
        cerca(perdida, perdida_esperada, tol=1e-5),
        f"la pérdida media debería ser {perdida_esperada:.6f} y dio {perdida}",
    )
    afirmar(
        cerca(exactitud, exactitud_esperada, tol=1e-9),
        f"la exactitud debería ser {exactitud_esperada:.6f} y dio {exactitud}",
    )


# ----------------------------------------------------------------------
# Parte 3: entrenar de punta a punta
# ----------------------------------------------------------------------
CLAVES_HISTORIAL = {
    "epoca",
    "perdida_entrenamiento",
    "exactitud_entrenamiento",
    "perdida_validacion",
    "exactitud_validacion",
}


@prueba(3, "entrenar_modelo: largo del historial y claves de cada época", 6)
def p3_historial():
    X, y = datos_de_prueba(120)
    Xv, yv = datos_de_prueba(60, semilla=21)
    modelo = sub.crear_mlp(784, [16], 10, "relu")
    historial = sub.entrenar_modelo(
        modelo, cargador(X, y, 30), cargador(Xv, yv, 30), epocas=3, tasa=0.1
    )
    afirmar(
        isinstance(historial, list) and len(historial) == 3,
        f"con epocas=3 el historial debe tener 3 entradas y tuvo "
        f"{len(historial) if isinstance(historial, list) else type(historial)}",
    )
    for i, fila in enumerate(historial):
        afirmar(
            isinstance(fila, dict) and CLAVES_HISTORIAL <= set(fila),
            f"a la entrada {i} le faltan claves: "
            f"{sorted(CLAVES_HISTORIAL - set(fila))}",
        )
        afirmar(
            fila["epoca"] == i + 1,
            f"la clave 'epoca' numera desde 1: en la posición {i} vale "
            f"{fila['epoca']}",
        )


@prueba(3, "entrenar_modelo: usa SGD con la tasa indicada", 6)
def p3_optimizador():
    X, y = datos_de_prueba(40)
    Xv, yv = datos_de_prueba(20, semilla=21)
    tasa = 0.25

    modelo = modelo_espia(semilla=7)
    antes = [p.detach().clone() for p in modelo.parameters()]
    # Un solo lote y una sola época: el paso de SGD es exactamente
    # w <- w - tasa * gradiente, así que se puede verificar a mano.
    esperados = []
    referencia = modelo_espia(semilla=7)
    F.cross_entropy(referencia(X), y).backward()
    for p in referencia.parameters():
        esperados.append(p.detach() - tasa * p.grad.detach())

    sub.entrenar_modelo(
        modelo, cargador(X, y, 40), cargador(Xv, yv, 20), epocas=1, tasa=tasa
    )
    for antes_i, obtenido, esperado in zip(antes, modelo.parameters(), esperados):
        afirmar(
            not torch.equal(antes_i, obtenido.detach()),
            "los pesos no cambiaron: ¿se creó el optimizador?",
        )
        afirmar(
            cerca(obtenido.detach(), esperado, tol=1e-5),
            "tras un paso los pesos no coinciden con w - tasa * gradiente; "
            "el optimizador debe ser torch.optim.SGD(..., lr=tasa) sin momento",
        )


@prueba(3, "entrenar_modelo: la red efectivamente aprende", 8)
def p3_aprende():
    datos = util.datos_sinteticos(600, 200, semilla=3)
    Xv, yv = datos.X_val, datos.y_val
    util.fijar_semilla(0)
    modelo = sub.crear_mlp(784, [64, 64], 10, "relu")
    historial = sub.entrenar_modelo(
        modelo,
        cargador(datos.X_ent, datos.y_ent, 32, barajar=True),
        cargador(Xv, yv, 64),
        epocas=4, tasa=0.1,
    )
    primera, ultima = historial[0], historial[-1]
    afirmar(
        ultima["perdida_entrenamiento"] < primera["perdida_entrenamiento"],
        "la pérdida de entrenamiento no bajó entre la primera y la última "
        f"época ({primera['perdida_entrenamiento']:.4f} -> "
        f"{ultima['perdida_entrenamiento']:.4f})",
    )
    afirmar(
        ultima["exactitud_validacion"] > 0.5,
        "tras cuatro épocas la exactitud de validación debería superar 0,5 "
        f"y quedó en {ultima['exactitud_validacion']:.4f}",
    )
    perdida_final, exactitud_final = sub.evaluar(modelo, cargador(Xv, yv, 64))
    afirmar(
        cerca(perdida_final, ultima["perdida_validacion"], tol=1e-5)
        and cerca(exactitud_final, ultima["exactitud_validacion"], tol=1e-9),
        "las métricas de validación de la última época no coinciden con "
        "evaluar() sobre el modelo entrenado: ¿se midió en el momento "
        "equivocado?",
    )


# ----------------------------------------------------------------------
# Parte 4: el estudio de ablación
# ----------------------------------------------------------------------
def configuracion_chica(nombre, **cambios):
    base = {
        "nombre": nombre,
        "forma": "cilindrica",
        "ancho": 16,
        "profundidad": 1,
        "activacion": "relu",
        "tasa": 0.1,
        "lote": 32,
        "epocas": 1,
    }
    base.update(cambios)
    return base


@prueba(4, "configuraciones_ablacion: base más una variación por fila", 6)
def p4_configuraciones():
    configuraciones = sub.configuraciones_ablacion()
    esperadas = 1 + len(sub.VARIACIONES_ABLACION)
    afirmar(
        isinstance(configuraciones, list) and len(configuraciones) == esperadas,
        f"se esperaban {esperadas} configuraciones (la base y una por "
        f"variación) y hubo {len(configuraciones)}",
    )
    afirmar(
        configuraciones[0] == sub.CONFIGURACION_BASE,
        "la primera configuración debe ser la base, sin cambios",
    )
    afirmar(
        configuraciones[0] is not sub.CONFIGURACION_BASE,
        "cada configuración debe ser un diccionario nuevo: si devuelves la "
        "base misma, modificarla después contamina todo el estudio",
    )
    for (nombre, clave, valor), obtenida in zip(
        sub.VARIACIONES_ABLACION, configuraciones[1:]
    ):
        afirmar(
            obtenida.get("nombre") == nombre,
            f"la fila de la variación «{nombre}» quedó con nombre "
            f"{obtenida.get('nombre')!r}",
        )
        afirmar(
            obtenida.get(clave) == valor,
            f"en «{nombre}» la clave {clave!r} debería valer {valor!r} y vale "
            f"{obtenida.get(clave)!r}",
        )
        distintas = {
            k for k in sub.CONFIGURACION_BASE
            if k != "nombre" and obtenida.get(k) != sub.CONFIGURACION_BASE[k]
        }
        afirmar(
            distintas == {clave},
            f"«{nombre}» debe diferir de la base solo en {clave!r} y difiere "
            f"en {sorted(distintas)}",
        )
    nombres = [c["nombre"] for c in configuraciones]
    afirmar(len(set(nombres)) == len(nombres),
            f"los nombres de las filas deben ser distintos: {nombres}")


@prueba(4, "ejecutar_ablacion: una fila completa por configuración", 4)
def p4_filas():
    datos = util.datos_sinteticos(120, 60, semilla=5)
    configuraciones = [
        configuracion_chica("pequeña"),
        configuracion_chica("mas-ancha", ancho=24),
    ]
    with tempfile.TemporaryDirectory() as carpeta:
        filas = sub.ejecutar_ablacion(
            configuraciones, datos, Path(carpeta) / "tabla.csv"
        )
    afirmar(
        isinstance(filas, list) and len(filas) == 2,
        f"debe devolver una fila por configuración (2) y devolvió "
        f"{len(filas) if isinstance(filas, list) else type(filas)}",
    )
    for configuracion, fila in zip(configuraciones, filas):
        faltantes = [c for c in util.COLUMNAS_CSV if c not in fila]
        afirmar(not faltantes, f"a la fila le faltan columnas: {faltantes}")
        for clave in ("nombre", "forma", "ancho", "profundidad", "activacion",
                      "tasa", "lote", "epocas"):
            afirmar(
                fila[clave] == configuracion[clave],
                f"la columna {clave!r} debe copiar el valor de la "
                f"configuración ({configuracion[clave]!r}) y trae "
                f"{fila[clave]!r}",
            )
        afirmar(
            0.0 <= float(fila["exactitud_validacion"]) <= 1.0
            and 0.0 <= float(fila["exactitud_entrenamiento"]) <= 1.0,
            "las exactitudes deben ser fracciones entre 0 y 1",
        )
        afirmar(
            math.isfinite(float(fila["perdida_validacion"])),
            "la pérdida de validación no es un número finito",
        )


@prueba(4, "ejecutar_ablacion: el CSV queda escrito y bien formado", 5)
def p4_csv():
    import csv

    datos = util.datos_sinteticos(120, 60, semilla=5)
    configuraciones = [
        configuracion_chica("pequeña"),
        configuracion_chica("mas-profunda", profundidad=2),
    ]
    with tempfile.TemporaryDirectory() as carpeta:
        ruta = Path(carpeta) / "tabla.csv"
        sub.ejecutar_ablacion(configuraciones, datos, ruta)
        afirmar(ruta.exists(), f"no se escribió el archivo {ruta.name}")
        with ruta.open(encoding="utf-8") as f:
            leidas = list(csv.DictReader(f))
            f.seek(0)
            encabezado = f.readline().strip().split(",")
    afirmar(
        encabezado == util.COLUMNAS_CSV,
        f"el encabezado debe ser util.COLUMNAS_CSV y fue {encabezado}",
    )
    afirmar(len(leidas) == 2, f"el CSV debe tener 2 filas y tuvo {len(leidas)}")
    afirmar(
        [f["nombre"] for f in leidas] == ["pequeña", "mas-profunda"],
        "las filas del CSV deben ir en el mismo orden que las configuraciones",
    )


# ----------------------------------------------------------------------
# Ejecución
# ----------------------------------------------------------------------
def ejecutar_una(caso):
    """Corre una prueba y devuelve (puntos obtenidos, detalle del error)."""
    try:
        with limite_de_tiempo(LIMITE_SEGUNDOS):
            caso["funcion"]()
    except TiempoAgotado:
        return 0, (
            f"tiempo agotado: la prueba superó {LIMITE_SEGUNDOS} s, revisa si "
            "hay un ciclo que no termina"
        )
    except NotImplementedError:
        return 0, "sin implementar todavía"
    except AssertionError as exc:
        return 0, str(exc)
    except Exception as exc:  # noqa: BLE001 — el grader nunca debe caerse
        return 0, f"{type(exc).__name__}: {exc}"
    return caso["puntos"], ""


def main():
    torch.manual_seed(0)
    # Todo va con flush: si una prueba se cuelga, el estudiante igual alcanza
    # a ver hasta dónde llegó el autograder.
    print("=" * 72, flush=True)
    print(" Autograder — Tarea: Perceptrón Multicapa", flush=True)
    print("=" * 72, flush=True)

    obtenido_total = 0
    maximo_total = 0
    hay_extra = any(c["extra"] for c in PRUEBAS)

    for numero, titulo, puntos_declarados in PARTES:
        casos = [c for c in PRUEBAS if c["parte"] == numero]
        if not casos:
            continue
        print(f"\nParte {numero}: {titulo}", flush=True)
        obtenido_parte = sum_parte = 0
        for caso in casos:
            puntos, detalle = ejecutar_una(caso)
            obtenido_parte += puntos
            sum_parte += caso["puntos"]
            marca = "OK   " if puntos == caso["puntos"] else "FALLA"
            sufijo = "  (adicional)" if caso["extra"] else ""
            print(f"  [{marca}] {caso['descripcion']}{sufijo}"
                  f"  ->  {puntos}/{caso['puntos']}", flush=True)
            if detalle:
                print(f"           {detalle}", flush=True)
        if sum_parte == puntos_declarados:
            print(f"  Subtotal parte {numero}: {obtenido_parte}/{sum_parte}",
                  flush=True)
        else:
            # Sin las pruebas adicionales el subtotal no llega al puntaje que
            # el enunciado le pone al problema: hay que decirlo, o los números
            # no cuadran con el PDF.
            print(f"  Subtotal parte {numero}: {obtenido_parte}/{sum_parte} en "
                  f"las pruebas visibles (el Problema {numero} vale "
                  f"{puntos_declarados} puntos)", flush=True)
        obtenido_total += obtenido_parte
        maximo_total += sum_parte

    print("\n" + "=" * 72, flush=True)
    if hay_extra:
        print(f" PUNTAJE DE LA TAREA: {obtenido_total}/{maximo_total}",
              flush=True)
    else:
        print(f" PUNTAJE EN LAS PRUEBAS VISIBLES: {obtenido_total}/{maximo_total}",
              flush=True)
        print(f" La tarea vale {PUNTAJE_TAREA} puntos: los "
              f"{PUNTAJE_TAREA - maximo_total} restantes se obtienen en las "
              "pruebas adicionales", flush=True)
        print(" que el corrector del curso ejecuta sobre tu entrega. Todo el "
              "puntaje es automático.", flush=True)
    print("=" * 72, flush=True)
    return 0 if obtenido_total == maximo_total else 1


if __name__ == "__main__":
    sys.exit(main())
