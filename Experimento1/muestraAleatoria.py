"""
Experimento 1:
Repetir N veces una muestra aleatoria de tamaño n=50,
calcular el promedio de cada muestra y luego el promedio
(y opcionalmente la varianza) de esos promedios.

Objetivo: comprobar que el valor esperado del promedio
se aproxima a la media teórica de la distribución.

El usuario elige el valor de N.
"""

from random import randint, random, gauss, expovariate
from typing import List, Callable
import statistics


def generar_muestra_uniforme_discreta(n: int) -> List[int]:
    """
    Genera una muestra de tamaño n con distribución uniforme discreta
    en el intervalo [1, 101].
    """
    return [randint(1, 101) for _ in range(n)]


def generar_muestra_uniforme_continua(n: int) -> List[float]:
    """
    Genera una muestra de tamaño n con distribución uniforme continua
    en el intervalo [0, 1].
    """
    return [random() for _ in range(n)]


def generar_muestra_normal(n: int, mu: float = 0.0, sigma: float = 1.0) -> List[float]:
    """
    Genera una muestra de tamaño n con distribución normal(mu, sigma).
    """
    return [gauss(mu, sigma) for _ in range(n)]


def generar_muestra_exponencial(n: int, lambd: float = 1.0) -> List[float]:
    """
    Genera una muestra de tamaño n con distribución exponencial(lambda).
    """
    return [expovariate(lambd) for _ in range(n)]


def promedio_de_muestra(muestra: List[float]) -> float:
    """Calcula el promedio aritmético de una lista de números."""
    return sum(muestra) / len(muestra)


def experimento_promedios(
    generador_muestra: Callable[[int], List[float]],
    n: int,
    N: int
) -> List[float]:
    """
    Repite N veces la generación de una muestra de tamaño n,
    calcula el promedio de cada muestra y devuelve la lista
    de esos N promedios.
    """
    promedios: List[float] = []
    for _ in range(N):
        muestra = generador_muestra(n)
        prom = promedio_de_muestra(muestra)
        promedios.append(prom)
    return promedios


def pedir_entero_positivo(mensaje: str) -> int:
    """
    Solicita al usuario un número entero positivo de forma segura.
    Repite la pregunta hasta que se introduzca un valor válido.
    """
    while True:
        try:
            valor = int(input(mensaje))
            if valor > 0:
                return valor
            else:
                print("Error: el número debe ser mayor que 0. Intenta de nuevo.")
        except ValueError:
            print("Error: debes introducir un número entero válido. Intenta de nuevo.")


def main() -> None:
    n = 50  # tamaño de cada muestra (fijo por ahora)

    print("=" * 60)
    print("EXPERIMENTO: Valor esperado del promedio")
    print(f"Tamaño de muestra n = {n}")
    print("=" * 60)

    # El usuario elige N
    N = pedir_entero_positivo("¿Cuántas veces quieres repetir el experimento? (N): ")

    print(f"\nSe realizarán {N} repeticiones.\n")

    # ---------- Uniforme discreta ----------
    promedios_ud = experimento_promedios(generar_muestra_uniforme_discreta, n, N)
    media_de_promedios_ud = statistics.mean(promedios_ud)
    print("1. Uniforme discreta [1, 101]")
    print(f"   Promedios obtenidos: {promedios_ud}")
    print(f"   Promedio de los promedios: {media_de_promedios_ud:.4f}")
    print(f"   Media teórica ≈ 51.0")

    # ---------- Uniforme continua [0,1] ----------
    promedios_uc = experimento_promedios(generar_muestra_uniforme_continua, n, N)
    media_de_promedios_uc = statistics.mean(promedios_uc)
    print("\n2. Uniforme continua [0, 1]")
    print(f"   Promedios obtenidos: {[round(p, 4) for p in promedios_uc]}")
    print(f"   Promedio de los promedios: {media_de_promedios_uc:.4f}")
    print(f"   Media teórica = 0.5")

    # ---------- Normal(0,1) ----------
    promedios_norm = experimento_promedios(generar_muestra_normal, n, N)
    media_de_promedios_norm = statistics.mean(promedios_norm)
    print("\n3. Normal(0, 1)")
    print(f"   Promedios obtenidos: {[round(p, 4) for p in promedios_norm]}")
    print(f"   Promedio de los promedios: {media_de_promedios_norm:.4f}")
    print(f"   Media teórica = 0.0")

    # ---------- Exponencial(λ=1) ----------
    promedios_exp = experimento_promedios(generar_muestra_exponencial, n, N)
    media_de_promedios_exp = statistics.mean(promedios_exp)
    print("\n4. Exponencial(λ=1)")
    print(f"   Promedios obtenidos: {[round(p, 4) for p in promedios_exp]}")
    print(f"   Promedio de los promedios: {media_de_promedios_exp:.4f}")
    print(f"   Media teórica = 1.0")


if __name__ == "__main__":
    main()