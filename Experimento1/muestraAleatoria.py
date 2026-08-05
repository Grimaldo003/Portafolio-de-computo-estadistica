"""
Se creó: 11:06 a. m. 04/08/2026
Clase: TIA

Fecha de cambios:
Se corrigió 11:36 a. m. 05/08/2026

-- Por propositos de practicidad se uso de una IA (Grok)
para generar el código, lo relevantee es observar el
comportamiento de los promedios generados --

Experimento 1:
Repetir N veces una muestra aleatoria de tamaño n,
calcular el promedio de cada muestra y luego el promedio
de esos promedios.

Objetivo: comprobar que el valor esperado del promedio
se aproxima a la media teórica de la distribución.

El usuario elige n y N.
Se muestran histogramas de los promedios.

Requerimientso para ejecutar el programa:
Se requiere de la librería matplotlib y la librería numpy
<pip install matplotlib numpy>

"""

from random import randint, random, gauss, expovariate
from typing import List, Callable
import statistics
import matplotlib.pyplot as plt
import numpy as np  # solo para la curva normal teórica


def generar_muestra_uniforme_discreta(n: int) -> List[int]:
    """Genera una muestra de tamaño n con distribución uniforme discreta [1, 101]."""
    return [randint(1, 101) for _ in range(n)]


def generar_muestra_uniforme_continua(n: int) -> List[float]:
    """Genera una muestra de tamaño n con distribución uniforme continua [0, 1]."""
    return [random() for _ in range(n)]


def generar_muestra_normal(n: int, mu: float = 0.0, sigma: float = 1.0) -> List[float]:
    """Genera una muestra de tamaño n con distribución normal(mu, sigma)."""
    return [gauss(mu, sigma) for _ in range(n)]


def generar_muestra_exponencial(n: int, lambd: float = 1.0) -> List[float]:
    """Genera una muestra de tamaño n con distribución exponencial(lambda)."""
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


def graficar_histograma(
    promedios: List[float],
    titulo: str,
    media_teorica: float,
    color: str = "skyblue"
) -> None:
    """
    Dibuja un histograma de los promedios y superpone
    una curva de densidad normal teórica (aproximación por TLC).
    """
    plt.figure(figsize=(8, 5))
    
    # Histograma de los promedios obtenidos
    plt.hist(promedios, bins=30, density=True, alpha=0.7, color=color, edgecolor="black", label="Promedios obtenidos")
    
    # Curva normal teórica (aproximación por el Teorema del Límite Central)
    media_muestral = statistics.mean(promedios)
    desv_muestral = statistics.stdev(promedios) if len(promedios) > 1 else 0.1
    
    x = np.linspace(min(promedios), max(promedios), 200)
    y = (1 / (desv_muestral * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - media_muestral) / desv_muestral) ** 2)
    
    plt.plot(x, y, "r-", linewidth=2, label="Curva normal aproximada")
    
    # Línea vertical de la media teórica
    plt.axvline(media_teorica, color="green", linestyle="--", linewidth=2, label=f"Media teórica = {media_teorica}")
    
    plt.title(titulo)
    plt.xlabel("Valor del promedio")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    print("=" * 60)
    print("EXPERIMENTO: Valor esperado del promedio")
    print("=" * 60)

    # El usuario elige n y N
    n = pedir_entero_positivo("¿Cuál es el tamaño de cada muestra? (n): ")
    N = pedir_entero_positivo("¿Cuántas veces quieres repetir el experimento? (N): ")

    print(f"\nSe realizarán {N} repeticiones con muestras de tamaño {n}.\n")

    # ---------- 1. Uniforme discreta ----------
    promedios_ud = experimento_promedios(generar_muestra_uniforme_discreta, n, N)
    media_ud = statistics.mean(promedios_ud)
    print("1. Uniforme discreta [1, 101]")
    print(f"   Promedio de los promedios: {media_ud:.4f}")
    print(f"   Media teórica ≈ 51.0")
    graficar_histograma(promedios_ud, "Uniforme discreta [1, 101] - Promedios", 51.0, "skyblue")

    # ---------- 2. Uniforme continua ----------
    promedios_uc = experimento_promedios(generar_muestra_uniforme_continua, n, N)
    media_uc = statistics.mean(promedios_uc)
    print("\n2. Uniforme continua [0, 1]")
    print(f"   Promedio de los promedios: {media_uc:.4f}")
    print(f"   Media teórica = 0.5")
    graficar_histograma(promedios_uc, "Uniforme continua [0, 1] - Promedios", 0.5, "lightgreen")

    # ---------- 3. Normal(0,1) ----------
    promedios_norm = experimento_promedios(generar_muestra_normal, n, N)
    media_norm = statistics.mean(promedios_norm)
    print("\n3. Normal(0, 1)")
    print(f"   Promedio de los promedios: {media_norm:.4f}")
    print(f"   Media teórica = 0.0")
    graficar_histograma(promedios_norm, "Normal(0, 1) - Promedios", 0.0, "salmon")

    # ---------- 4. Exponencial(λ=1) ----------
    promedios_exp = experimento_promedios(generar_muestra_exponencial, n, N)
    media_exp = statistics.mean(promedios_exp)
    print("\n4. Exponencial(λ=1)")
    print(f"   Promedio de los promedios: {media_exp:.4f}")
    print(f"   Media teórica = 1.0")
    graficar_histograma(promedios_exp, "Exponencial(λ=1) - Promedios", 1.0, "violet")


if __name__ == "__main__":
    main()