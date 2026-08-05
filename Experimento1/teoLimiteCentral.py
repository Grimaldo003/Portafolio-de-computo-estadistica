"""
Se creó: 12:06 p. m. 05/08/2026
Clase: TIA

| Por propositos de practicidad se hizo uso de una IA (Grok)
| para generar el código, lo relevantee es observar el
| comportamiento de los promedios generados


Experimento: Teorema del Límite Central (TLC)

Se generan N muestras de tamaño n de diferentes distribuciones
(normal, uniforme, exponencial y gamma).

Se calculan los promedios de cada muestra y se grafican
para observar cómo la distribución de los promedios
se aproxima a una normal (según el TLC).

El usuario elige los valores de n y N.


*  Requerimientos para ejecutar el programa:
Se requiere de la librería matplotlib y la librería numpy
<pip install matplotlib numpy>

¬
"""

from typing import List, Callable
import statistics
import numpy as np
import matplotlib.pyplot as plt


def generar_muestra_normal(n: int, mu: float = 0.0, sigma: float = 1.0) -> List[float]:
    """Genera una muestra de tamaño n con distribución Normal(mu, sigma)."""
    return list(np.random.normal(loc=mu, scale=sigma, size=n))


def generar_muestra_uniforme(n: int, a: float = 0.0, b: float = 1.0) -> List[float]:
    """Genera una muestra de tamaño n con distribución Uniforme(a, b)."""
    return list(np.random.uniform(low=a, high=b, size=n))


def generar_muestra_exponencial(n: int, lambd: float = 1.0) -> List[float]:
    """Genera una muestra de tamaño n con distribución Exponencial(lambda)."""
    return list(np.random.exponential(scale=1.0 / lambd, size=n))


def generar_muestra_gamma(n: int, shape: float = 2.0, scale: float = 1.0) -> List[float]:
    """Genera una muestra de tamaño n con distribución Gamma(shape, scale)."""
    return list(np.random.gamma(shape=shape, scale=scale, size=n))


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


def graficar_histograma_tlc(
    promedios: List[float],
    titulo: str,
    media_teorica: float,
    color: str = "skyblue"
) -> None:
    """
    Dibuja un histograma de los promedios y superpone
    una curva de densidad normal teórica (aproximación por el TLC).
    """
    plt.figure(figsize=(9, 5))

    # Histograma de los promedios obtenidos
    plt.hist(
        promedios,
        bins=40,
        density=True,
        alpha=0.7,
        color=color,
        edgecolor="black",
        label="Promedios obtenidos"
    )

    # Parámetros de la normal aproximada (TLC)
    media_muestral = statistics.mean(promedios)
    desv_muestral = statistics.stdev(promedios) if len(promedios) > 1 else 0.1

    # Curva normal teórica
    x = np.linspace(min(promedios), max(promedios), 300)
    y = (1 / (desv_muestral * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - media_muestral) / desv_muestral) ** 2
    )

    plt.plot(x, y, "r-", linewidth=2.5, label="Curva normal (TLC)")

    # Línea de la media teórica
    plt.axvline(
        media_teorica,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Media teórica = {media_teorica}"
    )

    plt.title(titulo, fontsize=13)
    plt.xlabel("Valor del promedio")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    print("=" * 65)
    print("EXPERIMENTO: Teorema del Límite Central (TLC)")
    print("=" * 65)

    # El usuario elige n y N
    n = pedir_entero_positivo("¿Cuál es el tamaño de cada muestra? (n): ")
    N = pedir_entero_positivo("¿Cuántas veces quieres repetir el experimento? (N): ")

    print(f"\nSe realizarán {N} repeticiones con muestras de tamaño {n}.\n")
    print("Observa cómo los histogramas de los promedios se aproximan a una normal.\n")

    # ---------- 1. Normal(0, 1) ----------
    # Media teórica = 0
    promedios_norm = experimento_promedios(generar_muestra_normal, n, N)
    media_norm = statistics.mean(promedios_norm)
    print("1. Distribución Normal(0, 1)")
    print(f"   Promedio de los promedios: {media_norm:.4f}")
    print(f"   Media teórica = 0.0")
    graficar_histograma_tlc(
        promedios_norm,
        f"TLC - Normal(0,1) | n={n}, N={N}",
        media_teorica=0.0,
        color="salmon"
    )

    # ---------- 2. Uniforme(0, 1) ----------
    # Media teórica = 0.5
    promedios_unif = experimento_promedios(generar_muestra_uniforme, n, N)
    media_unif = statistics.mean(promedios_unif)
    print("\n2. Distribución Uniforme(0, 1)")
    print(f"   Promedio de los promedios: {media_unif:.4f}")
    print(f"   Media teórica = 0.5")
    graficar_histograma_tlc(
        promedios_unif,
        f"TLC - Uniforme(0,1) | n={n}, N={N}",
        media_teorica=0.5,
        color="skyblue"
    )

    # ---------- 3. Exponencial(λ=1) ----------
    # Media teórica = 1.0
    promedios_exp = experimento_promedios(generar_muestra_exponencial, n, N)
    media_exp = statistics.mean(promedios_exp)
    print("\n3. Distribución Exponencial(λ=1)")
    print(f"   Promedio de los promedios: {media_exp:.4f}")
    print(f"   Media teórica = 1.0")
    graficar_histograma_tlc(
        promedios_exp,
        f"TLC - Exponencial(λ=1) | n={n}, N={N}",
        media_teorica=1.0,
        color="lightgreen"
    )

    # ---------- 4. Gamma(shape=2, scale=1) ----------
    # Media teórica = shape * scale = 2.0
    promedios_gamma = experimento_promedios(generar_muestra_gamma, n, N)
    media_gamma = statistics.mean(promedios_gamma)
    print("\n4. Distribución Gamma(shape=2, scale=1)")
    print(f"   Promedio de los promedios: {media_gamma:.4f}")
    print(f"   Media teórica = 2.0")
    graficar_histograma_tlc(
        promedios_gamma,
        f"TLC - Gamma(2,1) | n={n}, N={N}",
        media_teorica=2.0,
        color="violet"
    )

    print("\n" + "=" * 65)
    print("Experimento finalizado.")
    print("Cuanto mayor sea n, más se parecerán los histogramas a una campana normal.")
    print("=" * 65)


if __name__ == "__main__":
    main()