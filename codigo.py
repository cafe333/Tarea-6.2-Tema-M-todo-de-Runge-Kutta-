"Ejercicio 1"

import numpy as np
import matplotlib.pyplot as plt

# Definición de la ecuación diferencial: dT/dx = -0.25(T - 25)
def f(x, T):
    return -0.25 * (T - 25)

# Método de Runge-Kutta de cuarto orden con impresión de valores
def runge_kutta_4(f, x0, y0, x_end, h):
    x_vals = [x0]
    y_vals = [y0]

    x = x0
    y = y0

    # Encabezado de la tabla
    print(f"{'x':>10} {'T_aproximado':>15}")
    print(f"{x:10.4f} {y:15.6f}")

    while x < x_end:
        k1 = f(x, y)
        k2 = f(x + h/2, y + h/2 * k1)
        k3 = f(x + h/2, y + h/2 * k2)
        k4 = f(x + h, y + h * k3)

        y += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h

        x_vals.append(x)
        y_vals.append(y)

        print(f"{x:10.4f} {y:15.6f}")

    return x_vals, y_vals

# Parámetros iniciales
x0 = 0
T0 = 100  # Temperatura inicial
x_end = 2
h = 0.1

# Llamada al método de Runge-Kutta
x_vals, T_vals = runge_kutta_4(f, x0, T0, x_end, h)

# Solución exacta
T_exacta = [25 + 75 * np.exp(-0.25 * x) for x in x_vals]

# Graficar la solución
plt.figure(figsize=(8,5))
plt.plot(x_vals, T_vals, 'bo-', label="Solución RK4")
plt.plot(x_vals, T_exacta, 'r-', label="Solución Exacta")
plt.xlabel("x")
plt.ylabel("T (°C)")
plt.title("Transferencia de calor en un tubo")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("temperatura_runge_kutta.png")
plt.show()

"Ejercicio 2"

import numpy as np
import matplotlib.pyplot as plt

# Definimos la ecuación diferencial: dq/dt = (V - q/C) / R
def f(t, q, V=10, R=1000, C=0.001):
    return (V - q/C) / R

# Método de Runge-Kutta de 4to orden con impresión de valores
def runge_kutta_4(f, t0, q0, t_end, h):
    t_vals = [t0]
    q_vals = [q0]

    t = t0
    q = q0

    # Encabezado de la tabla
    print(f"{'t':>10} {'q_aproximado':>15}")
    print(f"{t:10.4f} {q:15.6f}")

    while t < t_end:
        k1 = f(t, q)
        k2 = f(t + h/2, q + h/2 * k1)
        k3 = f(t + h/2, q + h/2 * k2)
        k4 = f(t + h, q + h * k3)

        q += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h

        t_vals.append(t)
        q_vals.append(q)

        print(f"{t:10.4f} {q:15.6f}")

    return np.array(t_vals), np.array(q_vals)

# Parámetros iniciales
t0 = 0
q0 = 0  # Carga inicial
t_end = 1
h = 0.05

# Llamada al método de Runge-Kutta
t_vals, q_vals = runge_kutta_4(f, t0, q0, t_end, h)

# Graficar la solución
plt.figure(figsize=(8,5))
plt.plot(t_vals, q_vals, 'bo-', label="Solución RK4")
plt.plot(t_vals, q_exacta, 'r-', label="Solución Exacta")
plt.xlabel("t")
plt.ylabel("q (Carga)")
plt.title("Carga de un capacitor en un circuito RC")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("carga_capacitor_rk4.png")
plt.show()

"Ejercicio 3"

import numpy as np
import matplotlib.pyplot as plt

# Definimos el sistema de ecuaciones
def f(t, y):
    y1, y2 = y
    dy1_dt = y2
    dy2_dt = -2*y2 - 5*y1
    return np.array([dy1_dt, dy2_dt], dtype=np.float64)  # Asegurar tipo float64

# Implementamos el método de Runge-Kutta de 4to orden
def runge_kutta_4(f, t0, y0, t_end, h):
    t_vals = [t0]
    y_vals = [y0.astype(np.float64)]  # Asegurar tipo float64 en la inicialización

    t = t0
    y = y0.astype(np.float64)  # Convertir y a float64

    print(f"{'t':>10} {'y1':>15} {'y2':>15}")
    print(f"{t:10.4f} {y[0]:15.6f} {y[1]:15.6f}")

    while t < t_end:
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)

        y += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h

        t_vals.append(t)
        y_vals.append(y.copy())

        print(f"{t:10.4f} {y[0]:15.6f} {y[1]:15.6f}")

    return np.array(t_vals), np.array(y_vals)

# Condiciones iniciales
t0 = 0
y0 = np.array([1.0, 0.0], dtype=np.float64)  # Convertir a float64
t_end = 5
h = 0.1

# Resolvemos con Runge-Kutta
t_vals, y_vals = runge_kutta_4(f, t0, y0, t_end, h)

# Graficamos la solución
plt.figure(figsize=(8,5))
plt.plot(t_vals, y_vals[:,0], 'bo-', label="Posición (y1)")
plt.plot(t_vals, y_vals[:,1], 'ro-', label="Velocidad (y2)")
plt.xlabel("t")
plt.ylabel("y1 (Posición)_y2 (Velocidad)")
plt.title("Movimiento de un resorte amortiguado")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("resorte_amortiguado.png")
plt.show()
