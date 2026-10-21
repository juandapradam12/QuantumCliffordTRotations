##############################################################
########################## UNIR  #############################
##########    MASTER EN COMPUTACIÓN CUÁNTICA    ##############
##########        ALGORTIMOS CUÁNTICOS        ################
#####  LABORATORIO: COMPUTACIÓN CUÁNTICA DIFERENCIAL   #######
##############################################################
##############################################################

##############################################################
#####Este programa permite comparar visualmente la salida#####
###de las puertas H, S, S+, T y T+ con el de la aplicación####
########de una puerta U3 para los parámetros elegidos#########
##############################################################

# Se importan las librerías y herramientas
import pennylane as qml
from pennylane import numpy as np
import numpy as onp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definimos los parámetros de la compuerta U3
theta = np.pi / 2
phi = np.pi / 2
lam = np.pi

# Creamos el estado inicial: (|0⟩ + i|1⟩) / √2
init_state = np.array([1/np.sqrt(2), 1j/np.sqrt(2)])

# Creamos el dispositivo cuántico
dev = qml.device("default.qubit", wires=1)

# Creamos un diccionario de puertas
comparison_gates = {
    "H": lambda: qml.Hadamard(wires=0),
    "S": lambda: qml.S(wires=0),
    "S†": lambda: qml.adjoint(qml.S)(wires=0),
    "T": lambda: qml.T(wires=0),
    "T†": lambda: qml.adjoint(qml.T)(wires=0),
}

# Creamos la puerta U3
u3_gate = lambda: qml.U3(theta, phi, lam, wires=0)

# Creamos el circuito y el estado final
def get_state(gate_func, init_state):
    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(init_state, wires=0)
        gate_func()
        return qml.state()
    return circuit()

# Creamos la función para obtener coordenadas de Bloch
def bloch_coordinates(state):
    rho = np.outer(state, np.conj(state))
    x = 2 * np.real(rho[0,1])
    y = 2 * np.imag(rho[1,0])
    z = np.real(rho[0,0] - rho[1,1])
    return np.array([x, y, z])

# Obtenemos el estado U3
u3_state = get_state(u3_gate, init_state)
u3_vector = bloch_coordinates(u3_state)

# Creamos la figura con 1 fila y 5 columnas
fig = plt.figure(figsize=(24, 6))

for idx, (name, gate_func) in enumerate(comparison_gates.items()):
    other_state = get_state(gate_func, init_state)
    other_vector = bloch_coordinates(other_state)

    ax = fig.add_subplot(1, 5, idx+1, projection='3d')

    # Esfera de Bloch
    u, v = onp.mgrid[0:2*onp.pi:100j, 0:onp.pi:100j]
    x = onp.cos(u) * onp.sin(v)
    y = onp.sin(u) * onp.sin(v)
    z = onp.cos(v)
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.1, edgecolor='gray')

    # Ejes X, Y, Z
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.05)

    # Vectores cuánticos
    ax.quiver(0, 0, 0, *u3_vector, color='orange', linewidth=2.5, label='U3')
    ax.scatter(*u3_vector, color='orange', s=80)

    ax.quiver(0, 0, 0, *other_vector, color='purple', linewidth=2.5, label=name)
    ax.scatter(*other_vector, color='purple', s=80)

    # Formato del gráfico
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'U3 vs {name}', fontsize=12)
    ax.legend()

# Ajustes finales
plt.subplots_adjust(left=0.03, right=0.97, bottom=0.1, top=0.85, wspace=0.3)
plt.suptitle("Comparación de Estados Cuánticos: U3 vs otras puertas", fontsize=18)
plt.show()
