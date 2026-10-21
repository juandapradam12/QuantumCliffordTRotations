##############################################################
########################## UNIR  #############################
##########    MASTER EN COMPUTACIÓN CUÁNTICA    ##############
##########        ALGORTIMOS CUÁNTICOS        ################
#####  LABORATORIO: COMPUTACIÓN CUÁNTICA DIFERENCIAL   #######
##############################################################
##############################################################

##############################################################
##Este programa permite comparar la salida de la aplicación###
###de las puertas H, S, S+, T y T+ con el de la aplicación####
########de una puerta U3 para los parámetros elegidos#########
##############################################################

# Se importan las librerías y herramientas
import pennylane as qml
from pennylane import numpy as np
import numpy as onp

# Definimos los parámetros del operador U3
theta = np.pi / 2
phi = np.pi / 2
lam = np.pi

# Creamos el estado inicial: (|0⟩ + i|1⟩) / √2
init_state = np.array([1/np.sqrt(2), 1j/np.sqrt(2)])

# Creamos el dispositivo cuántico
dev = qml.device("default.qubit", wires=1)

# Creamos un diccionario de puertas
gates = {
    "U3": lambda: qml.U3(theta, phi, lam, wires=0),
    "H": lambda: qml.Hadamard(wires=0),
    "S": lambda: qml.S(wires=0),
    "S†": lambda: qml.adjoint(qml.S)(wires=0),
    "T": lambda: qml.T(wires=0),
    "T†": lambda: qml.adjoint(qml.T)(wires=0),
}

# Creamos el circuito y el estado final
def get_state(gate_func, init_state):
    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(init_state, wires=0)
        gate_func()
        return qml.state()
    return circuit()

# Obtenemos las coordenadas de Bloch
def bloch_coordinates(state):
    rho = np.outer(state, np.conj(state))
    x = 2 * np.real(rho[0,1])
    y = 2 * np.imag(rho[1,0])
    z = np.real(rho[0,0] - rho[1,1])
    return np.array([x, y, z])

# Calculamos y mostramos los estados y coordenadas
bloch_vectors = {}
states = {}

print("Estados y Vectores de Bloch:")
print("-----------------------------")
for name, gate_func in gates.items():
    state = get_state(gate_func, init_state)
    bloch_vector = bloch_coordinates(state)
    bloch_vectors[name] = bloch_vector
    states[name] = state
    print(f"\n{name}:")
    print(f"  Estado: {state}")
    print(f"  Vector de Bloch: {bloch_vector}")
