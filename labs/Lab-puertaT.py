##############################################################
########################## UNIR  #############################
##########    MASTER EN COMPUTACIÓN CUÁNTICA    ##############
##########        ALGORTIMOS CUÁNTICOS        ################
#####  LABORATORIO: COMPUTACIÓN CUÁNTICA DIFERENCIAL   #######
##############################################################
##############################################################

##############################################################
#####Este programa permite aproximar rotaciones en torno al###
###eje x mediante las rotaciones en torno al eje z mediante###
##################el uso de la puerta T#######################
##############################################################

# Se importan las librerías y herramientas
import pennylane as qml
from pennylane import numpy as np

# Definimos el dispositivo de trabajo
dev = qml.device("default.qubit", wires=1)

# Creamos el circuito aproximado usando H y T
@qml.qnode(dev)
def approx_rx(theta):
# Aplicamos el operador de Hadamard inicial
    qml.Hadamard(wires=0)
# Aproximamos Rz(theta) con T o T† dependiendo del ángulo
    if np.isclose(theta, np.pi/4, atol=0.1):
# Para Rz(pi/4)
        qml.T(wires=0)
    elif np.isclose(theta, -np.pi/4, atol=0.1):
# Paara Rz(-pi/4)
        qml.T(wires=0).inv()
    else:
# Usamos el método exacto para otros ángulos
        qml.RZ(theta, wires=0)
# Aplicamos el operador de Hadamard final
    qml.Hadamard(wires=0)
    return qml.state()

# Creamos el circuito exacto usando Rx
@qml.qnode(dev)
def exact_rx(theta):
    qml.RX(theta, wires=0)
    return qml.state()

# Comparamos los dos resultados
theta = np.pi / 4
approx_state = approx_rx(theta)
exact_state = exact_rx(theta)
# Mostramos los estados directamente
print("Estado APROXIMADO para Rx(pi/4):", approx_state)
print("Estado EXACTO     para Rx(pi/4):", exact_state)
# Comparamos los estados entre sí mediante el producto escalar
fidelity = np.abs(np.vdot(approx_state, exact_state))**2
print(f"\nFidelidad (|⟨ψ_approx|ψ_exact⟩|²): {fidelity:.6f}")
