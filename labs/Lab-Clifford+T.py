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
#####el uso de puertas del conjunto Clifford +T ##############
##############################################################


# Se importan las librerías y herramientas
import pennylane as qml
from pennylane import numpy as np

# Definimos el dispositivo cuántico
dev = qml.device("default.qubit", wires=1)

# Secuencia para aproximar Rz(θ) usando Clifford+T
# en función del ángulo de rotación
def rz_clifford_t_approx(theta):
    if np.isclose(theta, np.pi/4, atol=0.05):
        return [qml.T]
    elif np.isclose(theta, -np.pi/4, atol=0.05):
        return [qml.adjoint(qml.T)]
    elif np.isclose(theta, np.pi/8, atol=0.05):
        return [qml.T, qml.adjoint(qml.S), qml.T]
    elif np.isclose(theta, -np.pi/8, atol=0.05):
        return [qml.adjoint(qml.T), qml.S, qml.adjoint(qml.T)]
    else:
        return None

# Circuito aproximado Rx(θ) = H · Rz(θ) · H
@qml.qnode(dev)
def approx_rx(theta):
    qml.Hadamard(wires=0)
    seq = rz_clifford_t_approx(theta)
    if seq is not None:
        for gate in seq:
            gate(wires=0)
    else:
        qml.RZ(theta, wires=0)
    qml.Hadamard(wires=0)
    return qml.state()

# Circuito exacto con Rx(θ)
@qml.qnode(dev)
def exact_rx(theta):
    qml.RX(theta, wires=0)
    return qml.state()

# Comparamos ambos estados para θ = π/8
theta = np.pi / 8
approx_state = approx_rx(theta)
exact_state = exact_rx(theta)
fidelity = np.abs(np.vdot(approx_state, exact_state))**2

print("Estado APROXIMADO:", approx_state)
print("Estado EXACTO    :", exact_state)
print(f"Fidelidad (|⟨ψ_approx|ψ_exact⟩|²): {fidelity:.6f}")
