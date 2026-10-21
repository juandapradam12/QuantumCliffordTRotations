##############################################################
########################## UNIR  #############################
##########    MASTER EN COMPUTACIÓN CUÁNTICA    ##############
##########        ALGORTIMOS CUÁNTICOS        ################
#####  LABORATORIO: COMPUTACIÓN CUÁNTICA DIFERENCIAL   #######
##############################################################
##############################################################

##############################################################
#####Este programa permite normalizar un estado a la unidad###
##############################################################

# Se importan las librerías y herramientas
import pennylane as qml
from pennylane import numpy as np

# Creamos el dispositivo con 2 qubits
dev = qml.device("default.qubit", wires=2)


# Creamos el estado deseado |00>+|10>+|11>,
# que se expresa como [1, 0, 1, 1]
# en la base computacional
raw_state = np.array([1, 0, 1, 1], dtype=complex)
# Obtenemos su norma
norm = np.linalg.norm(raw_state)
# Normalizamos el estado
normalized_target = raw_state / norm
# Obtenemos su norma
norm2 = np.linalg.norm(normalized_target)

# Creamos las etiquetas binarias de los estados
basis_states = [format(i, '02b') for i in range(4)]

# Mostramos el estado no normalizado
print("Estado no normalizado:")
for i, amp in enumerate(raw_state):
    if not np.isclose(amp, 0.0):
        print(f"{amp:.4f} |{basis_states[i]}>")
# Mostramos la norma del estado no normalizado
print("\nNorma:", norm)

# Mostramos el estado normalizado
print("\nEstado normalizado:")
for i, amp in enumerate(normalized_target):
    if not np.isclose(amp, 0.0):
        print(f"{amp:.4f} |{basis_states[i]}>")
# Mostramos la norma del estado normalizado
print("\nNorma:", norm2)
