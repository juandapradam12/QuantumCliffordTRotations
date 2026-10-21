##############################################################
########################## UNIR  #############################
##########    MASTER EN COMPUTACIÓN CUÁNTICA    ##############
##########        ALGORTIMOS CUÁNTICOS        ################
#####  LABORATORIO: COMPUTACIÓN CUÁNTICA DIFERENCIAL   #######
##############################################################
##############################################################

##############################################################
#####Este programa permite aplicar la puerta U3 al estado 0###
###de un qubit en función de los parámetros decididos########
##############################################################

# Se importan las librerías y herramientas
import pennylane as qml
from pennylane import numpy as np

# Definimos los parámetros de la puerta U3
theta = np.pi / 2
phi = np.pi / 2
lam = np.pi

# Creamos un dispositivo de un qubit
dev = qml.device("default.qubit", wires=1)

# Definimos el circuito que aplica el operador
@qml.qnode(dev)
def circuit():
    qml.U3(theta, phi, lam, wires=0)
    return qml.state()

# Ejecutamos el circuito y obtenemos el estado
state = circuit()
print("Estado cuántico resultante:", state)
