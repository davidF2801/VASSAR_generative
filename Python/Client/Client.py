from py4j.java_gateway import JavaGateway
from py4j.java_collections import JavaList
import tensorflow as tf
import numpy as np



def evaluateA(design):
    # Create a Java gateway client
    #design = design.numpy().astype(np.float64)
    

    gateway = JavaGateway()
    # Get a reference to the Java object
    my_java_object = gateway.entry_point
    design_array = np.array(design, dtype=np.float64)
    # Convert Python lists to Java ArrayLists
    java_list_design = gateway.jvm.java.util.ArrayList()
    for i in range(len(design_array)):
        java_list_design.append(design_array[i])

    result = my_java_object.EvaluatePythonArchitecture(java_list_design)

    return list(result)



def evaluateP(instruments, orbits):
    # Create a Java gateway client
    #design = design.numpy().astype(np.float64)
    

    gateway = JavaGateway()
    # Get a reference to the Java object
    my_java_object = gateway.entry_point
    instruments_array = np.array(instruments, dtype=np.int32).tolist()  # Convert to list
    orbits_array = np.array(orbits, dtype=np.int32).tolist()
    if isinstance(instruments_array[0], list):
        instruments_array = instruments_array[0]
    

    # Convert orbits to list if not already a list

    if isinstance(orbits_array[0], list):
        orbits_array = orbits_array[0]
    # Convert Python lists to Java ArrayLists
    java_list_instruments = gateway.jvm.java.util.ArrayList()
    java_list_orbits = gateway.jvm.java.util.ArrayList()
    for i in range(len(instruments)):
        java_list_instruments.append(instruments_array[i])

    for i in range(len(orbits)):
        java_list_orbits.append(orbits_array[i])


    

    result = my_java_object.EvaluatePythonArchitecture(java_list_instruments,java_list_orbits)

    return list(result)


