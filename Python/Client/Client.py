from py4j.java_gateway import JavaGateway
from py4j.java_collections import JavaList
import tensorflow as tf
import numpy as np



def evaluate(design):
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




    # model_output_list = JavaList(model_output, gateway.jvm)
    # target_output_list = JavaList(target_output, gateway.jvm)
    # Call the computeLoss method with the Java lists

# # Create a Java gateway client
# gateway = JavaGateway()

# # Get a reference to the Java object
# my_java_object = gateway.entry_point

# model_output = [0.1, 0.2, 0.3, 0.4]
# target_output = [0.2, 0.3, 0.4, 0.5]

# java_list_model = gateway.jvm.java.util.ArrayList()
# java_list_target = gateway.jvm.java.util.ArrayList()
# for i in range(len(model_output)):
    
#     java_list_model.append(model_output[i])
#     java_list_target.append(target_output[i])

# # model_output_list = JavaList(model_output, gateway.jvm)
# # target_output_list = JavaList(target_output, gateway.jvm)
# # Call the computeLoss method with the Java lists
# loss = my_java_object.computeLoss(java_list_model, java_list_target)

# print(loss)
