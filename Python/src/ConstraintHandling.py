import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def calculate_feasibility_metric(instrument_partitioning, orbit_assignment):
    num_instruments = len(instrument_partitioning)
    num_partitions = len(set(instrument_partitioning))
    num_assigned_orbits = sum(orbit != -1 for orbit in orbit_assignment)

    if num_assigned_orbits != num_partitions:
        # Calculate the first condition distance
        condition1_distance = abs(num_assigned_orbits - num_partitions) / max(float(num_partitions),float(num_assigned_orbits))
    else:
        condition1_distance = 0

    max_partition_index = 0
    for partition in instrument_partitioning:
        if partition > max_partition_index:
            if partition != max_partition_index + 1:
                # Calculate the second condition distance
                condition2_distance = abs(partition - (max_partition_index + 1)) / float(partition)
                break
            max_partition_index = partition
    else:
        condition2_distance = 0

    # Calculate the overall feasibility metric as a weighted combination of the condition distances
    feasibility_metric = (1 - condition1_distance) * (1 - condition2_distance)

    return feasibility_metric


def generate_real_samples(num_samples):
        num_design_vars = 24
        num_instruments = 12
        num_orbits = 5
        data = []
        training_data = np.zeros((0, num_design_vars), dtype=int)  # Initialize as an empty array

        while len(data) < num_samples:
            instrument_partitioning = np.zeros(num_instruments, dtype=int)
            orbit_assignment = np.zeros(num_instruments, dtype=int)

            max_num_sats = np.random.randint(num_instruments) + 1

            for j in range(num_instruments):
                instrument_partitioning[j] = np.random.randint(max_num_sats)

            sat_index = 0
            sat_map = {}

            for m in range(num_instruments):
                sat_id = instrument_partitioning[m]
                if sat_id in sat_map:
                    instrument_partitioning[m] = sat_map[sat_id]
                else:
                    instrument_partitioning[m] = sat_index
                    sat_map[sat_id] = sat_index
                    sat_index += 1

            instrument_partitioning.sort()

            num_sats = len(sat_map.keys())

            for n in range(num_instruments):
                if n < num_sats:
                    orbit_assignment[n] = np.random.randint(num_orbits)
                else:
                    orbit_assignment[n] = -1

            design_tuple = (tuple(instrument_partitioning), tuple(orbit_assignment))

            if design_tuple not in data:  # Check if the design is already present in the data list
                data.append(design_tuple)
                appended = np.append(instrument_partitioning, orbit_assignment)
                training_data = np.vstack((training_data, appended))

        return training_data[:num_samples]  # Return only the required number of samples
    
# Example usage
n_samples = 100


for i in range(n_samples):
    random = np.random.random()

    if random<0.5:
        samples = generate_real_samples(1) #Feasible designs
    else:
        samples = np.concatenate((np.random.randint(low=0, high=11, size=(1, 12)),
                          np.random.randint(low=0, high=5, size=(1, 12))), axis=1)
        # feasible_design = generate_real_samples(1)  # Generate a feasible design
        # infeasible_design = feasible_design.copy()  # Create a copy of the feasible design

        # # Make a different change to the design to make it slightly infeasible
        # # For example, let's subtract 1 from the first element in the instrument partitioning
        # infeasible_design[0, 0] -= 1

        # samples = infeasible_design


    instrument_partitioning = tuple(samples[:,0:12][0])
    orbit_assignment = tuple(samples[:,12:24][0])

    feasibility_measure = calculate_feasibility_metric(instrument_partitioning, orbit_assignment)

    print('Instruments: '+str(instrument_partitioning))
    print('Orbits: '+str(orbit_assignment))
    print('Design feasibility: '+str(feasibility_measure))

    instrument_partitioning_oh = np.one


