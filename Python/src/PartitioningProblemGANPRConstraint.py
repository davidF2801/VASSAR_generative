import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.losses import binary_crossentropy
from tensorflow.keras.models import load_model
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
import sys
from os.path import abspath, dirname
sys.path.append(dirname(dirname(abspath(__file__))))
from Client import Client
from tensorflow.keras import backend as K
import warnings
import os
EPSILON = 1e-7
warnings.filterwarnings("ignore", message="Tensor._shape is private, use Tensor.shape instead.")


# Separate loss functions for generator and discriminator
import itertools




# Binary design variables example
class PartitioningProblemGAN:
    def __init__(self, model_name):
        self.num_design_vars = 24
        self.num_examples = 2000
        self.num_pareto_fronts = 4
        self.latent_dim = 256
        self.batch_size = 16
        self.num_epochs = 50
        self.num_episodes = 10
        self.learning_rate = 0.002
        self.beta1 = 0.1
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1)
        self.number_function_evaluations = 0
        self.model_name = model_name
        self.science_max = 0.425
        self.cost_max = 25000.0
        self.pareto_objectives = []
        self.evaluated_designs = {}
        self.num_instruments = 12
        self.num_orbits = 5
        self.div_lambda = 0.1
        self.pareto_lambda = 1
        self.disc_lambda = 0.5

        self.lambda0 = 2
        self.gamma1 = 0.5

        self.num_epochs_cons = 10


        # Build the discriminator model
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.discriminator_loss,
                                   optimizer=self.discriminator_optimizer,
                                   metrics=['accuracy'])

        # Build the generator model
        self.generator = self.build_generator()
        self.generator.compile(loss=self.generator_total_loss,
                               optimizer=self.generator_optimizer)

        # Build the GAN model
        self.gan = self.build_gan()
        self.gan.compile(loss=self.generator_total_loss, 
                         optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=self.beta1))






    def custom_activation_function(self,x):
        x_tanh = tf.math.tanh(x)
        instruments_dif = x_tanh[:, :self.num_instruments]
        orbits_dif = x_tanh[:, self.num_instruments:self.num_design_vars]
        
        instruments = (11/2) * instruments_dif + 11/2
        orbits = (5 / 2) * orbits_dif + 3 / 2
        
        output = tf.concat([instruments, orbits], axis=1)
        return output


    def build_generator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, input_dim=self.latent_dim, activation='relu'))
        model.add(layers.Dense(64, input_dim=128, activation='relu'))
        model.add(layers.Dense(32, input_dim=64, activation='relu'))
        model.add(layers.Dense(16, input_dim=32, activation='relu'))
        model.add(layers.Dense(self.num_design_vars, input_dim=16,activation=self.custom_activation_function))


        return model

    def build_discriminator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, input_dim=self.num_design_vars, activation='relu'))
        model.add(layers.Dense(64, input_dim=128, activation='relu'))
        model.add(layers.Dense(32, input_dim=64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model
    


    def build_gan(self):
        # Freeze the discriminator's weights during GAN training
        self.discriminator.trainable = True

        # Build the GAN by stacking the generator and discriminator
        gan_input = layers.Input(shape=(self.latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))
        gan = Model(gan_input, gan_output)
        return gan

 

    def evaluate_design(self,design):
            design = list(design)  # Convert tuple to a list
            instruments = design[0:self.num_instruments]
            orbits = design[self.num_instruments:self.num_design_vars]
                # Check instrument values


            for i in range(len(instruments)):
                if instruments[i] < 0:
                    instruments[i] = 0
                elif instruments[i] > 11:
                    instruments[i] = 11

            # Check orbit values
            for i in range(len(orbits)):
                if orbits[i] < -1:
                    orbits[i] = -1
                elif orbits[i] > 4:
                    orbits[i] = 4
            self.number_function_evaluations+=1



            
            return Client.evaluateP(instruments,orbits) 
    



    def calculate_feasibility_metric(self,instrument_partitioning, orbit_assignment):
        num_instruments = len(instrument_partitioning)
        num_partitions = len(set(instrument_partitioning))
        num_assigned_orbits = sum(orbit != -1 for orbit in orbit_assignment)

        if num_assigned_orbits != num_partitions:
            # Calculate the first condition distance
            condition1_distance = abs(num_assigned_orbits - num_partitions) / num_instruments
        else:
            condition1_distance = 0

        max_partition_index = 0
        violation_elements = 0
        for partition in instrument_partitioning:
            if partition > max_partition_index:
                if partition != max_partition_index + 1:
                    # Calculate the second condition distance
                    violation_elements+=1
                else:
                    max_partition_index = partition
            elif partition<max_partition_index:
                violation_elements+=1

        condition2_distance = violation_elements/ num_instruments
            

        # Calculate the overall feasibility metric as a weighted combination of the condition distances
        feasibility_metric = (1 - condition1_distance) * (1 - condition2_distance)

        #print('Feasibility: ' + str(feasibility_metric))

        return feasibility_metric
    


    def is_dominated(self, solution1, solution2):
        science1, cost1, feasibility1 = solution1
        science2, cost2, feasibility2 = solution2

        if science1 > science2 and cost1 < cost2 and feasibility1 > feasibility2:
            return True

        return False

    def compute_pareto(self,solutions):
        pareto_front_indices = []
        pareto_front_values = []
        
        for i in range(len(solutions)):
            is_dominated_by_others = False
            for j in range(len(solutions)):
                if i != j and self.is_dominated(solutions[j], solutions[i]):
                    is_dominated_by_others = True
                    break
            if not is_dominated_by_others:
                pareto_front_indices.append(i)
                pareto_front_values.append(solutions[i])
        
        return pareto_front_indices, pareto_front_values
    


    def pareto_ranking(self,values, designs, num_pareto_fronts):
        pareto_fronts = []
        pareto_designs = []
        pareto_values = []

        values = np.array(values)
        designs = np.array(designs)

        remaining_values = values.copy()
        remaining_designs = designs.copy()

        # Convert remaining_values to a list
        remaining_values_list = remaining_values.tolist()

        # Add feasibility scores to each sublist
        for i, design in enumerate(designs):
            instruments = tuple(design[:12])
            orbits = tuple(design[12:])
            feasibility_score = self.calculate_feasibility_metric(instruments, orbits)
            remaining_values_list[i].append(feasibility_score)

        # Convert remaining_values_list back to ndarray if needed
        remaining_values = np.array(remaining_values_list)


        for _ in range(num_pareto_fronts):
            pareto_front_indices, pareto_front_values = self.compute_pareto(remaining_values)
            pareto_front = [remaining_designs[i] for i in pareto_front_indices]

            if len(pareto_front) == 0:
                break

            pareto_fronts.append(pareto_front_indices)
            pareto_designs.append(pareto_front)
            pareto_values.append(pareto_front_values)

            remaining_values = np.delete(remaining_values, pareto_front_indices, axis=0)

        return pareto_values,pareto_designs


        
        

    def pareto_front_calculator(self, solutions, show, name, save = False, calculate = True):

        if calculate:
            costs = []
            sciences = []
            designs = []
            best_designs = []
            best_costs = []
            best_sciences = []
            values = []

            for i,sol in enumerate(solutions):
                sol = tuple(sol)

                if sol in self.evaluated_designs:
                    science_normalized, cost_normalized = self.evaluated_designs[sol]
                    science=-science_normalized*self.science_max
                    cost = self.cost_max*cost_normalized
                    costs.append(cost)
                    sciences.append(science)

                else:
                    print('Evaluating design '+str(i+1)+' out of '+str(len(solutions)))

                    science_normalized, cost_normalized = self.evaluate_design(sol)
                    if science_normalized>0:
                        print('Hello')
                    
                    science=-science_normalized*self.science_max
                    cost = self.cost_max*cost_normalized
                    self.evaluated_designs[sol] = science_normalized, cost_normalized
                    self.pareto_objectives.append([-science_normalized,cost_normalized])
                    costs.append(cost)
                    sciences.append(science)

                values.append([science,cost])
                designs.append(sol)



            pareto_values, pareto_designs = self.pareto_ranking(values,designs,self.num_pareto_fronts)


                    
            if show or save:

                for i, front_values in enumerate(pareto_values):
                    x = [val[0] for val in front_values]
                    y = [val[1] for val in front_values]
                    best_designs.extend(pareto_designs[i])
                    best_sciences.extend(x)
                    best_costs.extend(y)

                    plt.scatter(x, y, label='Pareto Front {}'.format(i+1))


                

                # Add labels and show the plot
                plt.xlabel('Science benefit')
                plt.ylabel('Cost')
                plt.title('Pareto Fronts')
                plt.xlim(0, 0.425)
                plt.ylim(0, 25000)
                plt.legend()
                plt.grid(True)

              
                if save:
                    path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\PartitioningProblem\Pareto_Front'

                    plt.savefig(os.path.join(path, name  + model_name))

                    costs = np.array(best_costs)
                    sciences = np.array(best_sciences)
                    designs = np.array(best_designs)


                    # Convert costs and sciences to 2D arrays
                    costs = costs.reshape(-1, 1)
                    sciences = sciences.reshape(-1, 1)

                    np.save(os.path.join(path, name + '.npy'), np.hstack((sciences, costs, designs)))

                    # Save as CSV
                    np.savetxt(os.path.join(path, name + '.csv'), np.hstack((sciences, costs, designs)), delimiter=',')

                if show:

                    plt.show()


            pareto_designs_array = np.concatenate(pareto_designs)
            plt.close()





            return pareto_designs_array

        

        else:
            # Specify the path to the saved files

            path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\PartitioningProblem\Pareto_Front'

            # Load the data from the saved files
            data = np.load(os.path.join(path, 'Pareto_Front_INIT.npy'))
            sciences = data[:, 0]  # Extract the sciences column
            costs = data[:, 1]  # Extract the costs column
            designs = data[:, 2:]  # Extract the designs columns
            sciences = -sciences/self.science_max
            costs = costs/self.cost_max
            designs_tuple = [tuple(design) for design in designs]
            for i,design in enumerate(designs_tuple):
                self.evaluated_designs[design] = sciences[i],costs[i]
                self.pareto_objectives.append((-sciences[i],costs[i]))
            # # Display the loaded data
            # print("Costs:", costs)
            # print("Sciences:", sciences)
            # print("Designs:", designs)
            return designs




    def pareto_loss(self, science_design, cost_design):
        distances = []
        for i in range(len(self.pareto_objectives)):
            science_pareto, cost_pareto = self.pareto_objectives[i]
            distance = ((science_pareto - science_design) ** 2 + (cost_pareto - cost_design) ** 2) ** 0.5
            distances.append(distance)

        best_distance = tf.math.reduce_min(distances)

        return best_distance


    #def generator_loss(self, fake_output):

    #  return tf.reduce_mean(tf.math.log(1 - fake_output)) 

    def generator_loss(self,fake_output):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))

    def discriminator_loss(self, real_output, fake_output):
        d_loss_real = tf.reduce_mean(binary_crossentropy(tf.ones_like(real_output), real_output))
        d_loss_fake = tf.reduce_mean(binary_crossentropy(tf.zeros_like(fake_output), fake_output))
        d_loss = (d_loss_real + d_loss_fake) / 2

        return d_loss
    

    def dpp_loss(self,data):
            p = []
            for i,design in enumerate(data):
                p.append(self.evaluated_designs[design])

            
            p = abs(np.asarray(p))
            w = tf.ones(p.shape)
            r = tf.reduce_sum(tf.square(data), axis=1, keepdims=True)
            D = r - 2*tf.matmul(data, tf.transpose(data)) + tf.transpose(r)
            S = tf.exp(-0.5*D) # similarity matrix (rbf)

            #w = np.random.rand(p.shape[1])

            y = tf.reduce_sum(w * p, axis=1)
            
            Q = tf.tensordot(tf.expand_dims(y, 1), tf.expand_dims(y, 0), 1) # quality matrix
            if self.lambda0 == 0.:
                L = S
            else:
                # Convert the tensors to the desired data type
                S = tf.cast(S, tf.float32)
                Q = tf.cast(Q, tf.float32)

                # Perform element-wise multiplication
                L = S * tf.pow(Q, self.lambda0)

            eig_val, _ = tf.linalg.eigh(L)
            loss = -tf.reduce_sum(tf.math.log(tf.maximum(eig_val, EPSILON)))

            return loss/len(data)

    
    
    # Wasserstein losses
    # def discriminator_loss(self,real_output, fake_output):
    #    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    # def generator_wasserstein_loss(self,fake_output):
    #    return -tf.reduce_mean(fake_output)


    def generator_total_loss(self, fake_output, designs_science, designs_cost, designs):

        g_loss = self.generator_loss(fake_output) 
        p_loss = tf.cast(self.pareto_loss(designs_science, designs_cost), tf.float32)
        dpp_loss = self.dpp_loss(designs)
        #div_loss = self.diversity_loss()
        # div_loss = self.diversity_score(designs)

        feasibility_score = 0

        for design in designs:
            instruments = design[:12]
            orbits = design[12:]
            feasibility_score += self.calculate_feasibility_metric(instruments,orbits)

        feasibility_score = abs(feasibility_score/len(designs))


        return g_loss + (1-feasibility_score) + self.gamma1*dpp_loss



    def generate_real_samples(self, num_samples):
        data = []
        training_data = np.zeros((0, self.num_design_vars), dtype=int)  # Initialize as an empty array

        while len(data) < num_samples:
            instrument_partitioning = np.zeros(self.num_instruments, dtype=int)
            orbit_assignment = np.zeros(self.num_instruments, dtype=int)

            max_num_sats = np.random.randint(self.num_instruments) + 1

            for j in range(self.num_instruments):
                instrument_partitioning[j] = np.random.randint(max_num_sats)

            sat_index = 0
            sat_map = {}

            for m in range(self.num_instruments):
                sat_id = instrument_partitioning[m]
                if sat_id in sat_map:
                    instrument_partitioning[m] = sat_map[sat_id]
                else:
                    instrument_partitioning[m] = sat_index
                    sat_map[sat_id] = sat_index
                    sat_index += 1

            instrument_partitioning.sort()

            num_sats = len(sat_map.keys())

            for n in range(self.num_instruments):
                if n < num_sats:
                    orbit_assignment[n] = np.random.randint(self.num_orbits)
                else:
                    orbit_assignment[n] = -1

            design_tuple = (tuple(instrument_partitioning), tuple(orbit_assignment))

            if design_tuple not in data:  # Check if the design is already present in the data list
                data.append(design_tuple)
                appended = np.append(instrument_partitioning, orbit_assignment)
                training_data = np.vstack((training_data, appended))

        return training_data[:num_samples]  # Return only the required number of samples
    
    def diversity_loss(self):
        z1 = tf.random.uniform([100, self.latent_dim])
        z2 = tf.random.uniform([100, self.latent_dim])
        generated1 = self.generator(z1)
        generated2 = self.generator(z2)
        diff_samples = generated1 - generated2
        diff_z = z1 - z2
        
        norm_diff_samples = tf.norm(diff_samples, axis=1)
        norm_diff_z = tf.norm(diff_z, axis=1)
        
        div_loss = tf.reduce_mean(norm_diff_samples / norm_diff_z)
        
        return -div_loss
    

    def diversity_score(data, subset_size=10, sample_times=1000):
        r = tf.reduce_sum(tf.square(data), axis=1, keepdims=True)
        D = r - 2*tf.matmul(data, tf.transpose(data)) + tf.transpose(r)
        S = tf.exp(-0.5*D) # similarity matrix (rbf)
        # Average log determinant

        eig_val, _ = tf.linalg.eigh(S)
        loss = -tf.reduce_sum(tf.math.log(tf.maximum(eig_val, EPSILON)))

        return loss




    def generate_fake_samples(self, training):
        if self.batch_size <= 1:
            n_samples = 1
        else:
            n_samples =4

        noise = tf.random.uniform([n_samples, self.latent_dim])
        generated_samples = self.generator(noise, training=training)
        #fake_samples,fake_samples_thresh = self.custom_activation_function(generated_samples)
        y = tf.zeros((self.batch_size, 1))
        return generated_samples
    
    


    def create_batches(self,data):
        # create an array of indices for the dataset
        indices = np.arange(data.shape[0])
        # shuffle the indices
        np.random.shuffle(indices)
        batch_data = []
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch = data[batch_indices]
            batch_data.append(batch)
        return batch_data

    def gumbel_softmax(self,logits, temperature):
        gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(logits.shape)))
        y = logits + gumbel_noise
        y = tf.nn.softmax(y / temperature)
        return y
    

    def translate_one_hot_to_design(self,one_hot_design):
        num_instruments = 12
        num_orbits = 6

        designs = []
        for i in range(one_hot_design.shape[0]):
            design = np.zeros(24, dtype=int)
            
            # Decode instrument partitioning
            instrument_partitioning = np.argmax(one_hot_design[i, :num_instruments*num_instruments].reshape((num_instruments, num_instruments)), axis=-1)
            # for j in range(num_instruments):
            #     design[j] = ins_indices[np.where(instrument_partitioning[j] == 1)[0][0]]
            
            # Decode orbit assignment
            orbit_assignment = np.argmax(one_hot_design[i, num_instruments*num_instruments:].reshape((num_instruments, num_orbits)), axis=-1)-1

            design = np.append(instrument_partitioning,orbit_assignment)
            
            designs.append(design)
        
        return designs


        
    def train(self):
      # Initialize the GAN training process
        num_batches = int(self.num_examples / self.batch_size)
        d_losses = []
        g_losses = []
        sciences = []
        costs = []
        data = self.generate_real_samples(self.num_examples)



        real_data_sliced_cons = self.create_batches(data)

        print('Constraint learning stage')


        for epoch in range(self.num_epochs_cons):

            for batch in real_data_sliced_cons:
                with tf.GradientTape() as tape:
                    #generated_samples = self.generator(noise, training=False)
                    generated_samples=self.generate_fake_samples(training=False)
                    generated_samples_thresh = tf.math.round(generated_samples)
                    # batch_soft = self.gumbel_softmax(batch, 0.5)
                    # gen_samples_soft = self.gumbel_softmax(generated_samples_thresh, 0.5)
                    real_output = self.discriminator(batch, training=True)
                    fake_output = self.discriminator(generated_samples, training=True)
                    #d_loss = self.discriminator_loss(real_output, fake_output)
                    d_loss = self.discriminator_loss(real_output, fake_output)
                # Calculate gradients with respect to discriminator trainable variables
                    grads = tape.gradient(d_loss, self.discriminator.trainable_variables)

                    # Add noise to gradients
                    if len(grads) == 0:
                        grads = [tf.random.normal(w.shape) for w in self.discriminator.trainable_variables]
                    else:
                        noisy_grads = []
                        for grad in grads:
                            noisy = tf.random.normal(grad.shape, mean=0.0, stddev=0.5)
                            noisy_grad = grad + noisy
                            noisy_grads.append(noisy_grad)
                        grads = noisy_grads

                    # Apply gradients with noise to update discriminator weights
                    self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
                    
                    
                # Train the generator
                with tf.GradientTape() as tape:
                    #generated_samples = self.generator(noise, training=True)
                    generated_samples=self.generate_fake_samples(training=True)
                    generated_samples_thresh = tf.math.round(generated_samples)
                    fake_output = self.discriminator(generated_samples, training=False)
                    feasibility_score = 0
                    for design in generated_samples_thresh.numpy():
                        design = tuple(design)
                        instruments = design[:12]
                        orbits = design[12:]
                        feasibility_score += self.calculate_feasibility_metric(instruments,orbits)
                    feasibility_score = abs(feasibility_score/len(generated_samples_thresh))

                    generator_loss = self.generator_loss(fake_output) + 1-feasibility_score
                    grads = tape.gradient(generator_loss, self.generator.trainable_weights)
                    if len(grads) == 0:
                        grads = [tf.random.normal(w.shape) for w in self.generator.trainable_variables]

                    self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
            print('Epoch: '+str(epoch)+' Generator Loss: '+str(generator_loss)+' Discriminator Loss: '+str(d_loss)+' Feasibility: '+str(feasibility_score))


        print('Optimization stage')



        #real_data = dataArray
        
        for nep in range(self.num_episodes):

            print(f"Episode {nep}")
            #calculate = nep!= 0
            real_data = self.pareto_front_calculator(solutions=data, show=nep==0, save=nep==0, calculate=nep!=0, name='Pareto_Front_INIT')
            real_data = np.asarray(self.translate_one_hot_to_design(real_data))
            #real_data = self.pareto_front_calculator(so lutions=real_data, show=nep==0, save=nep==0, calculate=True)
            #real_data = self.pareto_front_calculator(solutions=data, show=nep==0, save=nep==0, calculate=nep!=0)
            # self.batch_size=len(real_data)
            real_data_sliced = self.create_batches(real_data)
            data = real_data

        
            for epoch in range(self.num_epochs):
                g_losses_batch = []
                d_losses_batch = []
                science_batch = []
                costs_batch = []


                for batch in real_data_sliced:
                    with tf.GradientTape() as tape:
                        #generated_samples = self.generator(noise, training=False)
                        generated_samples=self.generate_fake_samples(training=False)
                        generated_samples_thresh = tf.math.round(generated_samples)
                        # batch_soft = self.gumbel_softmax(batch, 0.5)
                        # gen_samples_soft = self.gumbel_softmax(generated_samples_thresh, 0.5)
                        real_output = self.discriminator(batch, training=True)
                        fake_output = self.discriminator(generated_samples, training=True)
                        #d_loss = self.discriminator_loss(real_output, fake_output)
                        d_loss = self.discriminator_loss(real_output, fake_output)
                        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
                        if len(grads) == 0:
                            grads = [tf.random.normal(w.shape) for w in self.discriminator.trainable_variables]
                        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
                     
                    # Train the generator
                    with tf.GradientTape() as tape:
                        #generated_samples = self.generator(noise, training=True)
                        generated_samples=self.generate_fake_samples(training=True)
                        generated_samples_thresh = tf.math.round(generated_samples)

                        science = 0
                        cost = 0
                        designs_tuple = []
                        for sol in generated_samples_thresh:
                            sol = tuple(sol.numpy())
                            designs_tuple.append(sol)
                            if sol in self.evaluated_designs:
                                s, c = self.evaluated_designs[sol]
                                science -=s
                                cost+=c
                            else:
                                s, c = self.evaluate_design(sol)
                                self.evaluated_designs[sol] = s, c
                                science -=s
                                cost+=c
                        science = science/len(generated_samples_thresh)
                        cost = cost/len(generated_samples_thresh)

                        sciences.append(science)
                        costs.append(cost)
                        #gst = np.expand_dims(generated_samples_thresh,axis=0)
                        data = np.vstack((data,generated_samples_thresh))
                        # gen_samples_soft = self.gumbel_softmax(generated_samples_thresh, 0.5)
                        #data = np.concatenate(data,generated_samples_thresh)
                        fake_output = self.discriminator(generated_samples, training=False)
                        generator_loss = self.generator_total_loss(fake_output,science, cost,tuple(designs_tuple))
                        grads = tape.gradient(generator_loss, self.generator.trainable_weights)
                        if len(grads) == 0:
                            grads = [tf.random.normal(w.shape) for w in self.generator.trainable_variables]

                        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
                        

                    g_losses_batch.append(tf.reduce_mean(generator_loss))
                    #obj_losses_batch.append(obj_loss)
                    #gan_losses_batch.append(g_loss_obj)
                    d_losses_batch.append(tf.reduce_mean(d_loss))
                    science_batch.append(tf.reduce_mean(sciences))
                    costs_batch.append(tf.reduce_mean(costs))

                d_losses_m = tf.reduce_mean(d_losses_batch)
                g_losses_m = tf.reduce_mean(g_losses_batch)
                #obj_losses_m = tf.reduce_mean(obj_losses_batch)
                #gan_losses_m = tf.reduce_mean(gan_losses_batch)
                science_m = tf.reduce_mean(science_batch)
                cost_m = tf.reduce_mean(costs_batch)


 
                costs.append(cost_m)
                g_losses.append(g_losses_m)
                d_losses.append(d_losses_m)
                sciences.append(science_m)

                print(f"Epoch: {epoch + 1}/{self.num_epochs}, Discriminator Loss: {d_losses_m}, Generator Loss: {g_losses_m},  Science benefit =  {science_m}, Cost = {cost_m}")
                if epoch==self.num_epochs-1:
                    print(generated_samples_thresh)


        print("Number of function evaluations during training: "+str(self.number_function_evaluations))
        self.pareto_front_calculator(data,show=True,save=True,name='Pareto_Front_END')
        path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\PartitioningProblem\Losses'
        plt.figure('G losses')
        plt.plot(g_losses, label='Generator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(path +  '\gl.png')
        plt.show()

        plt.figure('Cost evolution')
        plt.plot(costs, label='Cost evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.legend()
        plt.savefig(path + '\cost.png')
        plt.show()

        plt.figure('Science benefit evolution')
        plt.plot(sciences, label='Science benefit evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Science benefit')
        plt.legend()
        plt.savefig(path + '\science.png')
        plt.show()



        plt.figure('D losses')
        plt.plot(d_losses, label='Discriminator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(path + '\dl.png')
        plt.show()
       



      



                

model_name = 'BCEPRfeasibility'


GAN = PartitioningProblemGAN(model_name=model_name)
# path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\AssigningProblem\Pareto_Front'

# # Load the data from the saved files
# data = np.load(os.path.join(path, 'initial_pareto.npy'))
# costs = data[:, 0]  # Extract the costs column
# sciences = data[:, 1]  # Extract the sciences column
# designs = data[:, 2:]  # Extract the designs columns
GAN.train()
GAN.generator.save(r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\PartitioningProblem\Models\generator_model_'+model_name+'.h5')



## Load the saved generator model
generator = load_model(r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\PartitioningProblem\Models\generator_model_'+model_name+'.h5', custom_objects={'generator_total_loss': GAN.generator_total_loss, 'discriminator_loss':GAN.discriminator_loss, 'genarator_loss':GAN.generator_loss, 'custom_activation_function':GAN.custom_activation_function})
# Generate designs
num_designs = input('Number of designs: ')



# create an empty dataframe to store the data
df = pd.DataFrame(columns=['instrument_{}'.format(i+1) for i in range(15)] + ['cost', 'generated_power'])

# loop through the designs and append to the dataframe
#for i in range(int(num_designs)):
num_designs = int(num_designs)
GAN.latent_dim = int(GAN.latent_dim)
noise = tf.random.normal([num_designs, GAN.latent_dim])
designs = GAN.generator.predict(noise)
binary_designs = np.round(designs)
print(binary_designs)
sciences = []
costs = []
for i,design in enumerate(binary_designs):
    print('Evaluating design '+str(i+1)+' out of '+str(len(designs)))
    design = tuple(design)
    if design in GAN.evaluated_designs:
        science_normalized, cost_normalized = GAN.evaluated_designs[design]
        science=-science_normalized*GAN.science_max
        cost = GAN.cost_max*cost_normalized

    else:
        science_normalized, cost_normalized = GAN.evaluate_design(design)
        GAN.number_function_evaluations+=1
        science=-science_normalized*GAN.science_max
        cost = GAN.cost_max*cost_normalized
        GAN.evaluated_designs[design] = science_normalized, cost_normalized

    sciences.append(science)
    costs.append(cost)
    print('Science benefit = '+str(science))
    print('Cost = '+str(cost))



path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\PartitioningProblem\Pareto_Front'

# Scatter plot of designs
#plt.scatter(costs, sciences, c='blue', label='Generated Designs')

# Load and plot points from initial_pareto_2.npy
#data = np.load(os.path.join(path, 'initial_pareto_2.npy'))
# initial_costs = data[:, 0]  # Extract the costs column
# initial_sciences = data[:, 1]  # Extract the sciences column
# plt.scatter(initial_costs, initial_sciences, c='red', label='Initial Pareto Designs')

# # Set axis labels and title
# plt.xlabel('Cost')
# plt.ylabel('Science')
# plt.title('Designs Scatter Plot')

# # Add legend
# plt.legend()

# # Show the plot
# plt.show()

columns = ['instrument_{}'.format(i+1) for i in range(GAN.num_design_vars)] + ['cost', 'science_benefit']
data = np.hstack((binary_designs, np.array([costs, sciences]).T))
df = pd.DataFrame(data, columns=columns)

# save the DataFrame to an Excel file
output_path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\PartitioningProblem\Designs\generated_designs_'+model_name+'.xlsx'
df.to_excel(output_path, index=False)

print('Number of function evaluations: '+str(GAN.number_function_evaluations))



