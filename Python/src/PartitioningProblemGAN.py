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
from pymoo.indicators.hv import Hypervolume

import os


# Separate loss functions for generator and discriminator
import itertools




# Binary design variables example
class PartitioningProblemGAN:
    def __init__(self, model_name):
        self.num_design_vars = 24
        self.num_examples = 1999
        self.latent_dim = 256
        self.batch_size = 1
        self.num_epochs = 50
        self.num_episodes = 20
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





    # def custom_activation_function(self, designs):
    #     def custom_activation(designs_tensor):
    #         designs_np = K.get_value(designs_tensor)  # Convert tensor to NumPy array

    #         # Perform the necessary operations on the NumPy array

    #         designs_int = designs_np.copy()
    #         designs_dif = designs_np.copy()

    #         for j, design in enumerate(designs_int):
    #             instruments = design[0:self.num_instruments]
    #             orbits = design[self.num_instruments:self.num_design_vars]
    #             instruments_dif = design[0:self.num_instruments]
    #             orbits_dif = design[self.num_instruments:self.num_design_vars]

    #             for i, design_variable in enumerate(instruments):
    #                 if design_variable < 0:
    #                     instruments[i] = 0
    #                     instruments_dif[i] = 0
    #                 else:
    #                     instruments[i] = int(instruments[i] * (self.num_instruments - 1))
    #                     instruments_dif[i] = instruments[i] * (self.num_instruments - 1)

    #             for i, design_variable in enumerate(orbits):
    #                 if design_variable < 0:
    #                     orbits[i] = -1
    #                     orbits_dif[i] = -1
    #                 else:
    #                     orbits[i] = int(orbits[i] * (self.num_orbits - 1))
    #                     orbits_dif[i] = orbits[i] * (self.num_orbits - 1)

    #             designs_int[j] = np.append(instruments, orbits)
    #             designs_dif[j] = np.append(instruments_dif, orbits_dif)

    #         return designs_int

    #     return tf.py_function(custom_activation, [designs], tf.float32)



    def custom_activation_function(self,x):
        designs_int = x[:, :self.num_instruments]
        designs_dif = x[:, :self.num_instruments]
        instruments_dif = x[:, :self.num_instruments]
        orbits_dif = x[:, self.num_instruments:self.num_design_vars]
        
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
        model.add(layers.Dense(self.num_design_vars,input_dim=16, activation='tanh'))
        model.add(layers.Dense(self.num_design_vars, input_dim=self.num_design_vars,activation=self.custom_activation_function))


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
        
        

    def pareto_front_calculator(self, solutions, show, save = False, calculate = True):

        if calculate:
            costs = []
            sciences = []
            best_designs = []

            for i,sol in enumerate(solutions):
                sol = tuple(sol)

                if sol in self.evaluated_designs:
                    science_normalized, cost_normalized = self.evaluated_designs[sol]
                    science=-science_normalized*self.science_max
                    cost = self.cost_max*cost_normalized

                else:
                    print('Evaluating design '+str(i+1)+' out of '+str(len(solutions)))

                    science_normalized, cost_normalized = self.evaluate_design(sol)
                    if science_normalized>0:
                        print('Hello')
                    
                    science=-science_normalized*self.science_max
                    cost = self.cost_max*cost_normalized
                    self.evaluated_designs[sol] = science_normalized, cost_normalized


                

                dominated = False
                for i, design in enumerate(best_designs):
                    p_science, p_cost = sciences[i], costs[i]

                    if (cost>= p_cost and science<=p_science) :
                        dominated = True
                        break

                if dominated==False:
                        for i, design in enumerate(best_designs):
                            p_science, p_cost = sciences[i], costs[i]
                            if cost<=p_cost and science>=p_science:
                                best_designs.pop(i)
                                costs.pop(i)
                                sciences.pop(i)
                                self.pareto_objectives.pop(i)

                        best_designs.append(sol)
                        costs.append(cost)
                        sciences.append(science)
                        self.pareto_objectives.append((-science_normalized,cost_normalized))
                    
            if show or save:        

                plt.figure('Pareto Front')
                # plt.scatter(savings_opt, powers_opt, c='red', label='Optimal Pareto')
                plt.scatter(costs, sciences,c='blue', label='Achieved Pareto')
                
                plt.xlabel('Cost')
                plt.ylabel('Science benefit')
                plt.legend()
                if save:
                    path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\PartitioningProblem\Pareto_Front'
                    costs = np.array(costs)
                    sciences = np.array(sciences)
                    designs = np.array(best_designs)

                    # Convert costs and sciences to 2D arrays
                    costs = costs.reshape(-1, 1)
                    sciences = sciences.reshape(-1, 1)

                    np.save(os.path.join(path, 'initial_pareto_wl.npy'), np.hstack((costs, sciences, designs)))

                    # Save as CSV
                    np.savetxt(os.path.join(path, 'initial_pareto_wl.csv'), np.hstack((costs, sciences, designs)), delimiter=',')

                    plt.savefig(os.path.join(path, 'Pareto_Front_INIT' + model_name))

                if show:

                    plt.show()


            return np.array(best_designs)
        else:
            
            # Specify the path to the saved files

            path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\PartitioningProblem\Pareto_Front'

            # Load the data from the saved files
            data = np.load(os.path.join(path, 'initial_pareto_part.npy'))
            costs = data[:, 0]  # Extract the costs column
            sciences = data[:, 1]  # Extract the sciences column
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

    #def pareto_loss(self, current_pareto, design):
    #    n_pareto = np.array(self.normalize(current_pareto))
    #    n_design = np.array(self.normalize(design))
        
    #    ind = GD(n_pareto)
    #    distance = ind(n_design)
    #    #distance2 = self.pareto_loss2(current_pareto, design)
    #    return distance


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
    

    def calc_hv(self,pareto_front):
        hv = Hypervolume(ref_point=np.array([1.0, 1.0]))
        return hv.calc(pareto_front)
 


    def hypervolume_loss(self,current_pareto, new_designs):

        pareto_values = self.evaluated_designs[current_pareto]
        new_values = self.evaluated_designs[new_designs]
        pareto_values = np.array(pareto_values)
        new_values = np.array(new_values)

        # Normalize science and cost values
        normalized_pareto = pareto_values / np.array([self.science_max, self.cost_max])
        normalized_new_designs = new_values / np.array([self.science_max, self.cost_max])

        # Calculate the hypervolume of the current Pareto front
        hv_current = self.calc_hv(normalized_pareto)

        # Calculate the hypervolume of the combined Pareto front (current Pareto + new designs)
        combined_pareto = np.concatenate((normalized_pareto, normalized_new_designs), axis=0)
        hv_combined = self.calc_hv(combined_pareto)

        # Calculate the difference in hypervolume
        hv_difference = hv_combined - hv_current

        return hv_difference
    




    
    # Wasserstein losses
    # def discriminator_loss(self,real_output, fake_output):
    #    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    # def generator_wasserstein_loss(self,fake_output):
    #    return -tf.reduce_mean(fake_output)


    def generator_total_loss(self, fake_output, designs_science, designs_cost):
        g_loss = self.generator_loss(fake_output) 
        p_loss = tf.cast(self.pareto_loss(designs_science, designs_cost), tf.float32)
        div_loss = self.diversity_loss()

        return self.disc_lambda*g_loss + self.pareto_lambda*p_loss + self.div_lambda*div_loss



    def generate_real_samples(self,num_samples):

        data = []
        training_data = np.zeros((1, self.num_design_vars)) 
        for i in range(num_samples):
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

            data.append((tuple(instrument_partitioning),tuple(orbit_assignment)))
            appended = np.append(instrument_partitioning, orbit_assignment)
            training_data = np.vstack((training_data, np.append(instrument_partitioning, orbit_assignment)))


        return training_data
    
    def diversity_loss(self):
        z1 = tf.random.uniform([1, self.latent_dim])
        z2 = tf.random.uniform([1, self.latent_dim])
        generated1 = self.generator(z1)
        generated2 = self.generator(z2)
        diff_samples = generated1 - generated2
        diff_z = z1 - z2
        
        norm_diff_samples = tf.norm(diff_samples, axis=1)
        norm_diff_z = tf.norm(diff_z, axis=1)
        
        div_loss = tf.reduce_mean(norm_diff_samples / norm_diff_z)
        
        return -div_loss



    def generate_fake_samples(self, training):
        if self.batch_size <= 1:
            n_samples = 1
        else:
            n_samples = round(self.batch_size/2)

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


        
    def train(self):
      # Initialize the GAN training process
        num_batches = int(self.num_examples / self.batch_size)
        d_losses = []
        g_losses = []
        sciences = []
        costs = []
        data = self.generate_real_samples(self.num_examples)
        #real_data = dataArray
        
        for nep in range(self.num_episodes):

            print(f"Episode {nep}")
            #calculate = nep!= 0
            real_data = self.pareto_front_calculator(solutions=data, show=False, save=False, calculate=False)
            #real_data = self.pareto_front_calculator(solutions=real_data, show=nep==0, save=nep==0, calculate=True)
            #real_data = self.pareto_front_calculator(solutions=data, show=nep==0, save=nep==0, calculate=nep!=0)
            self.batch_size=len(real_data)
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
                        for sol in generated_samples_thresh:
                            sol = tuple(sol.numpy())
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
                        generator_loss = self.generator_total_loss(fake_output,science, cost)
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
        final_pareto = self.pareto_front_calculator(data,True)
        path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\PartitioningProblem\Losses'
        plt.figure('G losses')
        plt.plot(g_losses, label='Generator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(path +  '\gl.png')
        plt.show()

        plt.figure('Costs')
        plt.plot(costs, label='Costs')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.legend()
        plt.savefig(path + '\cost.png')
        plt.show()

        plt.figure('Generated Power')
        plt.plot(sciences, label='Generated Power')
        plt.xlabel('Epochs')
        plt.ylabel('Power (Watts)')
        plt.legend()
        plt.savefig(path + '\pow.png')
        plt.show()



        plt.figure('D losses')
        plt.plot(d_losses, label='Discriminator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(path + '\dl.png')
        plt.show()
       



      



                

model_name = 'BCEDL'


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
utigenerator = load_model(r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\PartitioningProblem\Models\generator_model_'+model_name+'.h5', custom_objects={'generator_total_loss': GAN.generator_total_loss, 'discriminator_loss':GAN.discriminator_loss, 'genarator_loss':GAN.generator_loss, 'custom_activation_function':GAN.custom_activation_function})
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



