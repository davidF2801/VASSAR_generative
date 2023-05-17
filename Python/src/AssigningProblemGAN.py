import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.losses import binary_crossentropy
from tensorflow.keras.models import load_model
import pandas as pd
import math
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
import sys
from os.path import abspath, dirname
sys.path.append(dirname(dirname(abspath(__file__))))
from Client import Client

import os


# Separate loss functions for generator and discriminator
import itertools




# Binary design variables example
class AssigningProblemGAN:
    def __init__(self, model_name):
        self.num_design_vars = 60
        self.num_examples = 700
        self.latent_dim = 256
        self.batch_size = 1
        self.num_epochs = 15
        self.num_episodes = 7
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





  



    def build_generator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, input_dim=self.latent_dim, activation='relu'))
        model.add(layers.Dense(64, input_dim=128, activation='relu'))
        model.add(layers.Dense(32, input_dim=64, activation='relu'))
        model.add(layers.Dense(16, input_dim=32, activation='relu'))
        model.add(layers.Dense(self.num_design_vars, activation='sigmoid'))
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

 

    def evaluate_design(self,designs):
        # power = tf.reduce_sum(tf.multiply(design, self.component_power), axis=-1)
        # cost = tf.reduce_sum(tf.multiply(design, self.component_costs), axis=-1)
        # fitness = power/cost
        # return power,cost,fitness
        if len(designs) == 60:
            return Client.evaluate(designs)
        else:
            sciences = []
            costs = []

            for design in designs:
                design = design.numpy().tolist()
                power, cost = Client.evaluate(design)
                sciences.append(power)
                costs.append(cost)

            return sum(sciences)/len(sciences), sum(costs)/len(costs)
        






        



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
                    self.number_function_evaluations+=1
                    science=-science_normalized*self.science_max
                    cost = self.cost_max*cost_normalized
                    self.evaluated_designs[sol] = science_normalized, cost_normalized


                if science<0:
                    print('Hello')
                

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

                        best_designs.append(sol)
                        costs.append(cost)
                        sciences.append(science)
                        self.pareto_objectives.append((-science_normalized,cost_normalized))
                    
            if show:        

                plt.figure('Pareto Front')
                # plt.scatter(savings_opt, powers_opt, c='red', label='Optimal Pareto')
                plt.scatter(costs, sciences,c='blue', label='Achieved Pareto')
                
                plt.xlabel('Cost')
                plt.ylabel('Science benefit')
                plt.legend()
                if save:
                    path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\AssigningProblem\Pareto_Front'
                    costs = np.array(costs)
                    sciences = np.array(sciences)
                    designs = np.array(best_designs)

                    # Convert costs and sciences to 2D arrays
                    costs = costs.reshape(-1, 1)
                    sciences = sciences.reshape(-1, 1)

                    np.save(os.path.join(path, 'initial_pareto_2.npy'), np.hstack((costs, sciences, designs)))

                    # Save as CSV
                    np.savetxt(os.path.join(path, 'initial_pareto_2.csv'), np.hstack((costs, sciences, designs)), delimiter=',')

                    plt.savefig(os.path.join(path, 'Pareto_Front_INIT' + model_name))
                plt.show()


            return np.array(best_designs)
        else:
            
            # Specify the path to the saved files

            path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\AssigningProblem\Pareto_Front'

            # Load the data from the saved files
            data = np.load(os.path.join(path, 'initial_pareto.npy'))
            costs = data[:, 0]  # Extract the costs column
            sciences = data[:, 1]  # Extract the sciences column
            designs = data[:, 2:]  # Extract the designs columns

            # # Display the loaded data
            # print("Costs:", costs)
            # print("Sciences:", sciences)
            # print("Designs:", designs)
            return designs


    def normalize(self,set):
        normalized_solutions = []
        for solution in set:
            power, cost, _ = self.evaluate_design(solution)
            power_normalized = float(power) / float(self.max_power)
            cost_normalized = float(cost) / float(self.max_cost)
            normalized_solutions.append((power_normalized, cost_normalized))
        return normalized_solutions
    
        




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
        return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        d_loss_real = tf.reduce_mean(binary_crossentropy(tf.ones_like(real_output), real_output))
        d_loss_fake = tf.reduce_mean(binary_crossentropy(tf.zeros_like(fake_output), fake_output))
        d_loss = (d_loss_real + d_loss_fake) / 2

        return d_loss
    
    # Wasserstein losses
    # def discriminator_loss(self,real_output, fake_output):
    #    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    def generator_wasserstein_loss(self,fake_output):
       return -tf.reduce_mean(fake_output)


    def generator_total_loss(self, fake_output, designs_science, designs_cost):
        return self.generator_loss(fake_output) + self.pareto_loss(designs_science, designs_cost)


    def generate_real_samples(self,num_samples):
        X = np.zeros((num_samples, self.num_design_vars))
        for i in range(num_samples):
            # Generate random binary array
            design = np.random.randint(2, size=self.num_design_vars)
            # Ensure the cost constraint is satisfied
            X[i] = design
        return X
    


    #def constraint_satisfaction(self):




    
    def generate_fake_samples(self, training):
        if self.batch_size <= 1:
            n_samples = 1
        else:
            n_samples = round(self.batch_size/2)

        noise = tf.random.uniform([n_samples, self.latent_dim])
        fake_samples = self.generator(noise, training=training)
        fake_samples_thresh = tf.where(fake_samples >= 0.5, tf.ones_like(fake_samples), tf.zeros_like(fake_samples))
        y = tf.zeros((self.batch_size, 1))
        return fake_samples, fake_samples_thresh 



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
            real_data = self.pareto_front_calculator(solutions=data, show=nep==0, save=nep==0, calculate=True)
            real_data = self.pareto_front_calculator(solutions=real_data, show=nep==0, save=nep==0, calculate=True)
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
                        generated_samples, generated_samples_thresh=self.generate_fake_samples(training=False)
                        real_output = self.discriminator(batch, training=True)
                        fake_output = self.discriminator(generated_samples_thresh, training=True)
                        #d_loss = self.discriminator_loss(real_output, fake_output)
                        d_loss = self.discriminator_loss(real_output, fake_output)
                        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
                        if len(grads) == 0:
                            grads = [tf.random.normal(w.shape) for w in self.discriminator.trainable_variables]
                        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
                     
                    # Train the generator
                    with tf.GradientTape() as tape:
                        #generated_samples = self.generator(noise, training=True)
                        generated_samples, generated_samples_thresh=self.generate_fake_samples(training=True)
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
                                self.number_function_evaluations+=1
                                science -=s
                                cost+=c

                            science = science/len(generated_samples_thresh)
                            cost = cost/len(generated_samples_thresh)

                        sciences.append(science)
                        costs.append(cost)
                        #gst = np.expand_dims(generated_samples_thresh,axis=0)
                        data = np.vstack((data,generated_samples_thresh))
                        #data = np.concatenate(data,generated_samples_thresh)
                        fake_output = self.discriminator(generated_samples, training=False)
                        generator_loss = self.generator_total_loss(fake_output,science, cost)
                        self.number_function_evaluations+=1
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


        print(self.number_function_evaluations)
        final_pareto = self.pareto_front_calculator(data,True)
        path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\AssigningProblem\Losses'
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
       



      



                

model_name = 'BCEPL1000'


GAN = AssigningProblemGAN(model_name=model_name)
# path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\AssigningProblem\Pareto_Front'

# # Load the data from the saved files
# data = np.load(os.path.join(path, 'initial_pareto.npy'))
# costs = data[:, 0]  # Extract the costs column
# sciences = data[:, 1]  # Extract the sciences column
# designs = data[:, 2:]  # Extract the designs columns

#pareto = GAN.pareto_front_calculator(designs, show=True, save=False)

#design_space = list(itertools.product([0, 1], repeat=15))
#dataArray = np.array(design_space)
#np.savetxt('design_space.csv', design_space, delimiter=',')
# Load the design space from the file
#design_space = np.loadtxt('design_space.csv', delimiter=',')
#percentage = GAN.percentage_budget_fulfilled(design_space)

#print(percentage)
# Save the design space to a file

#optimal_pareto = GAN.pareto_front_calculator(design_space,True,True)

#data = np.load('optimal_pareto.npy')


#np.savetxt('optimal_pareto.csv', data, delimiter=",")


#savings = data[0]
#powers = data[1]


#plt.scatter(savings, powers)
#plt.xlabel('Savings (Max_cost - cost)')
#plt.ylabel('Generated Power')
#plt.show()
#Client.say_hello('David')

#GAN.train()






#GAN.generator.save(r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\AssigningProblem\Models\generator_model_'+model_name+'.h5')



## Load the saved generator model
generator = load_model(r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\AssigningProblem\Models\generator_model_'+model_name+'.h5', custom_objects={'generator_total_loss': GAN.generator_total_loss, 'discriminator_loss':GAN.discriminator_loss, 'genarator_loss':GAN.generator_loss})
# Generate designs
num_designs = input('Number of designs: ')



# create an empty dataframe to store the data
df = pd.DataFrame(columns=['instrument_{}'.format(i+1) for i in range(15)] + ['cost', 'generated_power'])

# loop through the designs and append to the dataframe
#for i in range(int(num_designs)):
num_designs = int(num_designs)
GAN.latent_dim = int(GAN.latent_dim)
noise = tf.random.normal([num_designs, GAN.latent_dim])
designs = generator.predict(noise)
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



path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\AssigningProblem\Pareto_Front'

# Scatter plot of designs
plt.scatter(costs, sciences, c='blue', label='Generated Designs')

# Load and plot points from initial_pareto_2.npy
data = np.load(os.path.join(path, 'initial_pareto_2.npy'))
initial_costs = data[:, 0]  # Extract the costs column
initial_sciences = data[:, 1]  # Extract the sciences column
plt.scatter(initial_costs, initial_sciences, c='red', label='Initial Pareto Designs')

# Set axis labels and title
plt.xlabel('Cost')
plt.ylabel('Science')
plt.title('Designs Scatter Plot')

# Add legend
plt.legend()

# Show the plot
plt.show()




        

        

columns = ['instrument_{}'.format(i+1) for i in range(GAN.num_design_vars)] + ['cost', 'science_benefit']
data = np.hstack((binary_designs, np.array([costs, sciences]).T))
df = pd.DataFrame(data, columns=columns)

# save the DataFrame to an Excel file
output_path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\AssigningProblem\Designs\generated_designs_'+model_name+'.xlsx'
df.to_excel(output_path, index=False)

print('Number of function evaluations: '+str(GAN.number_function_evaluations))



