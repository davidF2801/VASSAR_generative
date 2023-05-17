import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.losses import binary_crossentropy
from tensorflow.keras.models import load_model
import pandas as pd

# Separate loss functions for generator and discriminator
import itertools




# Binary design variables example
class SolarPowerExample:
    def __init__(self):
        self.num_design_vars = 10
        self.num_examples = 512
        self.latent_dim = 128
        self.batch_size = 128
        self.num_epochs = 3000
        self.learning_rate = 0.0002
        self.beta1 = 0.1
        self.lambda_s = 0.1
        # Define problem parameters
        self.budget = 1100
        # self.component_costs = np.array([1, 2, 3, 4, 5])
        # self.component_power = np.array([3, 5, 2, 4, 1])
        self.component_costs = [200, 150, 300, 400, 250, 200, 350, 300, 250, 400]
        self.component_power = [10, 12, 8, 9, 11, 13, 7, 6, 10, 12]
        self.budget_penalty = 0.01
        self.power_penalty = 0.1
        self.budget_tolerance = 0.05
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1)


        # Build the discriminator model
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.discriminator_loss,
                                   optimizer=self.discriminator_optimizer,
                                   metrics=['accuracy'])

        # Build the generator model
        self.generator = self.build_generator()
        self.generator.compile(loss=self.generator_loss,
                               optimizer=self.generator_optimizer)

        # Build the GAN model
        self.gan = self.build_gan()
        self.gan.compile(loss=self.generator_loss, 
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
        self.discriminator.trainable = False

        # Build the GAN by stacking the generator and discriminator
        gan_input = layers.Input(shape=(self.latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))
        gan = Model(gan_input, gan_output)
        return gan

 




    
    def objective_loss(self, fake_design):
        power_gen = tf.reduce_sum(tf.multiply(fake_design, self.component_power), axis=-1)

        budget_violation = tf.abs(tf.reduce_sum(tf.multiply(fake_design, self.component_costs), axis=-1) - self.budget) - self.budget_tolerance

        total_loss = self.budget_penalty * tf.reduce_mean(budget_violation) + self.power_penalty * tf.reduce_mean(1 / power_gen)


        return total_loss, power_gen


    

    def generator_loss(self, fake_output):

      return tf.reduce_mean(tf.math.log(1 - fake_output))


    def discriminator_loss(self, real_output, fake_output):
        d_loss_real = binary_crossentropy(tf.ones_like(real_output), real_output)
        d_loss_fake = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
        d_loss = (d_loss_real + d_loss_fake) / 2
        return d_loss


    def generate_real_samples(self,num_samples):
        X = np.zeros((num_samples, self.num_design_vars))
        for i in range(num_samples):
            # Generate random binary array
            design = np.random.randint(2, size=self.num_design_vars)
            cost = np.dot(design, self.component_costs)
            # Ensure the cost constraint is satisfied
            while cost > self.budget or np.any(np.all(X == design, axis=1)) :
                design = np.random.randint(2, size=self.num_design_vars)
                cost = np.dot(design, self.component_costs)
            X[i] = design
        return X
    


    #def constraint_satisfaction(self):

    def percentage_budget_fulfilled(self, design_space):
        # Count the number of designs that fulfill the budget constraint
        count = 0
        for design in design_space:
            cost = np.dot(design, self.component_costs)
            if cost <= self.budget:
                count += 1
        # Calculate the percentage of designs that fulfill the budget constraint
        percentage = count / len(design_space) * 100
        return percentage



    def generate_latent_points(self, num_samples):
        X = np.random.normal(0, 1, (int(num_samples), self.latent_dim))
        return X


    
    def generate_fake_samples(self, training):
        noise = tf.random.normal([self.batch_size, self.latent_dim])
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
        obj_losses = []
        gan_losses = []
        per_constraint = []
        power = []
        real_data = self.generate_real_samples(self.num_examples)
        #real_data = dataArray
        real_data_sliced = self.create_batches(real_data)

        
        for epoch in range(self.num_epochs):
            g_losses_batch = []
            d_losses_batch = []
            obj_losses_batch = []
            gan_losses_batch = []
            per_constraint_batch = []
            power_batch = []

            for batch in real_data_sliced:
            
                with tf.GradientTape() as tape:
                    #generated_samples = self.generator(noise, training=False)
                    generated_samples, generated_samples_thresh=self.generate_fake_samples(training=False)
                    
                    real_output = self.discriminator(batch, training=True)
                    fake_output = self.discriminator(generated_samples, training=True)
                    d_loss = self.discriminator_loss(real_output, fake_output)
                    grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
                    self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
                
                # Train the generator
            
                with tf.GradientTape() as tape:
                    #generated_samples = self.generator(noise, training=True)
                    generated_samples, generated_samples_thresh=self.generate_fake_samples(training=True)
                    fake_output = self.discriminator(generated_samples, training=False)
                    g_loss = self.generator_loss(fake_output)
                    obj_loss, power_generated = self.objective_loss(generated_samples)
                    g_loss_obj = g_loss + self.lambda_s*obj_loss
                    grads = tape.gradient(g_loss_obj, self.generator.trainable_weights)
                    self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

                per_constraint_batch.append(self.percentage_budget_fulfilled(generated_samples))
                g_losses_batch.append(g_loss)
                obj_losses_batch.append(obj_loss)
                gan_losses_batch.append(g_loss_obj)
                d_losses_batch.append(d_loss)
                power_batch.append(power_generated)

            per_constraint_m = tf.reduce_mean(per_constraint_batch)
            d_losses_m = tf.reduce_mean(d_losses_batch)
            g_losses_m = tf.reduce_mean(g_losses_batch)
            obj_losses_m = tf.reduce_mean(obj_losses_batch)
            gan_losses_m = tf.reduce_mean(gan_losses_batch)
            power_m = tf.reduce_mean(power_batch)

            g_losses.append(g_losses_m)
            obj_losses.append(obj_losses_m)
            gan_losses.append(gan_losses_m)
            d_losses.append(d_losses_m)
            per_constraint.append(per_constraint_m)
            power.append(power_m)

            print(f"Epoch: {epoch + 1}/{self.num_epochs}, Discriminator Loss: {d_losses_m}, Generator Loss: {g_losses_m}, Obj_loss = {obj_losses_m}, Percentage fulfilling = {per_constraint_m}, Generated power =  {power_m}")
            if epoch==self.num_epochs-1:
                print(generated_samples_thresh)


        plt.figure('G losses')
        plt.plot(g_losses, label='Generator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(r'C:\Users\dfornosf\Documents\TEST_GAN\gl.png')
        plt.show()

        plt.figure('Generated Power')
        plt.plot(power, label='Generated Power')
        plt.xlabel('Epochs')
        plt.ylabel('Power (Watts)')
        plt.legend()
        plt.savefig(r'C:\Users\dfornosf\Documents\TEST_GAN\pow.png')
        plt.show()

        plt.figure('Percentage constraint')
        plt.plot(per_constraint, label='Percentage of designs that fulfill the constraint')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend()
        plt.savefig(r'C:\Users\dfornosf\Documents\TEST_GAN\per.png')
        plt.show()
        


        plt.figure('D losses')
        plt.plot(d_losses, label='Discriminator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(r'C:\Users\dfornosf\Documents\TEST_GAN\dl.png')
        plt.show()
       


        plt.figure('Obj losses')
        plt.plot(obj_losses, label='Objective loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(r'C:\Users\dfornosf\Documents\TEST_GAN\objl.png')
        plt.show()
        


        plt.figure('GAN losses')
        plt.plot(gan_losses, label='GAN loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(r'C:\Users\dfornosf\Documents\TEST_GAN\ganl.png')
        plt.show()
        
                

      #print(f"Epoch: {epoch + 1}/{self.num_epochs}, Discriminator Loss: {d_loss}, Discriminator Accuracy: {discriminator_accuracy}, Generator Loss: {g_loss}, Generator Accuracy: {generator_accuracy}")

      



                




GAN = SolarPowerExample()
# design_space = list(itertools.product([0, 1], repeat=10))
# dataArray = np.array(design_space)
# np.savetxt('design_space.csv', design_space, delimiter=',')
# Load the design space from the file
design_space = np.loadtxt('design_space.csv', delimiter=',')
percentage = GAN.percentage_budget_fulfilled(design_space)

print(percentage)
# Save the design space to a file


GAN.train()

GAN.generator.save('generator_model.h5')


# Load the saved generator model
generator = load_model('generator_model.h5')

# Generate designs
num_designs = 128
designs = generator.predict(np.random.normal(size=(num_designs, 100)))


binary_designs = np.round(designs)

costs = np.sum(binary_designs * GAN.component_costs, axis=1)
generated_power = np.sum(binary_designs * GAN.component_power, axis=1)

df = pd.DataFrame(binary_designs, columns=['instrument_{}'.format(i+1) for i in range(10)])
df['cost'] = costs
df['generated_power'] = generated_power


df.to_excel('generated_designs.xlsx', index=False)



