from scipy.spatial.distance import pdist, squareform
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.losses import binary_crossentropy


EPSILON = 1e-7


def generator_loss(self,fake_output):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))

def discriminator_loss(self, real_output, fake_output):
    d_loss_real = tf.reduce_mean(binary_crossentropy(tf.ones_like(real_output), real_output))
    d_loss_fake = tf.reduce_mean(binary_crossentropy(tf.zeros_like(fake_output), fake_output))
    d_loss = (d_loss_real + d_loss_fake) / 2

    return d_loss

# Wasserstein losses
# def discriminator_loss(self,real_output, fake_output):
#    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

# def generator_wasserstein_loss(self,fake_output):
#    return -tf.reduce_mean(fake_output)


def generator_total_loss(self, fake_output, designs_science, designs_cost):
    g_loss = self.generator_loss(fake_output) 
    p_loss = tf.cast(self.pareto_loss(designs_science, designs_cost), tf.float32)
    div_loss = self.diversity_loss()
    return g_loss + p_loss + div_loss

def diversity_score(data, subset_size=10, sample_times=1000):
        r = tf.reduce_sum(tf.square(data), axis=1, keepdims=True)
        D = r - 2*tf.matmul(data, tf.transpose(data)) + tf.transpose(r)
        S = tf.exp(-0.5*D) # similarity matrix (rbf)
        # Average log determinant

        eig_val, _ = tf.linalg.eigh(S)
        loss = -tf.reduce_sum(tf.math.log(tf.maximum(eig_val, EPSILON)))

        return loss


def diversity_score(data, subset_size=10, sample_times=1000):
    # Average log determinant
    N = data.shape[0]
    data = data.reshape(N, -1)
    list_logdet = []
    for i in range(sample_times):
        ind = np.random.choice(N, size=subset_size, replace=False)
        subset = data[ind]
        D = squareform(pdist(subset, 'euclidean'))
        S = np.exp(-0.5*np.square(D))
        (sign, logdet) = np.linalg.slogdet(S)
        list_logdet.append(logdet)
    return tf.reduce_mean(list_logdet)





def diversity_loss(generated1,generated2):
   
    diff_samples = generated1 - generated2
    diff_z = z1 - z2
    
    norm_diff_samples = tf.cast(tf.norm(diff_samples, axis=1),tf.float64)
    norm_diff_z = tf.cast(tf.norm(diff_z, axis=1),tf.float64)
    
    div_loss = tf.reduce_mean(norm_diff_samples / norm_diff_z)
    
    return -div_loss



def generate_real_samples(num_samples):
        X = np.zeros((num_samples, 60))
        for i in range(num_samples):
            # Generate random binary array
            design = np.random.randint(2, size=60)
            # Ensure the cost constraint is satisfied
            X[i] = design
        return X



model_name = 'PR1'
generator = load_model(r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\AssigningProblem\Models\generator_model_'+model_name+'.h5', custom_objects={'generator_total_loss': generator_total_loss, 'discriminator_loss':discriminator_loss, 'genarator_loss':generator_loss})


samples1 = generate_real_samples(1000)
samples2 = generate_real_samples(1000)
z1 = tf.random.uniform([1000, 256])
z2 = tf.random.uniform([1000, 256])

generated1 = tf.math.round(generator(z1))
generated2 = tf.math.round(generator(z2))

diversity11 = diversity_score(generated1)
diversity12 = diversity_score(generated2)
diversity1 = (diversity11+diversity12)/2


diversity2 = diversity_loss(generated1,generated2)


div_samples = diversity_score(samples1)
div_samples2 = diversity_loss(samples1,samples2)


print(diversity1)
print(diversity2)
print(div_samples)
print(div_samples2)

    