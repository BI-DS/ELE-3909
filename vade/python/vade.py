import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
from tensorflow.keras import mixed_precision
import numpy as np

from enc_dec import GMM_Parameters, DecMNIST, EncMNIST

class VADE(tf.keras.Model):
    def __init__(self, enc, dec, alpha=10, batch_size=256, latent_dim=30, K=10, x_shape=(28,28,1)):
        super().__init__()
        self.gmm = GMM_Parameters(batch_size,latent_dim=latent_dim, K=K)
        self.enc = enc(latent_dim=latent_dim)
        self.dec = dec(latent_dim=latent_dim)
        self.K = K
        self.x_shape = x_shape
        self.alpha = alpha

        self.params = self.gmm.variables + self.enc.variables + self.dec.variables

    def call(self, x):
        qz_x = self.enc(x)
        # pz is a list with K priors
        pi, pz_c = self.gmm(qz_x.mean())
        px_z = self.dec(qz_x.sample())

        kl_all = []
        for i in range(self.K):
            kl = tfd.kl_divergence(qz_x, pz_c[i])
            kl_all.append(kl)
        kl_gmm = tf.stack(kl_all, axis=1)
        
        llik_log_prob = tf.reduce_sum(tf.reshape(px_z.log_prob(x),[-1,np.prod(self.x_shape)]),-1)
        entropy = tf.reduce_sum(-pi*tf.math.log(pi),axis=-1)
        kl = tf.reduce_sum(pi*kl_gmm,axis=-1)
        
        elbo = self.alpha*llik_log_prob - kl + entropy
        self.loss = -1*elbo 
        
        losses = {'llik':llik_log_prob, 'kl':kl, 'entropy':entropy}
        
        return losses

    def load_weights(self, path_to_weights='../output/ae'):
        enc_weights = np.load(os.path.join(path_to_weights,'enc_weights.npy'), allow_pickle=True)
        dec_weights = np.load(os.path.join(path_to_weights,'dec_weights.npy'), allow_pickle=True)
        
        print('loading pre-trained weights...')
        for i, w in enumerate(enc_weights):
            self.enc.variables[i].assign(w)
        for i, w in enumerate(dec_weights):
            self.dec.variables[i].assign(w)

    def draw_z(self, x):
        qz_x = self.enc(x)
        z = qz_x.mean()
        
        return z

    @tf.function
    def train(self, x, optimizer):
        with tf.GradientTape() as tape:
            losses = self.call(x)

        gradients = tape.gradient(self.loss, self.params)
        optimizer.apply_gradients(zip(gradients, self.params))
        
        return losses 

if __name__ == '__main__':
    batch_size = 100
    vade = VADE(EncMNIST, DecMNIST, batch_size=batch_size, K=3)
    x = np.random.rand(batch_size,28,28,1)
    losses = vade(x)
    #print(type(vade.gmm.variables))
