import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential

import tensorflow_probability as tfp
import tensorflow as tf 
tfpl = tfp.layers
tfd  = tfp.distributions

import numpy as np
from sklearn import mixture

class GMM_Parameters(layers.Layer):
    def __init__(self, batch_size, distribution=tfd.MultivariateNormalDiag, K=10, latent_dim=20, trainable=True, **kwargs):
        super().__init__()
        self.mu  = tf.Variable(tf.zeros((latent_dim,K)), trainable=trainable,name='mu_gmm')
        self.log_var = tf.Variable(tf.zeros((latent_dim,K)), trainable=trainable,name='logvar_gmm')
        
        self.batch_size = batch_size
        self.distribution = distribution
        self.K = K
        self.latent_dim = latent_dim
        
        # prior p(c)
        prior_probs  = tf.repeat(1./self.K, repeats=self.K)
        p_c = tfd.Categorical(probs=prior_probs)
        self.log_p_c = p_c.log_prob(range(self.K))

    def fit_gmm_params(self, z, use_gmm=False):
        if use_gmm:
            g = mixture.GaussianMixture(n_components=self.K, covariance_type='diag')
            g.fit(z)
            self.mu.assign(g.means_.astype(np.float32).T)
            self.log_var.assign(np.log(g.covariances_.astype(np.float32)).T)
        else:
            mu_gmm = np.load('mu.npy')
            var_gmm = np.load('var.npy')
            self.mu.assign(mu_gmm.T)
            self.log_var.assign(np.log(var_gmm).T)

        
    def call(self, z, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        log_p_c = tf.repeat(tf.expand_dims(self.log_p_c, axis=0),repeats=batch_size, axis=0)

        # XXX note mu and var have in each column the params for one Guassian out of K
        mu   = tf.repeat(tf.expand_dims(self.mu, axis=0), repeats=batch_size, axis=0)
        log_var  = tf.repeat(tf.expand_dims(self.log_var, axis=0), repeats=batch_size, axis=0)
        std = tf.math.exp(0.5*log_var)
        
        # I need to loop through K Gaussians!!!
        all_log_pz_c = []
        pz_c_all = []
        for i in range(self.K):
            pz_c = self.distribution(mu[...,i], std[...,i])
            pz_c_all.append(pz_c)
            all_log_pz_c.append(pz_c.log_prob(z))
        
        log_pz_c   = tf.stack(all_log_pz_c,axis=1)
        log_pc_pzc = log_p_c + log_pz_c
        pc_pzc     = tf.math.exp(log_pc_pzc)+1e-11 

        pi = pc_pzc/tf.reduce_sum(pc_pzc, axis=-1, keepdims=True)
        
        return pi, pz_c_all

    def generate_prior(self, K=1, L=1):
        mu       = self.mu[...,K]
        log_var  = self.log_var[...,K]
        std = tf.math.exp(0.5*log_var)
        
        pz_c = tfd.MultivariateNormalDiag(mu,std)

        z = pz_c.sample(L)

        return z

class DecMNIST(layers.Layer):
    def __init__(self,
                 latent_dim=512,
                 target_shape=(4,4,128),
                 kernel_size=3,
                 filters=32,
                 activation='relu',
                 name = 'decMNIST',
                 distribution=tfd.Laplace,
                 channel_out=1,
                 **kwargs):
        super(DecMNIST,self).__init__(name=name, **kwargs)
        units = np.prod(target_shape) 
        self.hidden_layers = Sequential(
                [
                layers.InputLayer(input_shape=(latent_dim,)),
                layers.Dense(units=units, activation=activation),
                layers.Reshape(target_shape=target_shape),
                layers.Conv2DTranspose(
                    filters=filters*2, kernel_size=kernel_size, strides=2, padding='same',output_padding=0,
                    activation=activation),
                layers.Conv2DTranspose(
                    filters=filters, kernel_size=kernel_size, strides=2, padding='same',output_padding=1,
                    activation=activation),
                layers.Conv2DTranspose(
                    filters=channel_out, kernel_size=kernel_size, strides=2, padding='same', output_padding=1),
                layers.Activation('linear', dtype='float32'),
                tfpl.DistributionLambda(lambda t: distribution(t, 0.75)),
                ]
                )
    
    def call(self, inputs):
        pz_x = self.hidden_layers(inputs)

        return pz_x

class EncMNIST(layers.Layer):
    def __init__(self,
                 input_shape = (28,28,1),
                 filters     = 32,
                 kernel_size = 3,
                 strides     = 2,
                 activation  = 'relu',
                 latent_dim  = 512,
                 n_samples   = 1,
                 name        = 'encMNIST',
                 distribution = tfd.MultivariateNormalDiag,
                 **kwargs):
        super(EncMNIST,self).__init__(name=name, **kwargs)
        self.hidden_layers = Sequential(
            [
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(
                filters=filters,   kernel_size=kernel_size, strides=strides, activation=activation, padding='same'),
            layers.Conv2D(
                filters=2*filters, kernel_size=kernel_size, strides=strides, activation=activation, padding='same'),
            layers.Conv2D(
                filters=4*filters, kernel_size=kernel_size, strides=strides, activation=activation, padding='same'),
            layers.Flatten(),
            layers.Dense(2*latent_dim),
            layers.Activation('linear', dtype='float32'),
            tfpl.DistributionLambda(lambda t: distribution(
                    t[..., :latent_dim], tf.math.exp(t[..., latent_dim:]))),
            ]
            )

    def call(self, inputs):
        qz_x = self.hidden_layers(inputs)

        return qz_x

if __name__ == '__main__':
    batch_size=10
    latent_dim=20

    mvn = tfd.MultivariateNormalDiag(loc=np.repeat(0.,latent_dim),scale_diag=np.repeat(1.,latent_dim))
    z = mvn.sample(batch_size)
    gmm = GMM_Parameters(batch_size)
    q_c_x,_ = gmm(z)
    gmm.fit_gmm_params(z)
    print(gmm.variables)
