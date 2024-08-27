import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from tensorflow.keras import mixed_precision
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
tf.get_logger().setLevel('INFO')

from enc_dec import DecMNIST, EncMNIST
from ae import AutoEncoder

from scipy.optimize import linear_sum_assignment as linear_assignment
from utils import plot_digits, image_processing


def cluster_acc(Y_pred, Y):
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = np.transpose(np.asarray(linear_assignment(w.max() - w)))
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def train():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--latent_dim", default= 10, help="Dimensionality of the latent space", type=int)
    parser.add_argument("--epochs", default= 80, help="No of samples that the encoder draws", type=int)
    parser.add_argument("--save_every", default= 50, help="No of epochs to save cktps", type=int)
    parser.add_argument("--batch_size", default= 256, help="No. of samples that the encoder draws", type=int)
    parser.add_argument("--dset", default= 'mnist', help="Data set")
    parser.add_argument("--output_folder", default= '../output/ae', help="Output folder to save outputs")
    

    args = parser.parse_args()
    
    try:
        os.mkdir(args.output_folder)
    except OSError:
        print("Creation of the directory {}  failed".format(args.output_folder))
    else:
        print("Successfully created the directory {}".format(args.output_folder))
    
    print (args)
    print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
    
    tr_data, info = tfds.load('mnist',split='train', with_info=True)
    te_data = tfds.load('mnist',split='test')
    
    BUFFER   = info.splits['train'].num_examples
    AUTOTUNE = tf.data.AUTOTUNE
    tr_data  = tr_data.map(image_processing).cache().shuffle(BUFFER).batch(args.batch_size,drop_remainder=True).prefetch(AUTOTUNE)
    te_data  = te_data.map(image_processing).batch(7)

    model = AutoEncoder(EncMNIST(latent_dim=args.latent_dim), DecMNIST(latent_dim=args.latent_dim))
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.weights])
    print('AE model has {} trainable params'.format(trainableParams))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model) 
    manager    = tf.train.CheckpointManager(checkpoint, os.path.join(args.output_folder,"ckpts"), max_to_keep=1)
    
    loss_ae = tf.keras.metrics.Mean()
    loss_all = []
    print('training...')
    while int(checkpoint.step) < args.epochs:
        for i, data_batch in enumerate(tr_data):
            loss = model.train(data_batch['image'], optimizer)
            
            loss_ae(loss)
        loss_all.append(loss_ae.result())

        if (int(checkpoint.step)+1) % 10 == 0:
            print('epoch {}'.format(int(checkpoint.step)+1))

        if (int(checkpoint.step)+1) % args.save_every == 0 or ((int(checkpoint.step)+1)==args.epochs):
            save_path = manager.save(checkpoint_number=int(checkpoint.step)+1)
            print("Saved checkpoint for step {}: {}".format(int(checkpoint.step)+1, save_path))
            
            # saving weights as npy
            weights_enc=[]
            for w in model.encoder.variables:
                weights_enc.append(w.numpy())
            weights_dec=[]
            for w in model.decoder.variables:
                weights_dec.append(w.numpy())
            np.save(os.path.join(args.output_folder,'enc_weights.npy'),weights_enc)
            np.save(os.path.join(args.output_folder,'dec_weights.npy'),weights_dec)

        
        # increse checkpoint  
        checkpoint.step.assign_add(1)

    for i, data_batch in enumerate(te_data):
        real_imgs = data_batch['image'].numpy()
        decoded_imgs = model.reconstruct(real_imgs).numpy()
        break
    plot_digits(real_imgs, decoded_imgs, 7, args)

    plt.plot(loss_all)
    plt.savefig(os.path.join(args.output_folder,'loss_ae.pdf'))
    plt.close()

if __name__ == '__main__':
    train()
