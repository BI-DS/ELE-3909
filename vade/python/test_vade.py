import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from tensorflow.keras import mixed_precision
import tensorflow_datasets as tfds
#tf.get_logger().setLevel('INFO')

from enc_dec import DecMNIST, EncMNIST
from vade import VADE

from utils import image_processing, cluster_acc, nice_scatter, plot_grid
import random
from sklearn.manifold import TSNE
import time
from datetime import timedelta

def train():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--latent_dim", default= 10, help="Dimensionality of the latent space", type=int)
    parser.add_argument("--epochs", default= 400, help="No of samples that the encoder draws", type=int)
    parser.add_argument("--save_every", default= 400, help="No of epochs to save cktps", type=int)
    parser.add_argument("--eval_every", default= 50, help="No of epochs to eval cluster acc", type=int)
    parser.add_argument("--K", default= 10, help="No of clusters", type=int)
    parser.add_argument("--alpha", default= 35, help="Scaling for log p(x|z)", type=int)
    parser.add_argument("--batch_size", default= 256, help="No. of samples that the encoder draws", type=int)
    parser.add_argument("--dset", default= 'mnist', help="Data set")
    parser.add_argument("--output_folder", default= '../output/vade', help="Output folder to save outputs")
    parser.add_argument("--load_weights", action='store_true',
            help="Whether to load pretrained weigths and use mu and var fitted by a GMM as starting points")
    
    start = time.time()
    args = parser.parse_args()
    print (args)
    print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
    
    try:
        os.mkdir(args.output_folder)
    except OSError:
        print("Creation of the directory {}  failed".format(args.output_folder))
    else:
        print("Successfully created the directory {}".format(args.output_folder))
   
    # load only test set
    tr_data, info = tfds.load('mnist',split='train', with_info=True)
    te_data = tfds.load('mnist',split='test')
    
    BUFFER   = info.splits['train'].num_examples
    AUTOTUNE = tf.data.AUTOTUNE
    tr_data  = tr_data.map(image_processing).cache().shuffle(BUFFER).batch(args.batch_size,drop_remainder=True).prefetch(AUTOTUNE)

    # define model
    model = VADE(EncMNIST, DecMNIST, alpha=args.alpha, batch_size=args.batch_size, K=args.K, latent_dim=args.latent_dim)
    
    # define optimizer and manager to save checkpoints 
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model) 
    manager    = tf.train.CheckpointManager(checkpoint, os.path.join(args.output_folder,"ckpts"), max_to_keep=1)
    
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print("Model restored from {}".format(manager.latest_checkpoint))
    else:
        print("Model does not exist")
    
    # get data out of DataLoader object 
    ytrue_all = []
    x_all = []
    for i, data_batch in enumerate(tr_data):
        ytrue_all.append(data_batch['label'])
        x_all.append(data_batch['image'])
    y = np.concatenate(ytrue_all,axis=0)
    x = np.concatenate(x_all,axis=0)
   
  
    # accuracy
    z = model.draw_z(x)
    gamma,_ = model.gmm(z,batch_size=z.shape[0])
    acc = cluster_acc(np.argmax(gamma,axis=1), y)[0]
    print('cluster accuracy {:.4f}'.format(acc))

    idx = random.sample(range(z.shape[0]),2000)
    # dimensionality reduction
    z_2d = TSNE(n_components=2,n_jobs=-1).fit_transform(z.numpy()[idx])
    # plot a nice scatter
    nice_scatter(z_2d, y[idx], args, acc)

    all_digits = []
    all_pis = []
    print('generating images from all clusters...')
    for i in range(10):
        z_prior = model.gmm.generate_prior(K=i,L=7)
        for k in range(7):
            pi = model.gmm.get_pi(z_prior[k,...],K=i,batch_size=1)
            all_pis.append(pi[0])
        px_z    = model.dec(z_prior)
        x_hat   = px_z.mean()
        all_digits.extend(x_hat)
    plot_grid(all_digits, N=10, C=7, pi=all_pis)

    print('elapsed time: {}'.format(timedelta(seconds=time.time()-start)))

if __name__ == '__main__':
    train()
