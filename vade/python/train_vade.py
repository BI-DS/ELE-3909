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
from vade import VADE

from utils import image_processing, cluster_acc, nice_scatter
import random
from sklearn.manifold import TSNE

def train():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--latent_dim", default= 10, help="Dimensionality of the latent space", type=int)
    parser.add_argument("--epochs", default= 100, help="No of samples that the encoder draws", type=int)
    parser.add_argument("--save_every", default= 100, help="No of epochs to save cktps", type=int)
    parser.add_argument("--eval_every", default= 20, help="No of epochs to eval cluster acc", type=int)
    parser.add_argument("--K", default= 10, help="No of clusters", type=int)
    parser.add_argument("--alpha", default= 35, help="Scaling for log p(x|z)", type=int)
    parser.add_argument("--batch_size", default= 256, help="No. of samples that the encoder draws", type=int)
    parser.add_argument("--dset", default= 'mnist', help="Data set")
    parser.add_argument("--output_folder", default= '../output/vade', help="Output folder to save outputs")
    parser.add_argument("--load_weights", action='store_true',
            help="Whether to load pretrained weigths and use mu and var fitted by a GMM as starting points")
    
    args = parser.parse_args()
    print (args)
    print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
    
    try:
        os.mkdir(args.output_folder)
    except OSError:
        print("Creation of the directory {}  failed".format(args.output_folder))
    else:
        print("Successfully created the directory {}".format(args.output_folder))
    
    tr_data, info = tfds.load('mnist',split='train', with_info=True)
    te_data = tfds.load('mnist',split='test')
    
    BUFFER   = info.splits['train'].num_examples
    AUTOTUNE = tf.data.AUTOTUNE
    tr_data  = tr_data.map(image_processing).cache().shuffle(BUFFER).batch(args.batch_size,drop_remainder=True).prefetch(AUTOTUNE)

    # define model
    model = VADE(EncMNIST, DecMNIST, alpha=args.alpha, batch_size=args.batch_size, K=args.K, latent_dim=args.latent_dim)
    
    # load pretrained weights
    if args.load_weights:
        model.load_weights()

        print('Using mu and var fitted by a GMM as starting points...')
        all_z = []
        for i, data_batch in enumerate(tr_data):
            z = model.draw_z(data_batch['image'])
            all_z.append(z)
        z = tf.concat(all_z,axis=0)
        model.gmm.fit_gmm_params(z)

    # get data out of DataLoader object 
    ytrue_all = []
    x_all = []
    for i, data_batch in enumerate(tr_data):
        ytrue_all.append(data_batch['label'])
        x_all.append(data_batch['image'])
    y = np.concatenate(ytrue_all,axis=0)
    x = np.concatenate(x_all,axis=0)
    
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.weights])
    print('VADE model has {} trainable params'.format(trainableParams))
    
    # define optimizer and manager to save checkpoints 
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model) 
    manager    = tf.train.CheckpointManager(checkpoint, os.path.join(args.output_folder,"ckpts"), max_to_keep=1)
    
    llik = tf.keras.metrics.Mean()
    kl   = tf.keras.metrics.Mean()
    entropy = tf.keras.metrics.Mean()
    
    acc_all  = []
    llik_all = []
    kl_all = []
    entropy_all = []
    print('training...')
    while int(checkpoint.step) < args.epochs:
        for i, data_batch in enumerate(tr_data):
            losses = model.train(data_batch['image'], optimizer)
        
            llik(losses['llik'])
            kl(losses['kl'])
            entropy(losses['entropy'])

        llik_all.append(llik.result())
        kl_all.append(kl.result())
        entropy_all.append(entropy.result())

        if (int(checkpoint.step)+1) % args.eval_every == 0 or ((int(checkpoint.step)+1)==args.epochs):
            print('============> epoch {}: llik {:.2f}, kl {:.2f}, and entropy {:.2f}'.format(int(checkpoint.step)+1,tf.reduce_mean(losses['llik']), tf.reduce_mean(losses['kl']), tf.reduce_mean(losses['entropy'])))
            print('evaluating cluster accuracy...')
            z = model.draw_z(x)
            gamma,_ = model.gmm(z,batch_size=z.shape[0])
            acc = cluster_acc(np.argmax(gamma,axis=1), y)
            acc_all.append(acc[0])
            print('cluster accuracy {:.4f}'.format(acc[0]))

        if (int(checkpoint.step)+1) % args.save_every == 0 or ((int(checkpoint.step)+1)==args.epochs):
            save_path = manager.save(checkpoint_number=int(checkpoint.step)+1)
            print("Saved checkpoint for step {}: {}".format(int(checkpoint.step)+1, save_path))
        
        # increse checkpoint  
        checkpoint.step.assign_add(1)

    idx = random.sample(range(z.shape[0]),2000)
    # dimensionality reduction
    z_2d = TSNE(n_components=2,n_jobs=-1).fit_transform(z.numpy()[idx])
    # plot a nice scatter
    nice_scatter(z_2d, y[idx], args, acc_all[-1])

    # plot losses and acc
    plt.plot(acc_all)
    plt.title('acc {:.4f}'.format(acc_all[-1]))
    plt.savefig(os.path.join(args.output_folder,'acc_'+str(args.alpha)+'.pdf'))
    plt.close()

    plt.plot(llik_all)
    plt.savefig(os.path.join(args.output_folder,'llik.pdf'))
    plt.close()
    
    plt.plot(kl_all)
    plt.savefig(os.path.join(args.output_folder,'kl.pdf'))
    plt.close()
    
    plt.plot(entropy_all)
    plt.savefig(os.path.join(args.output_folder,'entropy.pdf'))
    plt.close()

if __name__ == '__main__':
    train()
