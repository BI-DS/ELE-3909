import numpy as  np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
import os
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_grid(images,N=10,C=10,figsize=(24., 28.), plot_name='../output/vade/generative_model.png'):
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(N, C),  
                     axes_pad=0,  # pad between Axes in inch.
                     )
    for ax, im in zip(grid, images):
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(plot_name)

def nice_scatter(z_tsne, y_sample, args, acc):
    print('plotting nice scatter...')
    COLORS=[[1.0000, 0,      0     ],
            [0,      1.0000, 0     ],
            [0,      0,      1.0000],
            [1.0000, 0,      1.0000],
            [0.9569, 0.6431, 0.3765],
            [0.4000, 0.8039, 0.6667],
            [0.5529, 0.7137, 0.8039],
            [0.8039, 0.5882, 0.8039],
            [0.7412, 0.7176, 0.4196],
            [0,      0,      0     ]]

    fig, ax = plt.subplots()
    for i, ((x,y),) in enumerate(zip(z_tsne)):
        #rot = random.randint(0,0) # in case you want randomly rotated numbers
        rot = 0
        ax.text(x, y, y_sample[i], color=COLORS[y_sample[i]-1], ha="center", va="center", rotation = rot, fontsize=5)
        ax.plot(x,y, alpha=0.0)
        ax.axis('off')
        plt.title('Latent Space \n cluster acc {:.4f}'.format(acc))
    plt.savefig(os.path.join(args.output_folder,'latent_space.pdf'))
    plt.close()

def plot_digits(x_real, x_gen, n, args):
  plt.figure(figsize=(20, 4))
  for i in range(n):
      # Display original
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(x_real[i])
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      # Display reconstruction
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(x_gen[i])
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  plt.savefig(os.path.join(args.output_folder,'gen_digits.pdf'))
  plt.close()

def image_processing(row):
  x_train = row['image']/255
  
  row['image'] = x_train
  return row

def cluster_acc(Y_pred, Y):
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = np.transpose(np.asarray(linear_assignment(w.max() - w)))
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w
