from som import SOM
import numpy as np
import matplotlib.pyplot as plt

def main():
    # initiate obj
    som_dim = (60,90)

    input_dim = (100,3)
    som = SOM(som_dim, 
              input_dim,
              output_file='ELE3909'
            )
    
    # plotting som at initialization
    plt.figure()
    plt.imshow(som.som)
    plt.savefig(som.path + '/som_random.pdf')
    plt.close()

    # training
    dataset = np.random.random(input_dim)
    epochs = 200 
    eta0 = 0.1
    eta_decay=0.05
    sgm0 = 20 
    sgm_decay=0.05
   
    # XXX choose method = 'prioritized' for training P-SOM instead.
    # XXX in that case specify weights for each of the variable in input vectors
    som.training(dataset=dataset
                ,epochs=epochs
                ,eta0=eta0
                ,eta_decay=eta_decay
                ,sgm0=sgm0
                ,sgm_decay=sgm_decay
                ,print_every = 5
                ,distance_method='regular'
                ,weights = None
                )

    # fit kmeans at the top of som for different number of k
    #som.cluster_som(range_k=[6,10,14,18,22,26,30], batch_size=10)
    # for simplicity I choose only 2 values k = [3,7]
    som.cluster_som(range_k=[3, 8], batch_size=10)

    plt.figure()
    plt.imshow(som.som)
    labels = np.ravel(som.som_labels,order='F')
    
    # brute force approach just to check it works 
    for i, txt in enumerate(labels):
        # transform idx_bu into 2D index
        x,y = np.unravel_index(i,[60,90],'F')
        plt.annotate(txt, (y,x), size=4)
    plt.savefig(som.path + '/som_trained.pdf')
    
    print ('final som has a quatization error of %.2f' % som.qerror)
    
    
if __name__=="__main__":
    main()
