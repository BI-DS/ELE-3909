# Self-Organizing Map SOM
import numpy as np
import matplotlib.pyplot as plt
import os

class SOM(object):
    def __init__(self, som_dim, input_dim,output_file='test',som_weights=None, som_labels=None):
        som_rows, som_cols = som_dim
        input_obs, input_features = input_dim

        self.output_file = output_file
        self.path = '../output/' + self.output_file
        
        try:
            os.mkdir(self.path)
        except OSError:
            print ("creation of the directory %s failed" % self.path)
        else:
            print ("successfully created the directory %s" % self.path)

        self.som_rows = som_rows
        self.som_cols = som_cols
        self.som_dim  = input_features
        
        if som_weights is None: 
            # initialize SOM randomly
            self.som = np.random.randn(som_rows, som_cols, input_features)
        elif som_weights is not None:
            # TODO insert checks for dimension (cols, rows, dim)
            print ('loading  pre-trained weights')
            self.som = som_weights

        if som_labels is not None:
            print ('loading  pre-trained som clusters labels')
            self.som_labels = som_labels
            self.k = np.max(som_labels)+1
            print ('som object has %d different clusters'%self.k)

    def get_weights(self, priority_dim, alpha=0.2):
        # priority_dim is a list of two tuples. the first tuple has all priorities
        # for all one-hot-encoding variables and the second tuple has the number of
        # dimensions in each category
        ps, gs = priority_dim
        

        vector_weights = []
        for k,p in enumerate(ps):
            no_gps = len([i for i in ps if i > p])
            weight = 1 + no_gps/alpha
            rep_weights = np.repeat(weight,gs[k]).tolist()
            vector_weights.extend(rep_weights)
        
        return vector_weights

    def getEuclidian(self, x, distance_method='regular', weights=None):
        # transform som to 2 dim list of neurons
        som_2d = np.reshape(self.som,(self.som_rows*self.som_cols, self.som_dim),order='F')

        # repeat input vector x 
        xall = np.tile(x,self.som_rows*self.som_cols).reshape(self.som_rows*self.som_cols,self.som_dim)

        # return a list of length equal to all number of neurons in the som 
        # with Euclidean distances
        if distance_method == 'regular':
            euclidean = np.sqrt(np.sum(np.square(xall-som_2d),axis=1))
        elif distance_method == 'prioritized':
            if weights is None:
                raise AssertionError("a weight vector with equal dimension as input vectors must be passed")
            else:
                if len(weights) != x.shape[0]:
                    raise AssertionError("the weight vector must have equal dimension as the input vector")
                else:
                    euclidean = np.sqrt(np.sum(weights*np.square(xall-som_2d),axis=1))
        else:
            raise AssertionError("valid methods are 'regular' and 'prioritized'")

        return euclidean
    
    def training(self,dataset=None,epochs=None,eta0=None,eta_decay=None,sgm0=None,sgm_decay=None,print_every=2,distance_method='regular',weights=None):
        # define a grid to be used in the Gaussian funtion for self-learning 
        aa = np.linspace(0, self.som_cols-1, self.som_cols)
        bb =  np.linspace(0, self.som_rows-1, self.som_rows)
        xx,yy = np.meshgrid(aa, bb)
        error = np.zeros(epochs)

        for t in range(epochs):
            if (t+1)%print_every == 0:
                print ('training epoch %d out of %d and qerror is %0.2f' % (t+1,epochs,error[t-1]))
            
            # compute the learning rate for the current epoch
            eta = eta0 * np.exp(-t*eta_decay)

            # compute the variance of the Gaussian function for the current epoch
            sgm = sgm0 * np.exp(-t*sgm_decay)

            # consider the width of the Gaussian function as 3 sigma
            width = int(np.ceil(sgm*3.0))

            # full batch training
            # XXX consider a batch traning instead
            sum_distances = 0
            for i in range(dataset.shape[0]):
                #get input vector
                x_vector = dataset[i,:]

                # get Euclidean distance between input vector and each neuron in the SOM
                distances = self.getEuclidian(x_vector,distance_method=distance_method, weights=weights)
                
                # find the BMU
                idx_bmu = np.argmin(distances)
                sum_distances += np.min(distances)

                # transform idx_bu into 2D index
                bmurow, bmucol = np.unravel_index(idx_bmu,[self.som_rows,self.som_cols],'F')

                # generate Gaussian function centered at the location of the BMU
                g = np.exp(-(np.square(xx - bmucol) + np.square(yy - bmurow))/(2*sgm**2))

                # determine the boundary of the local neighberhood for self-learning
                fromrow = max(0,bmurow - width)
                torow   = min(bmurow + width, self.som_rows)
                fromcol = max(0,bmucol - width)
                tocol   = min(bmucol + width, self.som_cols)

                # get neighbors neurons np.random.randn(input_features, som_rows, som_cols)
                # i need to add +1 in python
                neighbors = self.som[fromrow:torow, fromcol:tocol,:]
                shape_neighbors = neighbors.shape # 1. dim, 2. no_rows, 3. no.cols

                T = np.reshape(np.tile(x_vector,(shape_neighbors[0]*shape_neighbors[1],1)),(shape_neighbors[0],shape_neighbors[1],shape_neighbors[2]))

                G = np.repeat(g[fromrow:torow, fromcol:tocol,np.newaxis], self.som_dim, axis=2)

                # updating equation
                neighbors = neighbors + eta * G * (T - neighbors)

                # insert updated witghts into som
                self.som[fromrow:torow,fromcol:tocol,:] = neighbors

            error[t] = sum_distances
        
        plt.figure()
        plt.plot(error)
        plt.title('Quantization error %0.2f'%error[-1])
        plt.xlabel('epoch')
        plt.grid()
        plt.savefig(self.path+'/qerror.pdf')
        plt.close('all')

        self.qerror = error[-1]
    
    def plot_someplanes(self, col_names=None, plot_contours=None, levels = (0.1,0.5), nrows=1, fig_name='some_planes'):
        import itertools
        
        if plot_contours is not None:
            target_planes = [i[1] for i in plot_contours]
            source_planes = [i[0] for i in plot_contours]
        else:
            raise AssertionError('plot_contours must be specified. use component planes functions to plot all component planes!')
        
        ncols = int(np.ceil(len(target_planes)/nrows))
        fig, ax = plt.subplots(nrows=nrows,ncols=ncols)
        cc = 0
        for a in ax.flatten():
            im = a.imshow(self.som[:,:,target_planes[cc]],cmap='seismic')
            a.contour(self.som[:, :, source_planes[cc]], levels, colors='green', linewidths=1.5)

            if col_names is not None:
                a.set_title(col_names[cc],fontsize=10,pad=1)

            a.set_xticks([])
            a.set_yticks([])
            cc+=1

        plt.savefig(self.path+'/'+fig_name+'.pdf')
        plt.close('all')
    
    def component_planes(self,col_names=None, plot_contours=None, priorities=None, levels = (0.1,0.5)):
        import itertools
        
        if plot_contours is not None:
            target_planes = [i[1] for i in plot_contours]
        else:
            target_planes = [] 
        
        # use priorities to plot in title
        if priorities is not None:
            ps, gs = priorities
            priorities_list = [np.tile(ps[i],gs[i]).tolist() for i in range(len(ps))]
            priorities_list = list(itertools.chain(*priorities_list))
        else:
            priorities_list = []

        
        ncols = int(np.ceil(self.som_dim/5.0))
        fig, ax = plt.subplots(nrows=5,ncols=ncols)
        cc = 0
        for a in ax.flatten():
            if cc < self.som_dim:
                im = a.imshow(self.som[:,:,cc],cmap='seismic')
                # plot the contour for the first component plane which is the target variable
                a.contour(self.som[:,:,0], levels,colors='yellow',linewidths=0.5)
                
                if cc in target_planes:
                    for e in (elem for elem in plot_contours if elem[1] == cc):
                        a.contour(self.som[:, :, e[0]], levels, colors='green', linewidths=0.5)

                if col_names is not None:
                    if len(priorities_list) > 0: 
                        a.set_title(col_names[cc]+'-'+str(priorities_list[cc]),fontsize=6.5,pad=1)
                    else:
                        a.set_title(col_names[cc],fontsize=6.5,pad=1)

            if cc >= self.som_dim:
                fig.delaxes(a)
            a.set_xticks([])
            a.set_yticks([])
            cc+=1

        plt.savefig(self.path+'/comp_planes.pdf')
        plt.savefig(self.path+'/comp_planes.jpg')
        plt.close('all')
               
    def cluster_som(self,range_k=[2], batch_size=None,file_name='kmeans', force_labels=False,n_reps=100,levels=(0.1,0.5)):
        from sklearn.cluster import KMeans
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import davies_bouldin_score as db_score

        if hasattr(self, 'som_labels') and force_labels==False:
            print('som object already has labels. set force_labels to TRUE if you want to re-run cluster_som()')
        else:
            # transform som to 2 dim list of neurons
            x = np.reshape(self.som,(self.som_rows*self.som_cols, self.som_dim),order='F')
            
            # loop trhough different values for k and calculate
            # the Davies-Bouldin score
            all_scores = {} 
            all_labels = {}
            print ('fitting %d kmeans models'%len(range_k))
            for k in range_k:
                scores_k = [] 
                labels_k = []
                for i in range(n_reps):
                    cluster = KMeans(n_clusters=k,init='k-means++')
                    cluster.fit(x)

                    labels_k.append(cluster.labels_)
                    scores_k.append(db_score(x, cluster.labels_))
                
                # get min score among all n_reps
                idx_minscore_k = np.argmin(scores_k)
                print ('min DB score %.4f for %d clusters, and max score %.4f'%(scores_k[idx_minscore_k],k,np.max(scores_k)))
                
                all_scores[k] = scores_k[idx_minscore_k]
                all_labels[k] = labels_k[idx_minscore_k]

            # get optimal k and extract cluster labels for the best cluster model
            k = min(all_scores, key=all_scores.get)
            print ('optimal number of clusters is %d'%k)
            labels = all_labels[k]
            self.k = k
            
            # now labels into som shape
            som_labels = np.reshape(labels, (self.som_rows, self.som_cols),order='F')
            self.som_labels = som_labels
            
        fig, ax = plt.subplots()
        ax.imshow(self.som_labels)
        plt.grid(False)
        plt.axis('off')
        plt.title('K-means clusters')
        plt.savefig(self.path+'/'+file_name+'_nolabels.pdf')
        ax.contour(self.som[:,:,0],levels,colors='red',linewidths=0.8)
        plt.close()


        plt.figure()
        plt.imshow(self.som)
        plt.title('Trained SOM')
        plt.savefig(self.path+'/som_trained.pdf')
        labels = np.ravel(self.som_labels,order='F')
        # brute force approach just to check it works 
        for i, txt in enumerate(labels):
            # transform idx_bu into 2D index
            x,y = np.unravel_index(i,[60,90],'F')
            plt.annotate(txt, (y,x),size=4)
        
        plt.title('SOM with kmeans labels')
        plt.savefig(self.path+'/'+file_name+'.pdf')
        plt.close('all')

        return self.som_labels
    
    def clustering_trajectories(self,dataset):
        # XXX dataset must have:
        # student_id in the 1 column
        # time in the 2 column
        # student features from the 3 column
        # it must be sorted by student_id, time
        
        # first check that the som object has labels
        if hasattr(self, 'som_labels') == False:
            raise AssertionError('som object doesnt have labels. run cluster_som() first')

        cluster_trajectories = np.zeros(dataset.shape[0])

        student_id = dataset[:,0]
        time = dataset[:,1]
        
        for i in range(dataset.shape[0]):
            x_vector = dataset[i,2:]    
                
            # get Euclidean distance between input vector and each neuron in the SOM
            distances = self.getEuclidian(x_vector)
            
            # find the BMU
            idx_bmu = np.argmin(distances)
                
            # transform idx_bu into 2D index
            bmurow, bmucol = np.unravel_index(idx_bmu,[self.som_rows,self.som_cols],'F')

            # get cluster at time = time
            cluster_trajectories[i] = self.som_labels[bmurow, bmucol]

        # append cluster trajectories to dataset
        x = np.c_[cluster_trajectories.astype(int),dataset]
                 
        return x 
    
    def spm(self,trajectories,file_name='output_file'):
        # XXX dataset must have:
        # clustering trayectories 1. col
        # student_id in the 2. col
        # time in the 3. column
        # student features from the 4 column
        # it must be sorted by student_id, time
        if os.path.exists("../output/"+self.output_file +"/input_spm_"+file_name+".txt"):
            os.remove("../output/"+self.output_file +"/input_spm_"+file_name+".txt")
        
        f = open("../output/"+self.output_file +"/input_spm_"+file_name+".txt","a" ) 

        i=1
        ct = []
        while i < trajectories.shape[0]:
            prev_id = trajectories[i-1,1]
            current_id = trajectories[i,1]
            if prev_id == current_id:
                ct.append(trajectories[i-1,0])
            else:
                ct.append(trajectories[i-1,0])
                string = ' -1 '.join(np.asarray(ct).astype(str)) + ' -1 -2'
                f.write(string)
                f.write("\n")
                ct = []
            
            # this  is for the last line
            if (i+1) == trajectories.shape[0]:
                ct.append(trajectories[i,0])
                string = ' -1 '.join(np.asarray(ct).astype(str)) + ' -1 -2'
                f.write(string)
                f.write("\n")

            i+=1
        f.close()
        str_spm = "java -jar ../spm/spmf.jar run CM-SPAM " + self.path +"/input_spm_"+file_name+".txt "\
                 +self.path +"/"+file_name+".txt 0.06 " 
        
        os.system(str_spm)

    def salient_dimensions(self,x_dic,var_names=None,z=2,file_name='salient_dimensions',print_delta=False):
        '''
        x_dic: dictionary with numpy matrix of features and cluster labels in the first column
        var_names: list of strings with all features names in the same order as in the x_dic 
        '''
        if os.path.exists(self.path +"/"+file_name+".txt"):
            os.remove(self.path +"/"+file_name+".txt")
        f = open(self.path +"/"+file_name+".txt","a" ) 
        
        if os.path.exists(self.path +"/zmost_"+file_name+".txt"):
            os.remove(self.path +"/zmost_"+file_name+".txt")
        f2 = open(self.path +"/zmost_"+file_name+".txt","a" ) 
        
        # number of clusters in som object
        k = self.k

        for ii,key in enumerate(x_dic.keys()):
            x = x_dic[key]
            labels = x[:,0]
            x_nolabels = x[:,1:]
            no_variables = x_nolabels.shape[1]

            delta   = np.zeros((no_variables,k))
            mu_in  = np.zeros((no_variables,k))
            mu_out = np.zeros((no_variables,k))

            for i in range(no_variables):
                feature = x_nolabels[:,i]
                for j in range(k):
                    feature_in  = feature[labels==j]
                    feature_out = feature[labels!=j]

                    mu_in[i,j]    = np.mean(feature_in)
                    mu_out[i,j]   = np.mean(feature_out)
                    delta[i,j] = (mu_in[i,j]-mu_out[i,j])/float(mu_out[i,j])

            if ii==0:
                delta_mus = delta
                mus_in  = mu_in
                mus_out = mu_out
            else:
                delta_mus = np.r_[delta_mus,delta]
                mus_in = np.r_[mus_in,mu_in]
                mus_out = np.r_[mus_out,mu_out]
        
        # average and sigma delta for each cluster (averaging out variables)
        # so we are looking at the distribution for deltas for each cluster
        mu_delta    = np.mean(delta_mus,axis=0)
        sigma_delta = np.std(delta_mus,axis=0)
        
        if print_delta:
            np.savetxt(self.path+"/deltas.csv", np.r_[delta_mus,mu_delta.reshape(1,k),sigma_delta.reshape(1,k)], delimiter=";")
    
        no_variables = delta_mus.shape[0]
        for i in range(no_variables):
            for j in range(k):
                lower_bound = mu_delta[j] - z*sigma_delta[j]
                upper_bound = mu_delta[j] + z*sigma_delta[j]
            
                # excluding delta = -1 because this correspond to empty in-patterns
                if ((delta_mus[i,j] <= lower_bound) or (delta_mus[i,j] >= upper_bound)) and delta_mus[i,j]!=-1:
                    if var_names is not None:
                        if var_names[i] =='on_exchange':
                            str_msg = 'the variable %s is a salient dimension in cluster %d with avg %.2f and others %.2f (both in percentage)' % (var_names[i],j,mus_in[i,j]*100,mus_out[i,j]*100)
                        else:
                            str_msg = 'the variable %s is a salient dimension in cluster %d with avg %.2f and others %.2f' % (var_names[i],j,mus_in[i,j],mus_out[i,j])
                        #print(str_msg)
                        f.write(str_msg)
                        f.write("\n")
                    else: 
                        str_msg = 'the variable %s is a salient dimension in cluster %d with avg %.2f and others %.2f'%(i,j,mus_in[i,j],mus_out[i,j])
                        #print(str_msg)
                        f.write(str_msg)
                        f.write("\n")
                
                z_crit = abs(mu_delta[j] - delta_mus[i,j])/float(sigma_delta[j])
                if var_names[i] =='on_exchange':
                    str_msg = 'the variable %s would be a salient dimension in cluster %d for z value at most %.2f with avg %.2f and others %.2f (both in percet)'%(var_names[i],j,z_crit,mus_in[i,j]*100,mus_out[i,j]*100)
                else:
                    str_msg = 'the variable %s would be a salient dimension in cluster %d for z value at most %.2f with avg %.2f and others %.2f'%(var_names[i],j,z_crit,mus_in[i,j],mus_out[i,j])
                f2.write(str_msg)
                f2.write("\n")
    
    def salient_dimensions_orig(self,x,var_names=None,z=2,file_name='salient_dimensions',print_delta=False):
        '''
        x: numpy matrix of raw data and cluster labels in the first column
        k: number of clusters, format: integer 
        '''
        if os.path.exists(self.path +"/"+file_name+".txt"):
            os.remove(self.path +"/"+file_name+".txt")
        f = open(self.path +"/"+file_name+".txt","a" ) 
        
        if os.path.exists(self.path +"/zmost_"+file_name+".txt"):
            os.remove(self.path +"/zmost_"+file_name+".txt")
        f2 = open(self.path +"/zmost_"+file_name+".txt","a" ) 
        
        # number of clusters in som object
        k = self.k
        labels = x[:,0]
        x_nolabels = x[:,1:]
        no_variables = x_nolabels.shape[1]

        delta_mus = np.zeros((no_variables,k))
        mus_in = np.zeros((no_variables,k))
        mus_out = np.zeros((no_variables,k))

        for i in range(no_variables):
            feature = x_nolabels[:,i]
            for j in range(k):
                feature_in  = feature[labels==j]
                feature_out = feature[labels!=j]
                mu_in  = np.mean(feature_in)
                mu_out = np.mean(feature_out)

                delta_mus[i,j] = (mu_in-mu_out)/float(mu_out)
                mus_in[i,j]    = mu_in
                mus_out[i,j]   = mu_out
        
        # average and sigma delta for each cluster (averaging out variables)
        # so we are looking at the distribution for deltas for each cluster
        mu_delta    = np.mean(delta_mus,axis=0)
        sigma_delta = np.std(delta_mus,axis=0)
        
        if print_delta:
            print(delta_mus)
            print (mu_delta)
            print (sigma_delta)
    
        for i in range(no_variables):
            for j in range(k):
                lower_bound = mu_delta[j] - z*sigma_delta[j]
                upper_bound = mu_delta[j] + z*sigma_delta[j]
            
                if (delta_mus[i,j] <= lower_bound) or (delta_mus[i,j] >= upper_bound):
                    if var_names is not None:
                        if var_names[i] =='on_exchange':
                            str_msg = 'the variable %s is a salient dimension in cluster %d with avg %.2f and others %.2f (both in percentage)' % (var_names[i],j,mus_in[i,j]*100,mus_out[i,j]*100)
                        else:
                            str_msg = 'the variable %s is a salient dimension in cluster %d with avg %.2f and others %.2f' % (var_names[i],j,mus_in[i,j],mus_out[i,j])
                        #print(str_msg)
                        f.write(str_msg)
                        f.write("\n")
                    else: 
                        str_msg = 'the variable %s is a salient dimension in cluster %d with avg %.2f and others %.2f'%(i,j,mus_in[i,j],mus_out[i,j])
                        #print(str_msg)
                        f.write(str_msg)
                        f.write("\n")
                
                z_crit = abs(mu_delta[j] - delta_mus[i,j])/float(sigma_delta[j])
                if var_names[i] =='on_exchange':
                    str_msg = 'the variable %s would be a salient dimension in cluster %d for z value at most %.2f with avg %.2f and others %.2f (both in percet)'%(var_names[i],j,z_crit,mus_in[i,j]*100,mus_out[i,j]*100)
                else:
                    str_msg = 'the variable %s would be a salient dimension in cluster %d for z value at most %.2f with avg %.2f and others %.2f'%(var_names[i],j,z_crit,mus_in[i,j],mus_out[i,j])
                f2.write(str_msg)
                f2.write("\n")
