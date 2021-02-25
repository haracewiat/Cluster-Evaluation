# # Feature extraction
# from tensorflow import keras
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.models import Model

# # process_data
# import os
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from tensorflow.keras.applications.vgg16 import preprocess_input


from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import MeanShift
from scipy.spatial import KDTree

from minisom import MiniSom
import matplotlib.pyplot as plt


from time import sleep
import progressbar
import pickle
import time
from sklearn.model_selection import KFold


class ClusterEvaluator:

    def __init__(self):
        
        self.intermediate_layer_model = None
        self.data = {}
        

    def import_vgg16(self):

        # Load VGG16 
        from tensorflow.keras.applications.vgg16 import VGG16
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.applications.vgg16 import preprocess_input
        from tensorflow.keras.models import Model

        model = VGG16(weights='imagenet', include_top= True )
        layer_name = 'fc2'
        self.intermediate_layer_model = keras.Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

        print("imported vgg16")

        



    
    def load_data(self):

        # Compile a list of classes
        data_path = os.getcwd() + "\\data\\dataset\\"
        contents = os.listdir(data_path)
        classes = [class_name for class_name in contents if os.path.isdir(data_path + class_name)]



        # For each class, check if there exists a dataframe with featurized vectors
        
        import pandas as pd 

        for class_name in classes:
            
            print("DEBUG: Loading category ", class_name) 
            # widgets = [
            #     '\x1b[33m{}\x1b[39m '.format(each),
            #     progressbar.Bar(marker='\x1b[32m#\x1b[39m'),
            #     progressbar.Percentage(),
            #     ' [', progressbar.Timer(), '] ',
            #     ' [', progressbar.ETA(), '] ',
            # ]


            # See if there's a file
            processed_data_path = os.getcwd() + "\\data\\processed_data\\"
            file_path = processed_data_path + class_name + ".csv"

            # If dataframe was found, load it 
            if os.path.isfile(file_path):
                
                dataframe = pd.read_csv(file_path)    
                self.data[class_name] = {'features': dataframe[dataframe.columns[:-1]], 'labels': dataframe[dataframe.columns[-1]], 'all': dataframe}
                

            # If there is no dataframe, create one
            else:

                if not os.path.exists(processed_data_path):
                    os.makedirs(processed_data_path)

                class_path = data_path + class_name
                self.data[class_name] = self.extract_features(class_path, class_name)

                # Combine features and labels into one dataframe
                dataframe = pd.DataFrame(self.data[class_name]['features'])
                dataframe[len(dataframe.columns) + 1] = self.data[class_name]['labels']

                

                dataframe.to_csv(file_path, index=False, header=False)
            



    def extract_features(self, class_path, class_name):
            
        files = os.listdir(class_path)
        
        # Temporary (move elsewhere)
        images = []
        batch = []
        labels = []

        j = 0


        widgets=[
            '\x1b[33m{}\x1b[39m '.format(class_name),
            progressbar.Bar(marker='\x1b[32m#\x1b[39m'),
            progressbar.Percentage(),
            ' [', progressbar.Timer(), '] ',
            ' [', progressbar.ETA(), '] ',
        ]

        with progressbar.ProgressBar(widgets=widgets, max_value=len(files)) as bar:

            for ii, file in enumerate(files, 1): #Loop for the imgs inside the folders
                # Load the images and resize it to 224X224(VGG-16 size)
                img = image.load_img(os.path.join(class_path, file), target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                # Extract features using the VGG-16 structure
                if self.intermediate_layer_model is None:
                    print("Importing")                    
                    self.import_vgg16()
                    features = self.intermediate_layer_model.predict(x)
                    # Append features and labels
                    batch.append(features[0])
                    
                    labels.append(file + ';' + str(j))
                else: 
                    features = self.intermediate_layer_model.predict(x)
                    # Append features and labels
                    batch.append(features[0])
                    
                    labels.append(file + ';' + str(j))

                j = j + 1

                bar.update(j)

            
        np_batch = np.array(batch)
        np_labels = np.array(labels)

        X_train = np_batch
        X_test = np_labels

        # Standardize
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(X_train) 

        d = {'features': standardized_data, 'labels': X_test}

        return d

    def evaluate_models(self):


        # VARIABLES
        # Create a split
        # (To imitate the actual data split when in the actual model)
        kf = KFold(n_splits=5, shuffle=True, random_state=1)


        # Bandwidths
        quantiles = [ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9, 1.]
        estimated_bandwidths = np.zeros((len(quantiles), kf.get_n_splits() + 1))
        estimated_bandwidths[:,0] = np.ravel(quantiles)    # Add all quantiles to the dataframe
        estimated_bandwidths_times = np.copy(estimated_bandwidths)

        # Models
        bandwidths = []
        meanshift_models = np.zeros((len(quantiles), kf.get_n_splits() + 1), dtype=object)
        
        scores = np.copy(meanshift_models)
        scores[:,0] = np.ravel(quantiles)
        
        cluster_centers = np.copy(scores)
        times = np.copy(scores)
        
        


        for category in self.data.keys():

            print("DEBUG: Starting evaluating category ", category)

            # Paths
            results_path = os.getcwd() +  "\\data\\results\\mean-shift\\" + category
            
            if not os.path.exists(results_path):
                os.makedirs(results_path)

    
            # Bind together the features and labels dataframes
            features = self.data[category]['features']
            labels = self.data[category]['labels']

        
                
            # Test 5 different splits to get accurate results 
            for split_index, split in enumerate(kf.split(features)):

                print("DEBUG: Starting split", str(split_index))

                # 80% of shuffled data
                train_index = split[0]
                training_data = features.iloc[train_index] 


                # GET THE ESTIMATED BANDWIDTH

                # Add estimated bandwidth for each split to the dataframe
                bandwidths = []

                for quantile_index in range(len(quantiles)):
                    print("     DEBUG: Starting quantile", str(quantile_index))
                    start_time = time.time()
                    estimated_bandwidth = estimate_bandwidth(training_data, quantile=quantiles[quantile_index])
                    finish_time = time.time() - start_time
                    print("            Time taken:", finish_time)

                    estimated_bandwidths[quantile_index][split_index + 1] = estimated_bandwidth
                    bandwidths.append(estimated_bandwidth)

                    estimated_bandwidths_times[quantile_index][split_index + 1] = finish_time


                meanshift_models[:,0] = np.ravel(bandwidths) # Fill first column with the bandwidths
                # -------------------------------------------
                
        
                # GET THE MODELS
                # For each of the bandwidths, create a means shift model
                
                
                for bandwidth_index in range(len(bandwidths)):
                    print("         DEBUG: Starting bandwidth", str(bandwidth_index), ":", bandwidths[bandwidth_index])


                    start_time = time.time()
                    model = MeanShift(bandwidths[bandwidth_index]).fit(training_data)
                    finish_time = time.time() - start_time
                    print("                Time taken:", finish_time)


                    meanshift_models[bandwidth_index][split_index + 1] = model

                    # Evaluate model (because you know the training data)
                    scores[bandwidth_index][split_index + 1] = self.evaluate_cluster(training_data, model)
                    # Get the number of clusters 
                    cluster_centers[bandwidth_index][split_index + 1] = len(model.cluster_centers_)
                    
                    # Store the time 
                    times[bandwidth_index][split_index + 1] = finish_time

            

            
                # SAVE THE MODELS
                with open(results_path + '\\models.pkl', 'wb') as f:
                    pickle.dump(meanshift_models, f)

                # with open(results_path + 'meanshift_models.pkl', 'rb') as f:
                #     test = pickle.load(f)


                # SAVE THE AVERAGE SCORES

                average_scores = np.zeros((scores.shape[0], 4))
                average_scores[:,0] = scores[:,0]

                count_davies_bouldin = 0
                count_silhouette = 0
                count_calinski_harabasz = 0

                for row in range(scores.shape[0]):

                    for score in scores[row, 1:]:
                        
                        if score['davies_bouldin'] is not None:
                            average_scores[row, 1] = average_scores[row, 1] + score['davies_bouldin'] 
                            count_davies_bouldin = count_davies_bouldin + 1
                        
                        if score['silhouette'] is not None:
                            average_scores[row, 2] = average_scores[row, 2] + score['silhouette']
                            count_silhouette = count_silhouette + 1

                        if score['calinski_harabasz'] is not None:
                            average_scores[row, 3] = average_scores[row, 3] + score['calinski_harabasz'] 
                            count_calinski_harabasz = count_calinski_harabasz + 1

                    average_scores[row, 1] = average_scores[row, 1] / count_davies_bouldin
                    average_scores[row, 2] = average_scores[row, 2] / count_silhouette
                    average_scores[row, 3] = average_scores[row, 3] / count_calinski_harabasz


                column_headers = ["Quantile", "davies_bouldin", "silhouette", "calinski_harabasz"]
                df_average_scores = pd.DataFrame(average_scores, columns=column_headers)

                df_average_scores.to_csv(results_path + "\\average_scores.csv", index=False, header=True)



                # STORE THE RESULTS -------------------------------------------------------------------
                column_headers = ["Split " + str(_) for _ in range(kf.get_n_splits())]
                column_headers.insert(0, "Quantile")

                # SAVE THE NUM. OF CLUSTERS
                df_cluster_centers = pd.DataFrame(cluster_centers, columns=column_headers)
                df_cluster_centers.to_csv(results_path + "\\cluster_centers.csv", index=False, header=True)

                # SAVE THE TIMES
                df_times = pd.DataFrame(times, columns=column_headers)
                df_times.to_csv(results_path + "\\times.csv", index=False, header=True)

                # SAVE THE ESTIMATED BANDWIDTH
                df_estimated_bandwidths = pd.DataFrame(estimated_bandwidths, columns=column_headers)
                df_estimated_bandwidths.to_csv(results_path + "\\estimated_bandwidths.csv", index=False, header=True)

                # SAVE THE ESTIMATE BANDWIDTH TIME
                df_estimated_bandwidths_times = pd.DataFrame(estimated_bandwidths_times, columns=column_headers)
                df_estimated_bandwidths_times.to_csv(results_path + "\\estimated_bandwidths_times.csv", index=False, header=True)
        


            

    def evaluate_cluster(self, training_data, model):

        scores = {
            "davies_bouldin"    : None,
            "silhouette"        : None,
            "calinski_harabasz" : None
        }

        # Best score is 0
        try: 
            scores["davies_bouldin"] = metrics.davies_bouldin_score(training_data, model.labels_)
        except ValueError:
            pass

        # Worst score is -1, best score is 1. 0 means overlapping clusters
        try: 
            scores["silhouette"] = metrics.silhouette_score(training_data, model.labels_)
        except ValueError:
            pass

        # The higher score the better
        try: 
            scores["calinski_harabasz"] = metrics.calinski_harabasz_score(training_data, model.labels_)
        except ValueError:
            pass


        return scores


    def evaluate_clusters_legacy(self, bandwidths):

        davies_bouldin_scores = []
        silhouette_scores = []
        calinski_harabasz_scores = []


        widgets=[
                '\x1b[33mEvaluating \x1b[39m ',
                progressbar.Bar(marker='\x1b[32m#\x1b[39m'),
                progressbar.Percentage(),
                ' [', progressbar.ETA(), '] ',
                ' [', progressbar.Timer(), '] '
            ]



        with progressbar.ProgressBar(widgets=widgets, max_value=len(bandwidths)) as bar:

            for i in range(len(bandwidths)):

                # print("Bandwidth: {}".format(bandwidths[i]))

                # Apply meanshift
                ms = MeanShift(bandwidths[i]).fit(self.data)
                
                # # Show the cluster centers
                # cluster_centers = ms.cluster_centers_
                # print("Clusters found: {}".format(cluster_centers.shape[0]))

                # # Print cluster centers
                # kdt_tree = KDTree(data)
                # centers = []

                # for i in range(len(cluster_centers)):   
                #     dist, ind = kdt_tree.query(cluster_centers[i], k=1)
                #     centers.append(ind)

                # print("Cluster centers: {}".format(np.sort(centers)))

                # Evaluation 

                # Best score is 0
                try: 
                    davies_bouldin_scores.append(metrics.davies_bouldin_score(self.data, ms.labels_))
                except ValueError:
                    davies_bouldin_scores.append(None)

                #print("Davies-Bouldin Index: {}".format(davies_bouldin_scores[i]))
                
                

                # # Worst score is -1, best score is 1. 0 means overlapping clusters
                try: 
                    silhouette_scores.append(metrics.silhouette_score(self.data, ms.labels_))
                except ValueError:
                    silhouette_scores.append(None)

                #print("Silhouette Coefficient : {}".format(silhouette_scores[i]))
                

                # # The higher score the better
                try: 
                    calinski_harabasz_scores.append(metrics.calinski_harabasz_score(self.data, ms.labels_))
                except ValueError:
                    calinski_harabasz_scores.append(None)

                #print("Calinski-Harabasz Index : {}".format(calinski_harabasz_scores[i]))

                #print("\n\n\n")

                bar.update(i)

        np.savetxt('scores.csv', [p for p in zip(bandwidths, davies_bouldin_scores, silhouette_scores, calinski_harabasz_scores)], delimiter=',', fmt='%s')


    def evaluate_clusters_som(self):
 
        test_range =  range(1,11)
        # som_shapes = [(8, 8), (5, 5), (3, 3)]

        # Create 10 x 10 matrix
        davies_bouldin_scores2 = [[0 for x in range(test_range[-1])] for y in range(test_range[-1])] 
        silhouette_scores2 = [[0 for x in range(test_range[-1])] for y in range(test_range[-1])] 
        calinski_harabasz_scores2 = [[0 for x in range(test_range[-1])] for y in range(test_range[-1])] 

        widgets=[
                        '\x1b[33mEvaluating \x1b[39m ',
                        # ' [', progressbar.Timer(), '] ',
                        progressbar.Bar(marker='\x1b[32m#\x1b[39m'),
                        progressbar.Percentage(),
                        ' [', progressbar.ETA(), '] ',
                    ]

        with progressbar.ProgressBar(widgets=widgets, max_value=(test_range[-1] * test_range[-1])) as bar:
            for row in test_range:
                for column in test_range:

                    som = MiniSom(row, column, self.data.shape[1], sigma=0.5, learning_rate=0.5)
                    som.random_weights_init(self.data)

                    winner_coordinates = np.array([som.winner(x) for x in self.data]).T
                    cluster_index = np.ravel_multi_index(winner_coordinates, (row, column))
                    assigned_clusters = cluster_index
                    
                    print(row, column)
                    # Best score is 0
                    try: 
                        davies_bouldin_scores2[row-1][column-1] = metrics.davies_bouldin_score(self.data, assigned_clusters)
                    except ValueError:
                        davies_bouldin_scores2[row-1][column-1] = None

                    print("Davies-Bouldin Index: {}".format(davies_bouldin_scores2[row-1][column-1]))


                    # # Worst score is -1, best score is 1. 0 means overlapping clusters
                    try: 
                        silhouette_scores2[row-1][column-1] = metrics.silhouette_score(self.data, assigned_clusters)
                    except ValueError:
                        silhouette_scores2[row-1][column-1] = None

                    print("Silhouette Coefficient : {}".format(silhouette_scores2[row-1][column-1]))
                    

                    # # The higher score the better
                    try: 
                        calinski_harabasz_scores2[row-1][column-1] = metrics.calinski_harabasz_score(self.data, assigned_clusters)
                    except ValueError:
                        calinski_harabasz_scores2[row-1][column-1] = None

                    print("Calinski-Harabasz Index : {}".format(calinski_harabasz_scores2[row-1][column-1]))

            bar.update(row*column)
                


        # for n in range(len(som_shapes)):

            # som = MiniSom(som_shapes[n][0], som_shapes[n][1], data.shape[1], sigma=0.5, learning_rate=0.5)
            # som.random_weights_init(data)

            # winner_coordinates = np.array([som.winner(x) for x in data]).T
            # cluster_index = np.ravel_multi_index(winner_coordinates, som_shapes[n])
            # assigned_clusters = cluster_index

            # #  print(cluster_index)

            # cluster_centers = som_shapes[n][0] * som_shapes[n][1]
            # print("Clusters defined: {}".format(cluster_centers))


            # # For each cluster, indicate the center
            # centers = []
            # empty_cluster_centers = 0
            # kdt_tree = KDTree(data)

            # for i in range(cluster_centers):   
                
            #     members = []

            #     # Select all members of the current cluster
            #     for j in range(len(assigned_clusters)):

            #         if assigned_clusters[j] == i:
            #             members.append(data[j])

        
            #     # EDIT Count empty cluster centers
            #     if not members:
            #         empty_cluster_centers = empty_cluster_centers  + 1
            #     else:
                    
            #         # Calculate mean member 
            #         mean_member = np.array(members).mean(axis=0)

            #         # Find the nearest member to the mean member 
            #         dist, ind = kdt_tree.query(mean_member, k=1)
            #         centers.append(ind)


            # print("Clusters found: {}".format(cluster_centers - empty_cluster_centers))
            # print("Cluster centers: {}".format(np.sort(centers)))

            # # Best score is 0
            # print("Davies-Bouldin Index: {}".format(metrics.davies_bouldin_score(data, assigned_clusters)))

            # # Worst score is -1, best score is 1. 0 means overlapping clusters
            # print("Silhouette Coefficient : {}".format(metrics.silhouette_score(data, assigned_clusters)))

            # # The higher score the better
            # print("Calinski-Harabasz Index : {}".format(metrics.calinski_harabasz_score(data, assigned_clusters)))
            # print("\n\n\n")

        np.savetxt('test_scoresSOM1.csv', davies_bouldin_scores2, delimiter=',', fmt='%s')
        np.savetxt('test_scoresSOM2.csv', silhouette_scores2, delimiter=',', fmt='%s')
        np.savetxt('test_scoresSOM3.csv', calinski_harabasz_scores2, delimiter=',', fmt='%s')


    def get_prototypes(self, ms):
        
        # # Show the cluster centers
        cluster_centers = ms.cluster_centers_
        print("Clusters found: {}".format(cluster_centers.shape[0]))

        # Print cluster centers
        kdt_tree = KDTree(data)
        centers = []

        for i in range(len(cluster_centers)):   
            dist, ind = kdt_tree.query(cluster_centers[i], k=1)
            centers.append(ind)

        print("Cluster centers: {}".format(np.sort(centers)))


    def get_prototypes_som(self):
        pass