import models as m
from params import *
import data_load as adl
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from sklearn.metrics import silhouette_score as sil

filepath = "model100.h5"
results = []

# Models Weights Record

model_name_softmax = "model-"+str(int(train_pct*100))+"-"+str(n_centers_per_class)+"-softmax.h5"
model_name_gauss_kmeans = "model-"+str(int(train_pct*100))+"-"+str(n_centers_per_class)+"-gauss-kmeans.h5"
model_name_gauss_kmedoids = "model-"+str(int(train_pct*100))+"-"+str(n_centers_per_class)+"-gauss-kmedoids.h5"
model_name_gauss_no_init = "model-"+str(int(train_pct*100))+"-"+str(n_centers_per_class)+"-gauss-no-init.h5"


# Callbacks Setup

cbs_softmax = [m.EarlyStopping(monitor='val_loss', patience=patience), 
                m.ModelCheckpoint(model_name_softmax, monitor='val_loss', 
                verbose=0, save_best_only=True, mode='min')]

cbs_gauss_kmeans = [m.EarlyStopping(monitor='val_loss', patience=patience), 
                    m.ModelCheckpoint(model_name_gauss_kmeans, monitor='val_loss', 
                    verbose=0, save_best_only=True, mode='min')]

cbs_gauss_kmedoids = [m.EarlyStopping(monitor='val_loss', patience=patience), 
                        m.ModelCheckpoint(model_name_gauss_kmedoids, monitor='val_loss', 
                        verbose=0, save_best_only=True, mode='min')]

cbs_gauss_no_init = [m.EarlyStopping(monitor='val_loss', patience=patience), 
                        m.ModelCheckpoint(model_name_gauss_no_init, monitor='val_loss', 
                        verbose=0, save_best_only=True, mode='min')]

# Dataset Setup

if dataset == "CIFAR-10":
    x_train, x_test, y_train, y_test = adl.load_CIFAR10()
    n_classes = 10
else:
    x_train, x_test, y_train, y_test = adl.load_CIFAR100()
    n_classes = 100

x_train_pct, y_train_pct = m.sample_train(x_train, y_train, train_pct)
m.print_params(feature_extractor, embedding_dim, n_centers_per_class, n_classes, 
                lr, sigma, batch_size, epochs, dataset, input_shape, patience)


# Training Models

''' Softmax Model / Plain Model.
    Without Initialization.
    With Inverse Kernel.
'''
rbf_model, softmax_model, embeddings = m.construct_models(feature_extractor, embedding_dim, 
                                                            n_centers_per_class, n_classes, lr, sigma)
history_softmax = softmax_model.fit(x_train_pct, y_train_pct,
            batch_size=  batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks = cbs_softmax)

del rbf_model
del softmax_model
del embeddings


''' Softmax Model.
    Without Initialization.
    With Gauss Kernel.
'''
rbf_model, softmax_model, embeddings = m.construct_models(feature_extractor, embedding_dim, 
                                                            n_centers_per_class, n_classes, lr, 
                                                            sigma, kernel_type = "gauss")
history_gauss_no_init = rbf_model.fit(x_train_pct, y_train_pct,
                batch_size=  batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test),
                callbacks = cbs_gauss_no_init)

del rbf_model
del softmax_model
del embeddings


''' Pre-trained Softmax Model.
    With K Means Initialization.
    With Gauss Kernel.
'''
rbf_model, softmax_model, embeddings = m.construct_models(feature_extractor, embedding_dim, 
                                                            n_centers_per_class, n_classes, 
                                                            lr, sigma, kernel_type = "gauss")
softmax_model.load_weights(model_name_softmax)
init_keys = m.get_initial_weights(embeddings, x_train_pct, y_train_pct, n_centers_per_class, n_classes, embedding_dim, init_method= "KMEANS")
rbf_model.layers[-1].set_keys(init_keys)

history_gauss_kmeans = rbf_model.fit(x_train_pct, y_train_pct,
                batch_size=  batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test),
                callbacks = cbs_gauss_kmeans)


del rbf_model
del softmax_model
del embeddings


''' Softmax Model.
    With K-Medois Initialization.
    With Gauss Kernel
'''
rbf_model, softmax_model, embeddings = m.construct_models(feature_extractor, embedding_dim, 
                                                            n_centers_per_class, n_classes, 
                                                            lr, sigma, kernel_type = "gauss")
softmax_model.load_weights(model_name_softmax)
init_keys = m.get_initial_weights(embeddings, x_train_pct, y_train_pct, n_centers_per_class, 
                                    n_classes, embedding_dim, init_method= "KMEDOIDS")
rbf_model.layers[-1].set_keys(init_keys)

history_gauss_kmedoids = rbf_model.fit(x_train_pct, y_train_pct,
                batch_size=  batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test),
                callbacks = cbs_gauss_kmedoids)


# Record of Highest Validation Accuracies

highest_softmax = np.max(history_softmax.history["val_acc"])
highest_gauss_kmeans = np.max(history_gauss_kmeans.history["val_acc"])
highest_gauss_no_init = np.max(history_gauss_no_init.history["val_acc"])
highest_gauss_kmedoids = np.max(history_gauss_kmedoids.history["val_acc"])

results.append({"softmax-"+dataset+"-"+str(int(n_centers_per_class)): highest_softmax,
                "gauss-kmeans-"+dataset+"-"+str(int(n_centers_per_class)): highest_gauss_kmeans,
                "gauss-kmedoids-"+dataset+"-"+str(int(n_centers_per_class)): highest_gauss_kmedoids
                "gauss-no-init-"+dataset+"-"+str(int(n_centers_per_class)): highest_gauss_no_init })

del rbf_model
del softmax_model
del embeddings

with open("Train_Results_Multiple_"+feature_extractor+str(int(train_pct*100))+"_trial_"+str(i), "wb") as f:
    pickle.dump(results, f)
