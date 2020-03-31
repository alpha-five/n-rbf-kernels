import models as m
from params import *
import data_load as adl
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from sklearn.metrics import silhouette_score as sil
import sys


filepath = "model10.h5"
results = []


# Command Line Arguments

if len(sys.argv) > 2:
    feature_extractor = str(sys.argv[1]).upper()
    dataset = str(sys.argv[2]).upper()


# Dataset Setup

if dataset == "CIFAR-10":
  x_train, x_test, y_train, y_test = adl.load_cifar10()
  n_classes = 10
else:
  x_train, x_test, y_train, y_test = adl.load_cifar100()
  n_classes = 100

x_train_pct, y_train_pct = m.sample_train(x_train, y_train, train_pct)


m.print_params(feature_extractor, embedding_dim, n_centers_per_class, n_classes, 
                lr, sigma, batch_size, epochs, dataset, input_shape, patience)

rbf_model, softmax_model, embeddings = m.construct_models(feature_extractor, embedding_dim, 
                                                            n_centers_per_class, n_classes, lr, sigma,  
                                                            kernel_type = "gauss")


# Callbacks Setup

callbacks = [m.EarlyStopping(monitor='val_loss', patience=patience)]
callbacks2 = [m.EarlyStopping(monitor='val_loss', patience=patience), m.ModelCheckpoint(filepath, 
                                                                                        monitor='val_loss', 
                                                                                        verbose=0, 
                                                                                        save_best_only=True, 
                                                                                        mode='min')]

# Training Models

''' Softmax Model / Plain Model.
'''
history_plain = softmax_model.fit(x_train_pct, y_train_pct,
                        batch_size=  batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks = callbacks2)

# Silhouette Index from Embeddings
softmax_model.load_weights(filepath)
y_pred = softmax_model.predict(x_test)
embed = embeddings.predict(x_test)
print("Y_PRED",y_pred)
sil_idx_plain = sil(embed, np.argmax(y_pred, 1))

del embed
del y_pred


''' Pre-trained RBF Model.
    With K-Means Initialization.
'''
init_keys = m.get_initial_weights(embeddings, x_train_pct, y_train_pct, n_centers_per_class, 
                                  n_classes, embedding_dim, init_method= "KMEANS")
rbf_model.layers[-1].set_keys(init_keys)
history_gauss_kmeans = rbf_model.fit(x_train_pct, y_train_pct,
                batch_size=  batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test),
                callbacks = callbacks)

# Silhouette Index from Embeddings
y_pred = rbf_model.predict(x_test)
embed = embeddings.predict(x_test)
print("Y_PRED",y_pred)
sil_idx_gauss = sil(embed, np.argmax(y_pred, 1))

# Evaluation of Silhouette Index Record

results.append({"sil_plain"+dataset+"_"+feature_extractor+str(int(train_pct*100)): sil_idx_plain,
                "sil_gauss"+dataset+"_"+feature_extractor+str(int(train_pct*100)): sil_idx_gauss})

with open("Silhouette_"+feature_extractor+dataset, "wb") as f:
  pickle.dump(results, f)
