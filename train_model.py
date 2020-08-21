from params import *
import models as m
import data_load as adl
import matplotlib.pyplot as plt
import pickle
import sys

filepath       = "model10.h5"
history        = []


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


m.print_params(feature_extractor, embedding_dim, n_centers_per_class, n_classes, lr, sigma, batch_size, epochs, dataset, input_shape, patience)

for i in range(n_trials):

  rbf_model, softmax_model, embeddings = m.construct_models(feature_extractor, embedding_dim, n_centers_per_class, n_classes, lr, sigma)


  # Callbacks Setup

  callbacks = [m.EarlyStopping(monitor='val_loss', patience=patience)]
  callbacks2 = [m.EarlyStopping(monitor='val_loss', patience=patience), m.ModelCheckpoint(filepath, 
                monitor='val_loss', verbose=0, save_best_only=True, mode='min')]


  # Training Models

  ''' Softmax Model / Plain Model
  '''
  history_plain = softmax_model.fit(x_train_pct, y_train_pct,
                          batch_size=  batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_test, y_test),
                          callbacks = callbacks2)


  ''' Pre trained Softmax Model.
      Without Initialization.
      With Inverse Kernel
  '''
  softmax_model.load_weights(filepath)

  history_vk = rbf_model.fit(x_train_pct, y_train_pct,
                  batch_size=  batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks = callbacks)


  ''' Pre trained Softmax Model.
      With K-Means Initialization.
      With Inverse Kernel
  '''
  softmax_model.load_weights(filepath)
  init_keys = m.get_initial_weights(embeddings, x_train_pct, y_train_pct, n_centers_per_class, n_classes, embedding_dim)
  rbf_model.layers[-1].set_keys(init_keys)
  kmeans_acc = rbf_model.evaluate(x_test, y_test)

  history_vk_1 = rbf_model.fit(x_train_pct, y_train_pct,
                  batch_size=  batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks = callbacks)


  ''' Pre trained Softmax Model.
      With K-Medois Initialization.
      With Inverse Kernel
  '''
  softmax_model.load_weights(filepath)
  init_keys = m.get_initial_weights(embeddings, x_train_pct, y_train_pct, n_centers_per_class, n_classes, 
                                    mbedding_dim, init_method= "KMEDOIDS")
  rbf_model.layers[-1].set_keys(init_keys)
  kmedoids_acc = rbf_model.evaluate(x_test, y_test)

  history_vk_2 = rbf_model.fit(x_train_pct, y_train_pct,
                  batch_size=  batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks = callbacks)


  ''' Pre trained Softmax Model.
      With K-Means Initialization.
      With Gauss Kernel.
  '''
  rbf_model, softmax_model, embeddings = m.construct_models(feature_extractor, embedding_dim, n_centers_per_class, n_classes, lr, sigma, kernel_type = "gauss")
  softmax_model.load_weights(filepath)
  init_keys = m.get_initial_weights(embeddings, x_train_pct, y_train_pct, n_centers_per_class, n_classes, embedding_dim, init_method= "KMEANS")
  rbf_model.layers[-1].set_keys(init_keys)

  history_gauss_kmeans = rbf_model.fit(x_train_pct, y_train_pct,
                  batch_size=  batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks = callbacks)


  ''' Non pre trained Model.
      Without Initialization.
      With Gauss Kernel.
  '''
  rbf_model, _, _ = m.construct_models(feature_extractor, embedding_dim, n_centers_per_class, n_classes, lr, sigma, kernel_type = "gauss")

  history_gauss = rbf_model.fit(x_train_pct, y_train_pct,
                  batch_size=  batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks = callbacks)


  ''' Non pre trained Model.
      Without Initialization.
      With Inverse Kernel.
  '''
  rbf_model, _ , _ = m.construct_models(feature_extractor, embedding_dim, n_centers_per_class, n_classes, lr, sigma)

  history_vk_no_init = rbf_model.fit(x_train_pct, y_train_pct,
                  batch_size=  batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks = callbacks)


  # Record of Highest Validation Accuracies

  highest_plain = np.max(history_plain.history["val_acc"])
  highest_vk = np.max(history_vk.history["val_acc"])
  highest_vk_1 = np.max(history_vk_1.history["val_acc"])
  highest_vk_2 = np.max(history_vk_2.history["val_acc"])
  highest_vk_no_init = np.max(history_vk_no_init.history["val_acc"])
  highest_gauss_kmeans = np.max(history_gauss_kmeans.history["val_acc"])
  highest_gauss = np.max(history_gauss.history["val_acc"])

  history.append({"plain": highest_plain,
                  "vk": highest_vk,
                  "vk_kmeans": highest_vk_1,
                  "vk_kmedoids": highest_vk_2,
                  "vk_no_init": highest_vk_no_init,
                  "gauss_means": highest_gauss_kmeans,
                  "gauss": highest_gauss,
                  "k_means": kmeans_acc,
                  "k_medoids": kmedoids_acc})


  with open("Train_Results_"+feature_extractor+str(int(train_pct*100))+"_trial_"+str(i), "wb") as f:
    pickle.dump(history, f)
