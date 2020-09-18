from params import *
import models as m
import data_load as adl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import argparse


def main(args):

    history        = []


    # Command Line Arguments

    feature_extractor = args.feature_extractor
    filepath = args.file_path
    dataset = args.dataset
    n_trials = args.n_trials

    # Dataset Setup

    if dataset == "CIFAR10":
        x_train, x_test, x_val, y_train, y_test, y_val = adl.load_cifar10()
        n_classes = 10
    elif dataset == "CIFAR100":
        x_train, x_test, x_val, y_train, y_test, y_val = adl.load_cifar100()
        n_classes = 100
    elif dataset == "TinyImagenet":
        x_train, x_test, x_val, y_train, y_test, y_val = adl.load_tiny_imagenet()
        n_classes = 200       


    for pct in ["10", "20", "30"]:
            
        x_train_pct, y_train_pct = x_train[pct], y_train[pct]


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
            print("Model with softmax layer")
            history_plain = softmax_model.fit(x_train_pct, y_train_pct,
                                    batch_size=  batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=(x_val, y_val),
                                    callbacks = callbacks2)

            softmax_model.load_weights(filepath)
            error_softmax = rbf_model.evaluate(x_test, y_test, verbose = 0) 


            ''' Pre trained Softmax Model.
                With K-Means Initialization.
                With Gauss Kernel.
            '''
            print("Model with gauss kernel and initialization")
            rbf_model, softmax_model, embeddings = m.construct_models(feature_extractor, embedding_dim, n_centers_per_class, n_classes, lr, sigma, kernel_type = "gauss")
            softmax_model.load_weights(filepath)
            init_keys = m.get_initial_weights(embeddings, x_train_pct, y_train_pct, n_centers_per_class, n_classes, embedding_dim, init_method= "KMEANS")
            rbf_model.layers[-1].set_keys(init_keys)

            history_gauss_kmeans = rbf_model.fit(x_train_pct, y_train_pct,
                            batch_size=  batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_val, y_val),
                            callbacks = callbacks)

            error_rbf_kmeans = rbf_model.evaluate(x_test, y_test, verbose = 0) 


            ''' Non pre trained Model.
                Without Initialization.
                With Gauss Kernel.
            '''
            print("Model with gauss kernel and without initialization")
            rbf_model, _, _ = m.construct_models(feature_extractor, embedding_dim, n_centers_per_class, n_classes, lr, sigma, kernel_type = "gauss")

            history_gauss = rbf_model.fit(x_train_pct, y_train_pct,
                            batch_size=  batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_val, y_val),
                            callbacks = callbacks)

            error_rbf = rbf_model.evaluate(x_test, y_test, verbose = 0) 
            # Record of Highest Validation Accuracies

           

            highest_plain = np.max(history_plain.history["val_acc"])
            highest_gauss_kmeans = np.max(history_gauss_kmeans.history["val_acc"])
            highest_gauss = np.max(history_gauss.history["val_acc"])

            history.append({"plain": highest_plain,
                            "gauss_means": highest_gauss_kmeans,
                            "gauss": highest_gauss,
                            "plain_error": error_softmax,
                            "error_rbf": error_rbf,
                            "error_rbf_kmeans": error_rbf_kmeans})



            with open("Train_Results_"+feature_extractor+str(int(train_pct*100))+"_trial_"+str(i), "wb") as f:
                pickle.dump(history, f)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='CIFAR10, CIFAR100, Tiny', default="CIFAR10")
    argparser.add_argument('--feature_extractor', type=str, help='ConvNet, Resnet', default="ConvNet")
    argparser.add_argument('--file_path', type=str, help='Filepath', default= "model10.h5")
    argparser.add_argument('--n_trials', type=int, help='Number of trials', default= 3)

    args = argparser.parse_args()
    main(args)
