# Improving Sample Efficiency with Normalized RBF Kernels

*  **data_load.py**: Include functions related to data loading, pre-processing, and sampling.
*  **models.py**: Include main clases for the models creation.
*  **params.py**: Contain training configuration parameters and models hyper-parameters.
*  **train_model.py**: Models training file with a single Center|Prototype by class.
*  **train_model_multiple_centers.py**: Models training file with multiple Centers|Prototypes by class.
*  **silhouette_eval.py**: Evaluation of the models Center|Prototypes distribution using the Silhouette coefficient.

For a simple command line execution of the training and the Silhouette evaluation files, reproduce as the following examples:

    >> python  train_model.py  RESNET|CNN  CIFAR-10|CIFAR-100
    >> python  train_model_multiple_centers.py  RESNET|CNN  CIFAR-10|CIFAR-100

    >> python  silhouette_eval.py  RESNET|CNN  CIFAR-10|CIFAR-100
