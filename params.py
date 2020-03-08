
# General
n_centers_per_class = 1
embedding_dim       = 64
kernel_type         = "gauss"
n_classes           = 10
n_output            = n_classes
feature_extractor   = "CNN"
dataset             = "CIFAR-10"
input_shape         = [32,32,3]
batch_size          = 32
lr                  = 0.00001
epochs              = 500
patience            = 20
n_trials            = 1
train_pct           = 0.10

# RBF Model
sigma            = 0

# Soft Triple Loss Model
gamma            = 0.1 
