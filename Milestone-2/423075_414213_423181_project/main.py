import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE
from src.activations import Sigmoid, ReLU
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, label_to_onehot, onehot_to_label, get_n_classes, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    HYPERPARAMETER_SEARCH_MODE = args.param_search # Set macro to run hyperparameter search (very slow!)

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        n_train = train_features.shape[0]
        indices = np.random.permutation(n_train)

        val_size = int(args.val_ratio * n_train)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        # Create Splits
        val_features = train_features[val_idx]
        new_train_features = train_features[train_idx]

        val_labels_reg = train_labels_reg[val_idx]
        new_train_labels_reg = train_labels_reg[train_idx]

        val_labels_classif = train_labels_classif[val_idx]
        new_train_labels_classif = train_labels_classif[train_idx]

        # Set train/test to new train/validation
        train_features = new_train_features
        train_labels_reg = new_train_labels_reg
        train_labels_classif = new_train_labels_classif
        test_features = val_features
        test_labels_reg = val_labels_reg
        test_labels_classif = val_labels_classif
        pass

    #normalize data to prevent over-influence of large features
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True) + 1e-8

    train_features = (train_features - mean) / std
    test_features = (test_features - mean) / std
    means = train_features.mean(axis=0, keepdims=True)
    
    stds  = train_features.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    
    train_features = normalize_fn(train_features, means, stds)
    test_features  = normalize_fn(test_features,  means, stds)

    # Hyperparameter search for MLP
    learning_rates = [0.2, 0.1, 0.05] # previously tested: 0.01, 0.001
    best_lr = None

    hidden_sizes = [24, 32, 48] # previously tested: 16, 64, 128
    best_hidden_size = None

    batch_sizes = [24, 32, 48] # previously tested: 16, 64
    best_batch_size = None

    epochs = [32, 50, 64] # previously tested: 25, 100
    best_epochs = None

    if HYPERPARAMETER_SEARCH_MODE:
        for lr in learning_rates:
            for hidden_size in hidden_sizes:
                for batch_size in batch_sizes:
                    for epoch in epochs:
                        if args.method == "mlp":
                            input_dim = train_features.shape[1]

                            if args.task == "classification":
                                best_val_acc = -1

                                mlp = MLP(
                                    dimensions=[input_dim, hidden_size, get_n_classes(train_labels_classif)],
                                    activations=[ReLU, Sigmoid]
                                )

                                y_train_one_hot = label_to_onehot(train_labels_classif, get_n_classes(train_labels_classif))

                                mlp.fit(
                                    train_features,
                                    y_train_one_hot,
                                    loss=MSE,
                                    epochs=epoch,
                                    batch_size=batch_size,
                                    learning_rate=lr
                                )

                                y_val_pred_scores = mlp.predict(test_features)
                                y_val_pred = onehot_to_label(y_val_pred_scores)

                                val_acc = accuracy_fn(y_val_pred, test_labels_classif)

                                if val_acc > best_val_acc:
                                    best_val_acc = val_acc
                                    best_lr = lr
                                    best_hidden_size = hidden_size
                                    best_batch_size = batch_size
                                    best_epochs = epoch

                            if args.task == "regression":
                                best_val_mse = None
                                mlp = MLP(
                                    dimensions=[input_dim, hidden_size, 1],
                                    activations=[ReLU, Sigmoid]  # or better: [ReLU, Linear]
                                )

                                y_train_reg = train_labels_reg.reshape(-1, 1)
                                y_val_reg = test_labels_reg.reshape(-1, 1)

                                mlp.fit(
                                    train_features,
                                    y_train_reg,
                                    loss=MSE,
                                    epochs=epoch,
                                    batch_size=batch_size,
                                    learning_rate=lr
                                )

                                y_val_pred = mlp.predict(test_features)
                                val_mse = mse_fn(y_val_pred, y_val_reg)

                                if best_val_mse == None or val_mse < best_val_mse:
                                    best_val_mse = val_mse
                                    best_lr = lr
                                    best_hidden_size = hidden_size
                                    best_batch_size = batch_size
                                    best_epochs = epoch

        print(f"Best learning rate: {best_lr}")
        print(f"Best hidden size: {best_hidden_size}")
        print(f"Best batch size: {best_batch_size}")
        print(f"Best epochs: {best_epochs}")

    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        ### WRITE YOUR CODE HERE
        pass

    elif args.method == "mlp":
        input_dim = train_features.shape[1]
        hidden_nodes = args.hidden_nodes

        if args.task == "classification":
            n_classes = get_n_classes(train_labels_classif)
            output_dim = n_classes

            if HYPERPARAMETER_SEARCH_MODE:
                hidden_nodes = best_hidden_size

            method_obj = MLP(
                dimensions=[input_dim, hidden_nodes, output_dim],
                activations=[ReLU, Sigmoid]
            )

        if args.task == "regression":
            
            method_obj = MLP(
                dimensions=[input_dim, hidden_nodes, 1],
                activations=[ReLU, Sigmoid]  # or better: [ReLU, Linear]
            )

        pass
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":

        if args.method == "mlp":
            n_classes = int(np.max(train_labels_classif)) + 1

            y_train_one_hot = label_to_onehot(train_labels_classif, n_classes)

            mlp_epochs = args.max_iters
            mlp_batch_size = args.batch_size
            mlp_lr = args.lr

            if HYPERPARAMETER_SEARCH_MODE:
                mlp_epochs = best_epochs
                mlp_batch_size = best_batch_size
                mlp_lr = best_lr

            method_obj.fit(
                train_features,
                y_train_one_hot,
                loss=MSE,
                epochs=mlp_epochs,
                batch_size=mlp_batch_size,
                learning_rate=mlp_lr
            )


            y_pred_scores = method_obj.predict(test_features)
            y_pred = onehot_to_label(y_pred_scores)

            acc = accuracy_fn(y_pred, test_labels_classif)
            macro_f1 = macrof1_fn(y_pred, test_labels_classif)

            print(f"Accuracy: {acc:.4f}")
            print(f"Macro F1:  {macro_f1:.4f}")
        pass

    elif args.task == "regression":
        assert args.method != "kmeans", "You should use kmeans as a classification method"

        if args.method == "mlp":
            y_train_reg = train_labels_reg.reshape(-1, 1)
            y_test_reg = test_labels_reg.reshape(-1, 1)

            mlp_epochs = args.max_iters
            mlp_batch_size = args.batch_size
            mlp_lr = args.lr

            if HYPERPARAMETER_SEARCH_MODE:
                mlp_epochs = best_epochs
                mlp_batch_size = best_batch_size
                mlp_lr = best_lr

            method_obj.fit(

                train_features,
                y_train_reg,
                loss=MSE,
                epochs=mlp_epochs,
                batch_size=mlp_batch_size,
                learning_rate=mlp_lr

            )

            y_pred = method_obj.predict(test_features)

            mse = mse_fn(y_pred, y_test_reg)
            print(f"MSE: {mse:.4f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / kmeans / mlp",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    # Feel free to add more arguments here if you need!
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="fraction of training data to use for validation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size for MLP training",
    )

    parser.add_argument(
        "--param_search",
        type=bool,
        default=False,
        help="search for optimal parameters (very slow)",
    )
    parser.add_argument(
        "--hidden_nodes",
        type=int,
        default=32,
        help="number of hidden nodes"
    )

    args = parser.parse_args()
    main(args)
