# GNN-RecSys

**Graph Neural Networks for Recommender Systems**\
This repository contains code to train and test GNN models for recommendation, mainly using the Deep Graph Library
([DGL](https://docs.dgl.ai/)). 

**What kind of recommendation?**\
For example, an organisation might want to recommend items of interest to all users of its ecommerce platforms.

**How can this repository can be used?**\
This repository is aimed at helping users that wish to experiment with GNNs for recommendation, by giving a real example of code
to build a GNN model, train it and serve recommendations.

No training data, experiments logs, or trained model are available in this repository.

To run the code, users need multiple data sources, notably interaction data between user and items and features of users and items.

## Run the code
There are 3 different usages of the code: hyperparametrization, training and inference.
Examples of how to run the code are presented in UseCases.ipynb.

All 3 usages require specific files to be available. Please refer to the docstring to
see which files are required.

### Hyperparametrization

Hyperparametrization is done using the main.py file. 
Going through the space of hyperparameters, the loop builds a GNN model, trains it on a sample of training data, and computes its performance metrics.
The metrics are reported in a result txt file, and the best model's parameters are saved in the models directory.
Plots of the training experiments are saved in the plots directory.
Examples of recommendations are saved in the outputs directory.
```bash
python main.py --from_beginning -v --visualization --check_embedding --remove 0.85 --num_epochs 100 --patience 5 --edge_batch_size 1024 --item_id_type 'ITEM IDENTIFIER' --duplicates 'keep_all'
```
Refer to docstrings of main.py for details on parameters.

### Training

When the hyperparameters are selected, it is possible to train the chosen GNN model on the available data.
This process saves the trained model in the models directory. Plots, training logs, and examples of recommendations are saved.
```bash
python main_train.py --fixed_params_path test/fixed_params_example.pkl --params_path test/params_example.pkl --visualization --check_embedding --remove .85 --edge_batch_size 512
```
Refer to docstrings of main_train.py for details on parameters.

### Inference
With a trained model, it is possible to generate recommendations for all users or specific users.
Examples of recommendations are printed.
```bash
python main_inference.py --params_path test/final_params_example.pkl --user_ids 123456 \
--user_ids 654321 --user_ids 999 \
--trained_model_path test/final_model_trained_example.pth --k 10 --remove .99
```
Refer to docstrings of main_inference.py for details on parameters.

