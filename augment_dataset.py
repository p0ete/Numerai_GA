#!/usr/bin/env python
"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
import random

def main():

    print("# Loading data...")
    # The training data is used to train your model how to predict the targets.
    train = pd.read_csv('numerai_training_data.csv', header=0)
    # The tournament data is the data that Numerai uses to evaluate your model.
    tournament = pd.read_csv('numerai_tournament_data.csv', header=0)

    # The tournament data contains validation data, test data and live data.
    # Validation is used to test your model locally so we separate that.
    validation = tournament[tournament['data_type'] == 'validation']

    # There are multiple targets in the training data which you can choose to model using the features.
    # Numerai does not say what the features mean but that's fine; we can still build a model.
    # Here we select the bernie_target.
    train_bernie = train.drop([
        'id', 'era', 'data_type', 'target_charles', 'target_elizabeth',
        'target_jordan', 'target_ken', 'target_frank', 'target_hillary'
    ],
                              axis=1)
    
    features = [f for f in list(train_bernie) if "feature" in f]
    #train = train[0:100]
    train_augmented =  train.copy()

    noise = np.random.normal(0,0.001,len(train_augmented))

    for f in features:
        train_augmented[f] = train_augmented[f] + noise

    train_augmented = train_augmented.append(train)

    #train_tmp = [train.loc[i, random.sample(features,1)[0]] + (random.random()-0.5)/100 for i in range(train.shape[0])]
    #train_augmented = train_augmented.append(train_tmp)

    #for i in range(train.shape[0]):
    #    print("{}/{}".format(i, train.shape[0]), end="\r")
    #    selected_feature = random.sample(features,1)[0]
        #print(train.iloc[i])
    #    train.loc[i, selected_feature] += (random.random()-0.5)/100

        #print(train.iloc[i])
    #    train_augmented = train_augmented.append(train.iloc[i], ignore_index=True)

    train_augmented = train_augmented.sample(frac=1).reset_index(drop=True)

    print("Original Dataset: {}".format(train.shape))
    print("Augmented Dataset: {}".format(train_augmented.shape))

    #print("Original Dataset: {}".format(train[features].head()))
    #print("Augmented Dataset: {}".format(train_augmented[features].head()))

    train_augmented.to_csv("numerai_training_data_augmented.csv", index = False)


if __name__ == '__main__':
    main()
