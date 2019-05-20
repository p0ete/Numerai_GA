#!/usr/bin/env python
"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
from sklearn import decomposition, ensemble, metrics, pipeline
#from xgboost import XGBClassifier

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
import random
import matplotlib.pyplot as plt
import os 
import gc

import clean_results_folder

TARGETS = ["target_bernie", "target_elizabeth", "target_charles", "target_jordan", "target_ken", "target_frank", "target_hillary"]


EPOCHS = 10
BATCH_SIZE = 128*12

LOAD_POPULATION_FROM_FOLDER = False
POPULATION_SIZE = 5
NB_TO_KEEP = 2
NB_GENERATIONS = 2
MUTATION_RATE = 0.2

ACTIVATION_FUNCTIONS = ["softmax", "elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid", "linear"]
LOSSES_FUNCTIONS = ["mean_squared_error", "mean_squared_logarithmic_error", "squared_hinge", "logcosh",\
 "binary_crossentropy", "kullback_leibler_divergence","poisson","cosine_proximity"]  

def uniqueid():
    seed = random.getrandbits(32)
    while True:
       yield seed
       seed += 1

def fit_model(model, train_sequence, validation_sequence):
    ea = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, restore_best_weights=True)
    epochs = EPOCHS
    batch_size = BATCH_SIZE

    try:
        model.fit(x=train_sequence[0], y=train_sequence[1], batch_size=batch_size,
        epochs=epochs, verbose=1, callbacks=[ea], validation_data=(validation_sequence[0], validation_sequence[1]))
    except:
        print("ERROR: Could not train the model !")
    return model

def build_model(nb_features, n_classes, params):

    layers = params["dense_layers"]
    input_main = Input(shape=(nb_features,), name="main")


    x = Dense(layers[0], activation=params["activation"])(input_main)
    if len(layers) >1:
        for nb_neurons in layers[1:]:
            x = Dropout(params["dropout"])(x)
            x = Dense(nb_neurons, activation=params["activation"])(x)
    predictions = Dense(n_classes, activation='sigmoid')(x)

    model = Model(inputs=[input_main], outputs=predictions)

    return model

def compile_model(model, params):
    
    #opt = SGD(lr= params["learning_rate"], momentum=params["momentum"], decay=params["decay"], nesterov=True)
    opt  = Adam(lr=params["learning_rate"], beta_1=params["beta_1"], beta_2=params["beta_2"], decay=params["decay"])

    model.compile(optimizer=opt,
                  loss=params["loss"])

    return model


def GA_train_and_evaluate(individual, features, train_sequence, validation_sequence, validation, x_prediction, unique_sequence):
    print(individual["params"])
    model = build_model(len(features), n_classes=1, params=individual["params"])
    model = compile_model(model, individual["params"])

    print("# Training...")
    model = fit_model(model, train_sequence, validation_sequence)
    
    print("# Predicting...")
    y_prediction = model.predict(x_prediction)
    probabilities = y_prediction[:,0]

    AUC= metrics.roc_auc_score(validation[TARGET], probabilities)
    print("- validation AUC for {}: {}".format(TARGET,AUC))

    with open("./results/results_trainon_{}/{}/{:f}_{}.txt".format(TARGET, TARGET, AUC, str(next(unique_sequence))[-8:]), "w") as text_file:
        print(f"{individual['params']}", file=text_file)

    targets = ["target_bernie", "target_elizabeth", "target_charles", "target_jordan", "target_ken", "target_frank", "target_hillary"]
    targets.remove(TARGET)
    for target in targets:

        correct = [
            round(x) == y
            for (x, y) in zip(probabilities, validation[target])
        ]
        AUC_tmp = sum(correct) / float(validation.shape[0])
        print("- validation AUC for "+str(target)+": "+str(AUC_tmp))

        with open("./results/results_trainon_{}/{}/{:f}_{}.txt".format(TARGET, str(target), AUC_tmp, str(next(unique_sequence))[-8:]), "w") as text_file:
            print(f"{individual['params']}", file=text_file)
    del model
    return AUC

def GA_extract_best(population):
    best_AUC = []
    for individual in population:
        if len(best_AUC) < NB_TO_KEEP:
            best_AUC.append(individual)
        else:
            if individual["AUC"] > min([k["AUC"] for k in best_AUC]):
                new_best_AUC = [j for j in best_AUC if j["AUC"] is not min([k["AUC"] for k in best_AUC])]
                new_best_AUC.append(individual)
                best_AUC = new_best_AUC
                del new_best_AUC
    
    return best_AUC

def mutate(baby_params, key):
    #print("The {} is mutating !".format(key))
    if key == "dense_layers":
        if round(random.random(),0) == 0:
            #Drop or duplicate layer
            if len(baby_params[key]) == 1:
                # duplicate, because there is only one layer
                baby_params[key].append(baby_params[key][0])
            else:
                if round(random.random(),0) == 0:
                    #duplicate
                    idx = random.sample([i for i in range(len(baby_params[key]))], 1)[0]
                    baby_params[key].insert(idx, baby_params[key][idx])
                else:
                    #drop
                    idx = random.sample([i for i in range(len(baby_params[key]))], 1)[0]
                    del baby_params[key][idx]
        else:
            #modify an existing layer
            idx = random.sample([i for i in range(len(baby_params[key]))], 1)[0]
            if round(random.random(),0) == 0:
                baby_params[key][idx] += 16
            else:
                baby_params[key][idx] -= 16
            baby_params[key][idx] = max(baby_params[key][idx], 16)
    elif key == "activation":
        baby_params[key] = random.sample(ACTIVATION_FUNCTIONS, 1)[0]
    elif key == "loss":
        baby_params[key] = random.sample(LOSSES_FUNCTIONS, 1)[0]

    elif key == "decay":
        if round(random.random(),0) == 0:
            baby_params[key] += 0.1*baby_params[key]
        else:
            baby_params[key] -= 0.1*baby_params[key]

        baby_params[key] = round(max(baby_params[key], 0), 4)
    elif key == "train_on":
        pass
    else:
        if round(random.random(),0) == 0:
            baby_params[key] += 0.05
        else:
            baby_params[key] -= 0.05

        baby_params[key] = round(max(baby_params[key], 0), 2)
    return baby_params
def GA_reproduce(best_AUC):
    babies = []
    for parent1 in best_AUC:
        for parent2 in best_AUC:
            if parent1["AUC"] is not parent2["AUC"]:
                baby_params = {}
                for key in parent1["params"]:
                    if round(random.random(),0) == 0:
                        baby_params[key] = parent1["params"][key]
                    else:
                        baby_params[key] = parent1["params"][key]

                    if random.random() < MUTATION_RATE:
                        baby_params = mutate(baby_params, key)
                babies.append({"params": baby_params, "AUC":-1})
    return babies

def plot(x):
    plt.figure()
    plt.plot(x)
    plt.show()
    plt.pause(0.0001)

def read_dict(text_file):
    string = text_file.read()
    string = string[1:-2].split(", '")
    i = 0
    params = {}
    for p in string:
        if i is 0:
            a = p.split(":")[0][1:-1]
        else:
            a = p.split(":")[0][:-1]
        i +=1
        b = p.split(":")[1]
        a = str(a)
        if "[" in b:
            b = b[2:-1].split(",")
            b = [int(elem) for elem in b]

        elif "'" in b:
            b = b[2:-1]
        else:
            b = float(b)
        params[a] = b
    return params
def main():
    np.random.seed(0)
    unique_sequence = uniqueid()

    #x = [ii for ii in range(NB_GENERATIONS)]

    #plt.ion()

    #fig, ax = plt.subplots()

    #ax.axis([0, NB_GENERATIONS, 0, 1])


    print("# Loading data...")
    train = pd.read_csv('./data/numerai_training_data.csv', header=0)
    tournament = pd.read_csv('./data/numerai_tournament_data.csv', header=0)
    validation = tournament[tournament['data_type'] == 'validation']

    train_bernie = train.drop([
        'id', 'era', 'data_type'
        ], axis=1)

    targets_to_drop = ["target_bernie", "target_elizabeth", "target_charles", "target_jordan", "target_ken", "target_frank", "target_hillary"]
    targets_to_drop.remove(TARGET)

    train_bernie = train_bernie.drop(targets_to_drop, axis=1)

    features = [f for f in list(train_bernie) if "feature" in f]
    X = train_bernie[features]
    Y = train_bernie[TARGET]
    x_prediction = validation[features]
    ids = tournament['id']

    train_sequence = [X[:int(0.9*len(X))], Y[:int(0.9*len(X))]]
    validation_sequence = [X[int(0.9*len(X)):], Y[int(0.9*len(X)):] ]

    ##################################################################################################
    ##################################### GENETIC ALGORITHM ##########################################
    ##################################################################################################

    if LOAD_POPULATION_FROM_FOLDER:
        folder = "./results/results_trainon_{}/{}/".format(TARGET, TARGET)
        population_tmp = []
        for file in os.listdir(folder):
            AUC = float(file.split("_")[0])
            with open(folder+file, "r") as text_file:
                params = read_dict(text_file)
                try:
                    tmp = params["activation"]
                    del tmp
                except:
                    params["activation"] = "relu"
                try:
                    tmp = params["loss"]
                    del tmp
                except:
                    params["loss"] = "binary_crossentropy"
            individual = {"params" : params, "AUC": AUC}
            if individual["params"]["train_on"] == TARGET:
                population_tmp.append(individual)
        while len(population_tmp) < POPULATION_SIZE:
            babies = GA_reproduce(population_tmp)
            population_tmp += babies 

        population = random.sample(population_tmp, POPULATION_SIZE)
        del population_tmp
    else:

        population = []

        kk=0
        for individual in range(POPULATION_SIZE):
            print("########################################")
            print(str(TARGET) + " GENERATION 0: Ind. nb. " + str(kk))
            print("########################################")
            kk+=1
            individual = {}
            individual["params"] = {
                "dense_layers": [16*int(1+(random.random()*10)) for i in range(int(random.random()*10)+1)],
                "dropout": round(random.random()*0.5,2),
                "learning_rate": round(random.random()*0.5,2),
                "momentum": round(random.random(),2),
                "beta_1": round(random.random(),2),
                "beta_2": round(random.random(),2),
                "decay": random.sample([0.1,0.01,0.001,0.0001,0.00001, 0],1)[0],
                "activation": random.sample(ACTIVATION_FUNCTIONS, 1)[0],
                "loss": random.sample(LOSSES_FUNCTIONS, 1)[0],
                "train_on": TARGET
            }

            
            individual["AUC"] = GA_train_and_evaluate(individual, features, train_sequence, validation_sequence, validation, x_prediction, unique_sequence)
            population.append(individual)

    history = [[],[]]
    best_individuals = GA_extract_best(population)

    print("The best AUC in this generation is : {}".format(max([k["AUC"] for k in best_individuals])))
    print("The mean AUC (of selected parents) in this generation is : {}".format(np.mean([k["AUC"] for k in best_individuals])))
    history[0].append(max([k["AUC"] for k in best_individuals]))
    history[1].append(np.mean([k["AUC"] for k in best_individuals]))

    #plot(history[1])
    #ax.clear()
    #ax.plot(history[1])

    #plt.plot(history[1])
    #plt.draw()
    #plt.pause(0.0001)
    #plt.clf()
    #line_mean.set_data(x[0], history[1])

    #line_mean, = ax.plot([0], history[1], color='k')

    #plt.pause(0.05)
    #fig.canvas.draw()
    #fig.canvas.flush_events()
    
    del population

    babies = GA_reproduce(best_individuals)
    population_tmp = babies 
    

    while len(population_tmp) < POPULATION_SIZE:
        babies = GA_reproduce(best_individuals)
        population_tmp += babies 
 
    population = random.sample(population_tmp, POPULATION_SIZE - len(best_individuals))
    population += best_individuals
    del population_tmp, babies, best_individuals

    for i in range(NB_GENERATIONS):
        kk = 0
        for individual in population:

            print("########################################")
            print(str(TARGET) + " GENERATION "+str(i+1)+": Ind. nb. " + str(kk))
            print("########################################")
            kk+=1
            if individual["AUC"] == -1:
                individual["AUC"] = GA_train_and_evaluate(individual, features, train_sequence, validation_sequence, validation, x_prediction, unique_sequence)

        
        best_individuals = GA_extract_best(population)
        print("The best AUC in this generation is : {}".format(max([k["AUC"] for k in best_individuals])))
        print("The mean AUC (of selected parents) in this generation is : {}".format(np.mean([k["AUC"] for k in best_individuals])))
        history[0].append(max([k["AUC"] for k in best_individuals]))
        history[1].append(np.mean([k["AUC"] for k in best_individuals]))

        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis

        ax.plot(history[0], 'r', history[1], 'b')
        fig.savefig('history_trainon_{}.png'.format(TARGET))
        plt.close(fig) 


        #plot(history[1])
        #ax.clear()
        #ax.plot(history[1])

        #plt.plot(history[1])
        #plt.draw()
        #plt.pause(0.0001)
        #plt.clf()

        #line_mean.set_data(x[:len(history[1])], history[1])
        #ax.axis([0, NB_GENERATIONS, 0, 1])
        #fig.canvas.draw()
        #fig.canvas.flush_events()

        del population

        babies = GA_reproduce(best_individuals)
        population_tmp = babies 
        

        while len(population_tmp) < POPULATION_SIZE:
            babies = GA_reproduce(best_individuals)
            population_tmp += babies 
     
        population = random.sample(population_tmp, POPULATION_SIZE - len(best_individuals))
        population += best_individuals
        del population_tmp, babies, best_individuals

        with open("./history_trainon_{}.txt".format(TARGET), "w") as text_file:
            print(f"{history}", file=text_file)

        clean_results_folder.main(main_folder = "results_trainon_{}".format(TARGET))
    #plt.show()
if __name__ == '__main__':
    for t in TARGETS:
        global TARGET
        TARGET = t
        main()
        gc.collect()


