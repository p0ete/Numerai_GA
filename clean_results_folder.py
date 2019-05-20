#!/usr/bin/env python
"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import random
import os

NB_TO_KEEP = 0
def main(main_folder):
    print("Keeping only the best {} files in each folder (other files are deleted).".format(NB_TO_KEEP))
    for folder in os.listdir("./results/{}".format(main_folder)):
        AUC = []
        for file in os.listdir("./results/{}/{}/".format(main_folder, folder)):
            AUC.append(file.split("_")[0])
        AUC.sort()
        if NB_TO_KEEP is 0:
            delete_files = AUC
        else:
            delete_files = AUC[:-NB_TO_KEEP]
        for delete_AUC in delete_files:
            for file in os.listdir("./results/{}/{}/".format(main_folder, folder)):
                if delete_AUC in file:
                    os.remove("./results/{}/{}/{}".format(main_folder, folder, file))
    print("Done !")

if __name__ == '__main__':
    for folder in os.listdir("./results/"):
        main(folder)
