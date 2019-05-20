#!/usr/bin/env python
"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import os

TARGETS = ["target_bernie", "target_elizabeth", "target_charles", "target_jordan", "target_ken", "target_frank", "target_hillary"]

def main():
    if not os.path.exists("./results"):
        os.makedirs("./results")
    for target in TARGETS:
        if not os.path.exists("./results/results_trainon_{}".format(target)):
            os.makedirs("./results/results_trainon_{}".format(target))
        for target_bis in TARGETS:
            if not os.path.exists("./results/results_trainon_{}/{}".format(target, target_bis)):
                os.makedirs("./results/results_trainon_{}/{}".format(target, target_bis))


if __name__ == '__main__':
    main()
