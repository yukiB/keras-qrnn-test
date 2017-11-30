#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, with_statement, print_function, unicode_literals
import argparse
from data import create_chaos_data, create_sin_data, create_train_data_and_test_data
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("data",
                      help="Mode of data. (sin/duffing/logistic)")
    argp.add_argument("model",
                      help="Mode of learning model. (qrnn/rnn/cnn)")
    argp.add_argument("-p", "--prediction", dest="prediction", default=1,
                      help="Prediction Num (default=1)")
    argp.add_argument("-s", "--sequence", dest="sequence", default=150,
                      help="sequence (default=150)")
    argp.add_argument("-e", "--epoch", dest="epoch", default=30,
                      help="epoch num (default=30)")
    argp.add_argument("-i", "--index", dest="index", default=0,
                      help="prediction start's index (default=0)")
    argp.add_argument("-l", "--load", dest="load", default=False,
                      action="store_true",
                      help="load (default=False)")

    args = argp.parse_args()

    df = None
    data = str(args.data)

    n_sequence = int(args.sequence)
    n_prediction = int(args.prediction)
    if data == "sin":
        df = create_sin_data()
    elif (data == "duffing" or data == "logistic"):
        df = create_chaos_data(data)
    else:
        print("Please input correct mode.\n")
        print(argp.parse_args('-h'.split()))
    if args.model:
        from train import SequentialModel
        (X_train, y_train), (X_test, y_test) = create_train_data_and_test_data(df[["data"]],
                                                                               n_prev=n_sequence,
                                                                               m=n_prediction)
        model = SequentialModel(str(args.model), data, n_sequence, n_prediction)
        start = time.time()
        if args.load:
            model.load()
        else:
            model.train(X_train, y_train, X_test, y_test, epoch=int(args.epoch))
        print("Elasped time: " + str(time.time() - start))
        model.save()
        model.predict(X_test, y_test)
        if n_prediction == 1:
            model.sequential_predict(df[["data"]], n_sequence, start=int(args.index))
        plt.show()
