from utils.arg_parser import args
import os
import pandas as pd
from algs.stul.STUL import STUL
from algs.mna.MNA import MNA
from algs.hydra.HYDRA import HYDRA

def run():
    if args.alg == 'mna':
        model = MNA(args)
    elif args.alg == 'stul':
        model = STUL(args)
    elif args.alg == 'hydra':
        model = HYDRA(args)
    else:
        print("Model not exist!")

    
    if not os.path.exists(args.feature_path):
        model.preprocessing(args)


    model.train(args)
    model.test(args)


if __name__ == '__main__':
    run()
