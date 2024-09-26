from cv01.main01 import main
from itertools import product
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dp', type=float, default=0)
    parser.add_argument('--model', type=str, default="dense")
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--scheduler', type=str, default="exponential")
    parser.add_argument('--use_normalization', type=bool, default=False)

    config = vars(parser.parse_args())
    # config = {
    #     "lr": 0.01,
    #     "use_normalization":False,
    #     "optimizer": "sgd", # ADAM,
    #     "batch_size":10,
    #     "dp":0,
    #     "scheduler":"exponential",
    #     "model": "dense"
    #
    # }

    main(config)
    # main()

