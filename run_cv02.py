from cv02.main02 import main
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--optimizer", type=str, default="adam")
    argparser.add_argument("--random_emb", type=int, default=1)
    argparser.add_argument("--emb_training", type=int, default=1)
    argparser.add_argument("--emb_projection", type=int, default=1)
    argparser.add_argument("--final_metric", type=str, default="neural")
    argparser.add_argument("--vocab_size", type=int, default=20000)
    argparser.add_argument("--batch_size", type=int, default=10000)
    argparser.add_argument("--lr_scheduler", type=str, default="step")

    my_config = vars(argparser.parse_args())
    my_config = {
        "vocab_size": 20000,
        "random_emb": False,
        "final_metric": "neural",
        "emb_training": False,
        "emb_projection": True,
        "lr": 0.01,
        "optimizer": "sgd",
        "batch_size": 1000,
        "lr_scheduler": "step"
    }

    main(my_config)
