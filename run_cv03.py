import argparse


from cv03.main03 import main, CNN_MODEL, MEAN_MODEL

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=CNN_MODEL)
    parser.add_argument('--batches', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=33)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--activation', type=str, default="relu")

    parser.add_argument('--gradient_clip', type=float, default=0.5)

    parser.add_argument('--proj_size', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--emb_training', type=int, default=0)
    parser.add_argument('--random_emb', type=int, default=0)
    parser.add_argument('--emb_projection', type=int, default=1)
    parser.add_argument('--device', type=str, default="cpu")

    parser.add_argument('--cnn_architecture', type=str, default="C")
    parser.add_argument('--n_kernel', type=int, default=64)

    config = vars(parser.parse_args())

    # config = {
    #     "model": CNN_MODEL,
    #     "batches": 500000,
    #     "batch_size": 33,
    #     "lr": 0.0001,
    #     "activation": "relu",
    #
    #     "gradient_clip": 0.5,
    #
    #     "proj_size": 100,
    #     "seq_len": 100,
    #     "vocab_size": 20000,
    #     "emb_training": False,
    #     "random_emb": False,
    #     "emb_projection": True,
    #     "device": "cpu",
    #
    #     "cnn_architecture": "C",
    #     "n_kernel": 64,
    #
    # }
    main(config)
