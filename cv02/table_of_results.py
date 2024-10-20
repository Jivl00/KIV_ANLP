import pandas as pd
import numpy as np

best1 = {"lr": 0.01, "optimizer": "adam", "random_emb": 1, "emb_training": 1, "emb_projection": 0,
         "final_metric": "neural", "vocab_size": 40000, "batch_size": 100, "lr_scheduler": "exp"}
# 1.721 +-0.07794
best2 = {"lr": 0.01, "optimizer": "adam", "random_emb": 1, "emb_training": 1, "emb_projection": 0,
         "final_metric": "neural", "vocab_size": 40000, "batch_size": 100, "lr_scheduler": "step"}
# 1.711 +- 0.08979
best3 = {"lr": 0.01, "optimizer": "adam", "random_emb": 0, "emb_training": 1, "emb_projection": 0,
         "final_metric": "neural", "vocab_size": 40000, "batch_size": 100, "lr_scheduler": "step"}
# 1.791 +- 0.09386

# Create table
data = {
    "model": ["Random emb, ExpLR", "Random emb, StepLR", "No random emb, StepLR", "Dummy"],
    # mean +- confidence interval
    "accuracy -+ 95% confidence": [f"{1.721:.3f} +- {1.96 * 0.07794 / np.sqrt(12):.3f}",
                                   f"{1.711:.3f} +- {1.96 * 0.08979 / np.sqrt(12):.3f}",
                                   f"{1.791:.3f} +- {1.96 * 0.09386 / np.sqrt(12):.3f}",
                                   f"{3.2223:.3f} +- {1.96 * 0.05698 / np.sqrt(12):.3f}"]
}
df = pd.DataFrame(data)
print(df.to_markdown(index=False))
