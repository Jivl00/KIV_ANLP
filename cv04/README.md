# CV04 - Named Entity Recognition/Morphological Tagging

**Deadline**: 20. 11. 2023 23:59:59

**Maximum points:** 20 + 5 (bonus)

**Contact:** Jan Pašek (pasekj@ntis.zcu.cz) - get in touch with me in case of any problems, but try not to leave
the problems for the weekends (I can't promise I'll respond during the weekend). Feel free to drop me an email or come to the next lesson
to discuss your problems. I'm also open to schedule online session in the meantime if you need
any support.

## Problem statement:

In this assignment you'll practice the RNN/LSTM-based neural networks for token classification
on two different tasks. The first task will be NER (Named Entity Recognition) and the second will be Morphological Tagging.

**What is NER:** Named entity recognition (NER) is the task of tagging entities in text with 
their corresponding type. Approaches typically use BIO notation, which differentiates the 
beginning (B) and the inside (I) of entities. O is used for non-entity tokens.
(source: https://paperswithcode.com/task/named-entity-recognition-ner)

**What is Tagging:** Morphological tagging is the task of assigning labels to a sequence of 
tokens that describe them morphologically. As compared to Part-of-speech tagging, 
morphological tagging also considers morphological features, such as case, #
gender or the tense of verbs. (source: https://paperswithcode.com/task/morphological-tagging).

To do so, we will use two datasets that are already pre-processed and ready to use
(including the data input pipeline). The first dataset is the [CNEC](https://ufal.mff.cuni.cz/cnec) (Czech Named Entity Corpus)
and is designated for the NER task. The second utilized dataset is the [UD](https://universaldependencies.org) (Universal Dependencies) -
Czech treebanks only. Both corpora are pre-processed to have the same format using labels in [BIO](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)) notation.

*In addition to the RNN and LSTM models, you will have a change to run some experiments with
BERT-like models (CZERT and Slavic) that are pre-trained for Czech and Slavic languages, respectively.
Thanks to that you will be able to understand the strength of the large pre-trained models.*

## Project Structure
- [data] - data for NER task (do not touch)
- [data-mt] - data for Tagging task (do not touch)
- [test] - unittest to verify your solutions (do not touch)
- main04.py - main source code of the assignment, training loops, etc.
- models.py - implementation of all models
- ner_utils.py - data input pipeline, etc. (do not touch)
- README.md

## Tasks:

### CKPT1 (Dataset Analysis)
Analyse the dataset - write the results into the discussion (secion 1). Answer all the following questions:
1. What labels are used by both datasets - write a complete list and explanation of the labels (use the referenced dataset websited).
2. How large are the two datasets (train, eval, test, overall).
3. What is the average length of a training example for the individual datasets - in number of whole words tokens as pre-tokenized in the dataset files.
4. What is the average length of a token for the individual datasets - in number of subword tokens when using `tokenizer = transformers.BertTokenizerFast.from_pretrained("UWB-AIR/Czert-B-base-cased")` - documentation: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer (methods: encode or batch_encode).
5. Count statistics about class distribution in dataset (train/dev/test) for the individual datasets.
6. Based on the statistic from the questions above - are the individual datasets balanced or unbalanced? In case at least one of the dataset is unbalanced, are there any implications for the model/solution or is there anything we should be aware of?

**[3 pt]** - Evaluation method: manually (each question 0.5pt)

### CKPT2 (RNN Model)
*Note: During the whole implementation, preserve the attribute/variable names if suggested (e.g. `self.__dropout_layer = ...` -> `self.__dropout_layer = torch.nn.Dropout(p=self.__dropout_prob)`. If you change the name of such variables, the tests will be failing.*

*Note: When implementing the models, mind the constructor params. All the parameters shall be employed by the model and the model shall adapt it's functionality based on the provided arguments (some parameters may be used in later steps as well -> no panic if you don't use them right now).*

1. State the equations used for computing the activations of an RNN model in the discussion (section 2).
2. Implement an RNN model with the allowed torch primitives only:
  - Allowed:
    - torch.nn.Embedding
    - torch.nn.Dropout
    - torch.nn.Linear
    - torch.nn.CrossEntropyLoss
    - all torch functions (e.g. tensor.view(), torch.tanh(), torch.nn.functional.Softmax(), etc...)
  - Not allowed:
    - torch.nn.Rnn
    - torch.nn.GRU
  - Architecture Description:
    - Inputs (come into the model tokenized using subword tokenizer) are embedded using an embedding layer.
    - Dropouts applied on the embedded sequence.
    - Sequence of hidden states is computed sequentially in a loop using one `torch.Linear` layer with `torch.tanh` activation (you have to save all the hidden states for later)
      - Hint: make sure you make a deep copy of the hidden state tensor preserving the gradient flow (`tensor.clone()`)
    - Dropout is applied to the sequence of hidden states
    - Compute output activations
    - Compute loss and return instance of `TokenClassifierOutput` (*Note: the loss is computed in the forward pass and returned in the `TokenClassifierOutput` to unify interface of our custom models with HuggingFace models*.)
3. Do a step by step debugging to ensure that the implementation works as expected - check the dimensionality of tensors flowing through the model (no points for that, but it is important for the experiments that the model works correctly)

**[4 pt]** - Evaluation method: passing unittests for ckpt2 (3.5pt), discussion manually (0.5pt)

### CKPT3 (LSTM Model)
*Note: During the whole implementation, preserve the attribute/variable names if suggested (e.g. `self.__dropout_layer = ...` -> `self.__dropout_layer = torch.nn.Dropout(p=self.__dropout_prob)`. If you change the name of such variables, the tests will be failing.*

*Note: When implementing the models, mind the constructor params. All the parameters shall be employed by the model and the model shall adapt it's functionality based on the provided arguments (some parameters may be used in later steps as well -> no panic if you don't use them right now).*

1. State the equations used for computing the activations of an LSTM model in the discussion and explain the individual gates (their purpose) (section 3).
2. Implement an LSTM model with any possible primitives :
  - Suggested:
    - torch.nn.Embedding
    - torch.nn.Dropout
    - torch.nn.Linear
    - torch.nn.LSTM
    - torch.nn.CrossEntropyLoss
    - all torch functions (e.g. tensor.view(), torch.tanh(), torch.nn.functional.Softmax(), etc...)
  - Architecture Description:
    - Inputs (come into the model tokenized using subword tokenizer) are embedded using an embedding layer.
    - Dropouts applied on the embedded sequence.
    - **Bi**LSTM layer with parameterizable number of layers is used to process the embedded sequence.
    - Dropout is applied to the sequential output of the LSTM layers
    - A dense layer with ReLu activation is applied
    - A classification head with softmax activation is applied to compute output activations
    - Compute loss and return instance of `TokenClassifierOutput` (*Note: the loss is computed in the forward pass and returned in the `TokenClassifierOutput` to unify interface of our custom models with HuggingFace models*.)
3. Do a step by step debugging to ensure that the implementation works as expected - check the dimensionality of tensors flowing through the model (no points for that, but it is important for the experiments that the model works correctly)

**[3 pt]** - Evaluation method: passing unittests for ckpt3 (2.5pt), discussion manually (0.5pt)


### CKPT4 (Freezing Parameters & L2 Regularization)

1. Implement a possibility to freeze an embedding layer of the RNN and LSTM model - it means that the embedding layer (that we alway init randomly) will not be trained at all.
    - method `self.__freeze_embedding_layer()` - for both models
2. Implement the following methods:
   - `compute_l2_norm_matching` - compute an L2 norm of all model parameters matching a pattern from a given list of patterns (python built-it function `any()` can be useful)
3. Implement `compute_l2_norm` method for both RNN and LSTM model and return the L2 scaled with the `self.__l2_alpha`. Use the previously implemented method
    - RNN: regularize only the dense layer for computing the new hidden state and the classification head
    - LSTM: regularize only the dense layer and classification head
4. In the discussion (section 4) explain in which case do we want to freeze the embedding layer. Also discuss whether it is useful to freeze embedding layer in our case when we initialize the embedding layer randomly - would you expect the model to work well with the frozen randomly initialized embedding layer?

**[3 pt]** - Evaluation method: passing unittest for ckpt4 (2pt), discussion manually (1pt)

### CKPT5 (Training loop & LR schedule)

*Note: During the whole implementation, preserve the attribute/variable names if suggested (e.g. `self.__dropout_layer = ...` -> `self.__dropout_layer = torch.nn.Dropout(p=self.__dropout_prob)`. If you change the name of such variables, the tests will be failing.*


1. Read through the training loop and understand the implementation. Check the usage of `scheduler` variable. (not evaluated by us, but helpful for you)
2. Implement `lr_schedule()` - LR scheduler with linear warmup during first `warmup_steps` training steps and linear decay to zero over the whole training.
    - The scheduler shall return number in [0, 1] - resulting LR used by the model is `training_args.learning_rate * lr_schedule()`
3. In the discussion (section 5) discuss why such LR scheduler can help to improve results. Discuss both the warmup and decay separately.

**[2 pt]** Evaluation method: passing unittests for ckpt5 (1pt), discussion manually (1pt)

### CKPT6 (Basic Experiments)

1. NER experiments with RNN and LSTM
   - Some hyperparameters to start with:
     - RNN
       - ```shell
          main04.py \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --model_type RNN \
            --data_dir data \
            --labels data/labels.txt \
            --output_dir /scratch.ssd/pasekj/job_3984202.cerit-pbs.cerit-sc.cz/output \
            --do_predict \
            --do_train \
            --do_eval \
            --eval_steps 200 \
            --logging_steps 50 \
            --learning_rate 0.0001 \
            --warmup_steps 4000 \
            --num_train_epochs 1000 \
            --no_bias \
            --dropout_probs 0.05 \
            --l2_alpha 0.01 \
            --lstm_hidden_dimension 64 \
            --num_lstm_layers 2 \
            --embedding_dimension 128 \
            --task NER
         ```
     - LSTM
       - ```shell
         main04.py \
            --model_type LSTM \
            --data_dir data \
            --labels data/labels.txt \
            --output_dir output \
            --do_predict \
            --do_train \
            --do_eval \
            --eval_steps 200 \
            --eval_dataset_batches 200 \
            --logging_steps 50 \
            --learning_rate 0.0001 \
            --warmup_steps 4000 \
            --num_train_epochs 1000 \
            --no_bias \
            --dropout_probs 0.05 \
            --l2_alpha 0.01 \
            --lstm_hidden_dimension 128 \
            --num_lstm_layers 2 \
            --embedding_dimension 128
         ```
   - Run at least one experiment with the following hyperparameters changed:
       - use `--no_bias`/don't use `--no_bias`
       - LR: {0.0001, 0.001} (`--learning_rate`)
       - L2 alpha: {0.01, 0} (`--l2_alpha`)
       - It means that you will have at least 6 runs for each model - always use the base hyperparameters and then change just the one
2. TAGGING experiments with RNN and LSTM
   - Some hyperparameters to start with:
     - RNN
       - ```shell
          main04.py \
            --model_type RNN \
            --data_dir data-mt \
            --labels data-mt/labels.txt \
            --output_dir output \
            --do_predict \
            --do_train \
            --do_eval \
            --eval_steps 300 \
            --eval_dataset_batches 200 \
            --logging_steps 50 \
            --learning_rate 0.0001 \
            --warmup_steps 4000 \
            --num_train_epochs 10 \
            --no_bias \
            --dropout_probs 0.05 \
            --l2_alpha 0.01 \
            --lstm_hidden_dimension 128 \
            --num_lstm_layers 2 \
            --embedding_dimension 128 \
            --task TAGGING
         ```
     - LSTM
       - ```shell
          main04.py \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --model_type LSTM \
            --data_dir data-mt \
            --labels data-mt/labels.txt \
            --output_dir output \
            --do_predict \
            --do_train \
            --do_eval \
            --eval_steps 300 \
            --logging_steps 50 \
            --learning_rate 0.0001 \
            --warmup_steps 4000 \
            --num_train_epochs 10 \
            --no_bias \
            --dropout_probs 0.05 \
            --l2_alpha 0.01 \
            --lstm_hidden_dimension 128 \
            --num_lstm_layers 2 \
            --embedding_dimension 128 \
            --task TAGGING
         ```
   - Run at least one experiment with the following hyperparameters changed:
       - use `--no_bias`/don't use `--no_bias`
       - LR: {0.0001, 0.001} (`--learning_rate`)
       - L2 alpha: {0.01, 0} (`--l2_alpha`)
       - It means that you will have at least 6 runs for each model - always use the base hyperparameters and then change just the one
3. CZERT and Slavic 
    - One experiment with each model for each task (4 runs in total) - hyperparameters provided below:
      - NER
        - CZERT
          - ```shell
              main04.py \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --model_type CZERT \
                --data_dir data \
                --labels data/labels.txt \
                --output_dir output \
                --do_predict \
                --do_train \
                --do_eval \
                --eval_steps 100 \
                --logging_steps 50 \
                --learning_rate 0.0001 \
                --warmup_steps 4000 \
                --num_train_epochs 50 \
                --dropout_probs 0.05 \
                --l2_alpha 0.01 \
                --task NER
            ```
        - SLAVIC
          - ```shell
              main04.py \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --model_type SLAVIC \
                --data_dir data \
                --labels data/labels.txt \
                --output_dir output \
                --do_predict \
                --do_train \
                --do_eval \
                --eval_steps 100 \
                --logging_steps 50 \
                --learning_rate 0.0001 \
                --warmup_steps 4000 \
                --num_train_epochs 50 \
                --dropout_probs 0.05 \
                --l2_alpha 0.01 \
                --task NER
            ```
      - TAGGING
        - CZERT
          - ```shell
             main04.py \
              --per_device_train_batch_size 32 \
              --per_device_eval_batch_size 32 \
              --model_type CZERT \
              --data_dir data-mt \
              --labels data-mt/labels.txt \
              --output_dir output \
              --do_predict \
              --do_train \
              --do_eval \
              --eval_steps 300 \
              --logging_steps 50 \
              --learning_rate 0.0001 \
              --warmup_steps 4000 \
              --num_train_epochs 10 \
              --dropout_probs 0.05 \
              --l2_alpha 0.01 \
              --task TAGGING
            ```
        - SLAVIC
          - ```shell
            main04.py \
              --per_device_train_batch_size 32 \
              --per_device_eval_batch_size 32 \
              --model_type SLAVIC \
              --data_dir data-mt \
              --labels data-mt/labels.txt \
              --output_dir output \
              --do_predict \
              --do_train \
              --do_eval \
              --eval_steps 300 \
              --logging_steps 50 \
              --learning_rate 0.0001 \
              --warmup_steps 4000 \
              --num_train_epochs 10 \
              --dropout_probs 0.05 \
              --l2_alpha 0.01 \
              --task TAGGING
            ```
4. Discussion for NER - compare results of the individual models and try to explain why the models achieve the results you observed. Specifically compare the results achieved with RNN/LSTM and CZERT/Slavic.
5. Discussion for TAGGING - compare results of the individual models and try to explain why the models achieve the results you observed. Specifically compare the results achieved with RNN/LSTM and CZERT/Slavic.

**[5 pt]** Evaluation method: passing unittests for ckpt6 (3pt), discussion manually (2pt) - it is not possible to get any points for the passing unittests if discussion is missing at this CKPT

### CKPT7 (Extended experiments)

1. Use the same hyperparameters for CZERT model as above and make additional experiments with the following hyperparameters on NER:
    - Freeze embedding layer and train the model - do you observe any difference in the achieved results?
    - Freeze first {2, 4, 6} layers of the CZERT model + freeze embeddings - do you observe any difference in the achieved results?
2. Adjust `main04.py` to enable to train another model `BERT` - simple change after line `243` and you can use the Czert model implementation and just provide a different model name. For this experiment, use the `bert-base-cased` (https://huggingface.co/bert-base-cased?text=Paris+is+the+%5BMASK%5D+of+France.). 
   - If you choose to implement this bonus, please use `--model_type BERT` so that unittest can recognize that correctly.
   - This experiment is a test of how well does a pre-trained model for English perform on a Czech tasks.
   - Run at least 5 experiments with different hyperparameters for each task (use MetaCentrum for that). Make sure you've run enough epochs to enable model to converge.
3. Discuss the results of both subtasks and also answer the following questions:
    - Does the model with frozen embeddings perform worse than the model with trainable embeddings?
    - Do you see any result improvement/impairment when freezing the lower layers of the CZERT model?
    - Does freezing the lower layers bring any benefits in term of results, speed of training, etc?
    - Does the BERT model work for Czech tasks? State the results and include a graph of eval metrics for the BERT model config for both tasks.

**[5pt]** Evaluation method: passing unittests for ckpt7 (3pt), discussion manually (2pt)

## Discussions

### Section 1 - Dataset Analysis

1. What labels are used by both datasets - write a complete list and explanation of the labels (use the referenced dataset websited).

---

    NER dataset - CNEC:
    O       -- Outside of a named entity
    I-T     -- Inside of a named entity Time expressions
    I-P     -- Inside of a named entity Personal names
    I-O     -- Inside of a named entity Artifact names
    I-M     -- Inside of a named entity Media names
    I-I     -- Inside of a named entity Institutions
    I-G     -- Inside of a named entity Geographical names
    I-A     -- Inside of a named entity Numbers in addresses
    B-T     -- Beginning of a named entity Time expressions
    B-P     -- Beginning of a named entity Personal names
    B-O     -- Beginning of a named entity Artifact names
    B-M     -- Beginning of a named entity Media names
    B-I     -- Beginning of a named entity Institutions
    B-G     -- Beginning of a named entity Geographical names
    B-A     -- Beginning of a named entity Numbers in addresses

---

    TAGGING dataset - UD:
    ADJ     -- Adjective
    ADP     -- Adposition
    ADV     -- Adverb
    AUX     -- Auxiliary
    CCONJ   -- Coordinating conjunction
    DET     -- Determiner
    INTJ    -- Interjection
    NOUN    -- Noun
    NUM     -- Numeral
    PART    -- Particle
    PRON    -- Pronoun
    PROPN   -- Proper noun
    PUNCT   -- Punctuation
    SCONJ   -- Subordinating conjunction
    SYM     -- Symbol
    VERB    -- Verb
    X       -- Other
    _       -- No label? What I found in the data: aby, abychom, abyste, _

2. How large are the two datasets (train, eval, test, overall).

Code details for this and following questions can be found in the `data_anal.py` file.

    NER dataset - CNEC:
    Train:  4688
    Dev:    577
    Test:   585
    Overall: 5850

    TAGGING dataset - UD:
    Train:  103143
    Dev:    11326
    Test:   12216
    Overall: 126685

As we can see, the second dataset is significantly larger than the first one.

3. What is the average length of a training example for the individual datasets - in number of whole words tokens as pre-tokenized in the dataset files.

---

    NER dataset - CNEC:
    Train:  25.5
    Dev:    25.6
    Test:   25.7

    TAGGING dataset - UD:
    Train: 17.6 
    Dev:   17.0 
    Test:  16.9

4. What is the average length of a token for the individual datasets - in number of subword tokens when using `tokenizer = transformers.BertTokenizerFast.from_pretrained("UWB-AIR/Czert-B-base-cased")` - documentation: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer (methods: encode or batch_encode).

---

    NER dataset - CNEC:
    Train:  35.6
    Dev:    35.8
    Test:   36.1

    TAGGING dataset - UD:
    Train:  23.4
    Dev:    22.4
    Test:   22.3

5. Count statistics about class distribution in dataset (train/dev/test) for the individual datasets.

![train.txt_CNEC_class_distribution.svg](img%2Ftrain.txt_CNEC_class_distribution.svg)
![dev.txt_CNEC_class_distribution.svg](img%2Fdev.txt_CNEC_class_distribution.svg)
![test.txt_CNEC_class_distribution.svg](img%2Ftest.txt_CNEC_class_distribution.svg)

![train.txt_UD_class_distribution.svg](img%2Ftrain.txt_UD_class_distribution.svg)
![dev.txt_UD_class_distribution.svg](img%2Fdev.txt_UD_class_distribution.svg)
![test.txt_UD_class_distribution.svg](img%2Ftest.txt_UD_class_distribution.svg)

6. Based on the statistic from the questions above - are the individual datasets balanced or unbalanced? In case at least one of the dataset is unbalanced, are there any implications for the model/solution or is there anything we should be aware of?

Both datasets are unbalanced. The NER dataset is more unbalanced than the TAGGING dataset.
This is because the NER dataset has a lot of O labels, which can be expected, as the majority of the words are not named entities.
The tagging dataset is more balanced, but still, some classes are more frequent than others (e.g. NOUN, VERB, PUNCT). This can be expected, as some classes are more common in the language than others.

The unbalanced dataset can lead to the model learning to always output the majority class, which can be a problem in the case of unbalanced classification.
Traditional metrics like accuracy can be misleading on unbalanced datasets, as they may not reflect the model's performance on minority classes.
Metrics such as F1-score, particularly the macro-averaged F1-score or per-class F1-scores, are more informative because they measure performance on each class separately and help assess how well the model identifies less frequent labels.
Adjusting class weights during training can help mitigate the imbalance by giving more importance to minority classes.

### Section 2 - RNN Model

![img.png](img%2Fimg.png)

where:
- `U` is the weight matrix for the input(`x_t`)-to-hidden connections
- `V` is the weight matrix for the hidden-to-hidden (recurrent) connections
- `b_h` is the bias term for the hidden layer
- `W` is the weight matrix for the hidden-to-output connections
- `b_y` is the bias term for the output layer

### Section 3 - LSTM Model
![img2.png](img%2Fimg2.png)
![img_1.png](img%2Fimg_1.png)

An LSTM model maintains a cell state `c_t` and a hidden state `h_t` at each time step `t`.
The LSTM cell uses four main components—an input gate `i`, forget gate `f`, output gate `o`, and a candidate cell state to control information flow and update these states.

* The forget gate decides how much of the previous cell state `c_{t-1}` should be "forgotten" or retained. Values close to 0 indicate that the cell state should be forgotten, while values close to 1 indicate that the cell state should be retained.
* The input gate decides how much of the candidate cell state `c_tilde` to add to the cell state. The candidate cell state represents the new information that the model wants to add to the cell state.
* The output gate decides how much of the cell state to output as the hidden state `h_t`. 

### Section 4 - Parameter Freezing & L2 Regularization

If the embedding layer is initialized with pretrained embeddings (e.g., word embeddings from Word2Vec, GloVe, or BERT embeddings), freezing the embedding layer can help retain this pretrained knowledge, especially when labeled data is limited.
In cases where the dataset is small or very domain-specific, freezing pretrained embeddings can reduce the risk of overfitting and preserve general language representations that might be useful in such specialized contexts.
Freezing the embedding layer also reduces the number of parameters that need to be updated, which can lead to faster training and lower memory usage. This is useful when computational resources are limited, and the embeddings already provide strong language representations.

In the scenario where the embedding layer is initialized randomly, I would expect that freezing the embedding layer would not be beneficial.
I think that randomly initialized embeddings do not carry any useful language information; they are simply vectors of random values. If frozen, these embeddings would remain random and uninformative throughout training, leading to poor model performance.


### Section 5 - LR Schedule

Learning rate (LR) schedulers can significantly impact model performance by adjusting the learning rate dynamically during training.
Warmup stabilizes training by preventing large, potentially destabilizing updates early on, allowing the model to ease into the learning process and establish robust initial representations.
Decay helps the model converge to a stable minimum, prevents oscillation around local minima.

### Section 6 - Basic Experiments Results
#### Discussion for NER

As we can see both CZERT and Slavic models outperform RNN and LSTM models. This is expected as
the pre-trained models had more data to learn from and were trained on a larger corpus (must be of course relevant).

![par_all.svg](img%2Fpar_all.svg)

The LSTM model was doing better than the RNN model - as expected.
The RNN model, while capable of capturing sequential dependencies, often struggles with long-term dependencies due to the
vanishing gradient problem. This can lead to suboptimal performance on tasks like NER, where context from distant tokens can be crucial.
The LSTM model, with its gating mechanisms, is better equipped to handle long-term dependencies. This typically results in 
better performance compared to RNNs on tasks requiring understanding of context over longer sequences.

![NER1.svg](img%2FNER1.svg)

As for the tuned hyperparameters, the `--no_bias` does not seem to have a significant impact on the model's performance.
Models with `--no_bias:false` were slightly better 0.4549 vs 0.4526 on the test F1 score. Also, the trends in the graphs
were almost identical.

The learning rate `0.001` was better than `0.0001` for both models. It is likely that the higher learning rate allowed
the model to converge faster.

![NER_lr.svg](img%2FNER_lr.svg)

The L2 alpha `0.01` slightly worse than `0` for both models. This came as a surprise, as L2 regularization is typically
used to prevent overfitting and improve generalization. It is possible that the L2 regularization was too strong? (but 0.0001 is quite low)

![NER_l2.svg](img%2FNER_l2.svg)

Below we can clearly see that the overfitting is present and therefore the L2 regularization should be used.

![NER_l2loss.svg](img%2FNER_l2loss.svg)

As mentioned both CZERT and Slavic models outperform RNN and LSTM models. CZERT, being a pre-trained transformer model
specifically designed for the Czech language, leverages large amounts of pre-training data. This allows it to capture
nuanced language patterns and context. Similar to CZERT, the Slavic model is pre-trained on Slavic languages, making it
adept at understanding the linguistic characteristics of these languages. Its performance on NER tasks is expected to be 
comparable to or slightly lower than CZERT, depending on the specific pre-training data and fine-tuning process.
Results suggest that it is indeed the truth as the Slavic reached 0.85 F1 score on the test set and CZERT 0.86.
However, the graphs below suggest that the CZERT was starting to overfit the training data - so its performance might
be even better with better regularization.

![NER_2.svg](img%2FNER_2.svg)

![NER_2.1.svg](img%2FNER_2.1.svg)

#### Discussion for TAGGING
The scenario for the TAGGING task is similar to the NER task.
The CZERT and Slavic models are having the best performance, this time closely followed by the LSTM model.
In general all models are performing better on the TAGGING task than on the NER task.

![TAG_all.svg](img%2FTAG_all.svg)

Performance of the RNN model is again worse than the LSTM model. The LSTM model is better at capturing long-term dependencies,
which is crucial for tasks like morphological tagging that require understanding of context over longer sequences.

![TAG1.svg](img%2FTAG1.svg)

The `--no_bias` hyperparameter does not seem to have a significant impact in this case either.
As for the learning rate, the `0.001` was better than `0.0001` for both models. The difference was less significant than in the NER task.

![TAG2.svg](img%2FTAG2.svg)

The L2 alpha `0.01` was slightly worse than `0` for both models. In this case, I see no signs of overfitting, so the L2 regularization might not be necessary
in this case.

![TAG3.svg](img%2FTAG3.svg)

![TAG4.svg](img%2FTAG4.svg)

As for the CZERT and Slavic models, the results clearly indicate that both models were overfitting the training data.
In the graphs below, the model are intentionally not grouped - so we can see that the shorter training time
was sufficient - in some cases maybe even lesser epochs would be enough. As for the model themselves, the CZERT
was again slightly better than the Slavic model. In this case, it is even more intuitive as the tagging task is
in my opinion more related to the language itself than the NER task.

![TAG_1.svg](img%2FTAG_1.svg)

![TAG_2.svg](img%2FTAG_2.svg)

![TAG_3.svg](img%2FTAG_3.svg)


### Section 7 - Extended Experiments Results (Bonus)

[TODO] (optional)

## Questions to think about (test preparation, better understanding):

1. Why CZERT and Slavic works when embeddings are freezed and RNN and LSTM model strugles in this setup?
2. Describe the benefits of subword tokenization?
3. Does W2V uses whole word od subword tokenization?
4. Name 3 real world use cases for NER?
5. How is the morphological tagging different from NER? Can we use the same model? If not, what would you change?
6. What is the benefit of using BiLSTM instead of unidirectional LSTM?
7. Is the dataset balanced or unbalanced? Why can it happen that a model can learn to always output the majority class in case of unbalanced classification?
8. Why can the bi-directionality of the LSTM help to solve taks such as NER or TAGGING?
9. How did you compute the L2 norm. Which weights did/didn't you used and why?
10. How are the following metrics calculated: F1, precission, recall. What is the difference between macro and micro averaging with when computing the F1?
11. Explain why F1=precision=recall when using micro averaging?
12. Can we use to predictions from tagging model to improve the named entity recognition model? If so, please describe how would you do that?

