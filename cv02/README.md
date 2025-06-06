# KIV / ANLP Exercise 02

*Deadline to push results:* 2024-10-27 23:59:59

*Maximum points:* 20+5

------------------------------------------------------------------------

The Goal
========

Implement missing parts in the given template of the supervised machine
learning stack for estimating semantic textual similarity (STS). Train and
evaluate it on the given dataset. Use architecture described in the
following section.

What is Semantic Textual Similarity
===================================

Semantic textual similarity deals with determining how similar two
pieces of texts are. This can take the form of assigning a score from 0
to 6 (Our data).

Project Structure
=================

- [data]
- [tests]
    - *anlp01-sts-free-train.tsv*
    - *anlp01-sts-free-test.tsv*
- *main02.py*

The Dataset
============

The dataset was generated during our collaboration with Czech News
Agency - so it is real dataset.

The training part of the dataset contains 116956 samples. Each sample
consists of two sentences and an annotation of their semantic
similarity.

    Příčinu nehody vyšetřují policisté.\tPříčinu kolize policisté vyšetřují.\t4.77

Tasks \[20+5 points in total\]
===============================

### Dataset Statistics **[1pt]**

Create histogram of pair similarity.

![Train Pair Similarity Histogram.svg](img%2FTrain%20Pair%20Similarity%20Histogram.svg)
![Test Pair Similarity Histogram.svg](img%2FTest%20Pair%20Similarity%20Histogram.svg)

_For the test dataset, values are rounded to whole numbers._

_As you can see, the dataset is not balanced. The most common value is 0 for the training dataset and 1 for the test
dataset._

Present mean and std of the dataset.

| Dataset |    Mean |     Std |
|:--------|--------:|--------:|
| Train   | 2.44669 | 2.11928 |
| Test    | 2.65509 | 1.78368 |

### Baseline analysis **[2pt]**

What would the loss of a model returning a random value between 0 and 6 uniformly look like?

What would the loss of a model returning best prior (most probable output) look like?

_Well that is highly dependent on the loss function used._

| Model    | Loss Type | Loss Value |
|:---------|:----------|-----------:|
| Random   | MSE       |    6.49034 |
| Random   | MAE       |    2.08383 |
| Majority | MSE       |    10.3517 |
| Majority | MAE       |       2.66 |

_In the table above, values of MSE and MAE are calculated on the test dataset.
For the random model 30 runs were performed and the mean value is presented. For the best prior model
the value of 0 was chosen as the most common value in the training dataset._

_MSE and MAE were chosen because the expected values in the test dataset are floating point numbers._

### Implement Dummy Model

1. **Analyze the Dataset**

   #### **CF\#1**

   Count occurrences of words in the datasset, and prepare a list of
   top\_n words

**CKPT\#1**

2. **Prepare Word Embeddings**.
   https://drive.google.com/file/d/1MTDoyoGRhvLf15yL4NeEbpYLbcBlDZ3c/view?usp=sharing

   [//]: # (# https://fasttext.cc/docs/en/crawl-vectors.html)

   [//]: # (# EMB_FILE = "b:/embeddings/Czech &#40;Web, 2012, 5b tokens &#41;/cztenten12_8-lema-lowercased.vec")

   #### **CF\#2**

   Use the *list of top N words* for pruning the given Word embeddings.
   The cache will be stored on the hard drive for future use **CF\#3**.

   You should see two new files *word2idx.pckl* and *vecs.pckl*
   **CKPT\#2**

4. **Implement SentenceVectorizer** This module takes text as an input
   and transforms it into a sequence of vocabulary ids. **CF\#4**

   example:

   Input: Příčinu nehody vyšetřují policisté

   Output: 3215 2657 2366 7063

   **CKPT\#3**

5. **Implement the DataLoader** Implement a Python iterator that loads data from a text file and yields batches of
   examples. The iterator should handle a dataset stored in the text file and process it in batches as it iterates.

   Use implemented SentenceVectorizer to preprocess the data during
   initialization. **CF\#5** Implement the `__next__` function to
   return a batch of samples. **CF\#6** Shuffle training dataset after
   each epoch **CF\#7**

6. **Implement training loop**

   Implement basic training loop **CF\#8a**. Implement testing for
   model and dataset**CF\#8b**.

7. **Implement DummyModel** DummyModel computes the mean STS value of
   given dataset in the initialization phase and returns this value as
   a prediction independently on model inputs. **CF\#9**

> **[4pt]**

8. **Implement Neural Network Model**

   #### **CF\#10a** **CF\#10b**

   The Model takes two sequences of numbers as an input (Sentence A and
   Sentence B). Use prepared word embeddings (task-2) to lookup word
   vectors from ids. Add one trainable projection layer on top of the
   embedding layer. Use the mean of all words vectors in a sequence as
   a representation of the sequence.

   Concatenate both sequence representations and pass them through two additional fully-connected layers. The final
   layer contains a single neuron without an activation function. The output represents the value of the STS (Semantic
   Textual Similarity) measure.

   ![Visualization of the architecture from task
   7](img/tt-d.png)

9. **Implement Cosine Similarity Measure**

   Change similarity measure implemented with neural network from
   task-7 to cosine similarity measure (cosine of the angle between
   sentA and sentB).

   ![Visualization of the architecture from task
   7](img/ttcos.png)

> **[4pt]**

10. **Log these:**
    1. mandatory logged
       hparams `["random_emb", "emb_training", "emb_projection", "vocab_size", "final_metric", "lr", "optimizer", "batch_size"]`
    2. mandatory metrics : `["train_loss", "test_loss"]`

11. **Run Experiments with different Hyper-parameters**

    1. Use randomly initialized embeddings/load pretrained.

       `random_emb = [True, False]`
    2. Freeze loaded embeddings and do not propagate your gradients into them / Train also loaded embeddings.

       `emb_training = [True, False]`
    3. Add projection layer right after your Embedding layer.

       `emb_projection = [True, False]`
    4. Size of loaded embedding matrix pruned by word frequency occurrence in your training data.

       `vocab_size = [20000]`
    5. Used final metric in two tower model.

       `final_metric = ["cos", "neural"]`


13. **The best performing experiment run at least 10 times** **[2pt]**


14. **Tune More Voluntarily [0-5pt]**
    Add more tuning and HP e.g. LR decay, tune neural-metric head, vocab_size, discuss

    **[5pt extra]**

NOTES TO THIS SECTION
=====================
I would personally definitely trim the score in the top n words calculation (not only splitting by space but by
tabulator as well). I gave it some thought
and I think that I would leave the case and the punctuation in the text - because memory is not a problem
and words like "Ahoj!" and "ahoj" should be treated as different words but have similar embedding vectors.
For the purpose of successful unit tests, I kept the original implementation.

As for the embeddings, &lt;PAD&gt; token is represented as a zero vector and &lt;UNK&gt; token is represented as a random vector
(uniform distribution). And they are not included in the vocabulary size, therefore the size of the vocabulary is the
vocab_size + 2. Here a little more detailed description would be nice - It would eliminate the need for the discussion
at the seminar.

Also, It would be lovely to know in advance that the implementation will be needing some adjustments - like random
embedding
etc. I would have prepared the code for that in advance and not be surprised by the requirements later on.

The figures in the task description really helped with the understanding of the task.

# My results

Overall, over 3000 experiments were run. Since this number is quite high and wandb processes only 50
runs in the figures, most of the pictures presented can be quite misleading (since 50 runs is not the general truth).
Great example is the first figure, where other batch sizes are not visible at all, since they were not included in the
last 50 runs.

## Hyper Parameter Analysis

### Parallel Coordinate Chart **[1pt]**

I had some trouble with boolean values in the shell script, so I used integers instead. I hope that is not a problem.
![W&B Chart 19. 10. 2024 23_21_24.svg](img%2FW%26B%20Chart%2019.%2010.%202024%2023_21_24.svg)
Regarding the figure above, I can only cite Mr. Cibulka who taught mathematical analysis 1: "Z grafu není vidět nic."
![W&B Chart 19. 10. 2024 23_21_242.svg](img%2FW%26B%20Chart%2019.%2010.%202024%2023_21_242.svg)
Here, only runs with test loss lower than 3 (dummy model) are presented. What is interesting is that only neural final
metric was used in these runs. It is also visible that Adam optimizer performed better than SGD and
that the best results were achieved with learning rate 0.01. I have also observed that the smaller the batch size, the
better the results (for the 3 values tested).

Given the number of experiments, I really did not want to run more. But I believe that it could be beneficial to run
more experiments with different batch sizes (especially smaller ones) to see to what extent the trend continues.

### Table of my results **[4pt]**

1. list all tuned HP

| `random_emb` | `emb_training` | `emb_projection` | `vocab_size` | `final_metric`           | `lrs`                         | `opts`   | `batch_size` | `lr_scheduler`       |
|--------------|----------------|------------------|--------------|--------------------------|-------------------------------|----------|--------------|----------------------|
| false true   | false true     | false true       | 20000 40000  | cosine_similarity neural | 0.1 0.01 0.001 0.0001 0.00001 | SGD Adam | 100 500 1000 | StepLR ExponentialLR |

3. add Dummy model into table
3. present results with confidence intervals (run more experiments with the same config)

I have learned my lesson and improved naming of my runs, therefore now I can easily group runs with the same
configuration
(I used the config itself as the name of the run). So now I can choose the best runs as a best mean of the runs with the
same
configuration.

Here are the results of the best runs and the dummy model:

| model                 | accuracy +- 95% confidence |
|:----------------------|:---------------------------|
| Random emb, ExpLR     | 1.721 +- 0.044             |
| Random emb, StepLR    | 1.711 +- 0.051             |
| No random emb, StepLR | 1.791 +- 0.053             |
| Dummy                 | 3.222 +- 0.032             |

Only distinct configurations are presented, all models shared: `learning_rate=0.01`, `optimizer=adam`, `emb_training=1`,
`emb_projection=0`, `final_metric=neural`, `batch_size=100`.
Overall, 12 runs were performed for each configuration.

I´m a bit surprised that the best results were achieved with no embedding projection. I would expect that the projection
layer would help the model to learn better. On the other hand, I intuitively expected that the embedding training would
benefit the model and that the neural metric would perform better than cosine similarity.

### Discussion **[2pt]**

Which HP I tuned?
Which had the most visible impact?
Did I use another techniques for more stable or better results?

I tuned the following hyperparameters: `random_emb`, `emb_training`, `emb_projection`, `vocab_size`, `final_metric`,
`lr`, `optimizer`, `batch_size`, `lr_scheduler`.
![img.png](img/img.png)
According to the parameter importance, the most important hyperparameter was learning rate. This result can be a bit
misleading
though, since more than 1000 experiments were run (over 3000 in total) and the parameter importance is calculated only
from 1000 runs.

It is kinda hard to say what hyperparameter value was the best overall, since there were so many experiments and the
figures
on wandb are made only from the last 50 runs. Therefore, I can´t for example see any other learning rate than 0.01.

For the learning rate, 0.01 was the best value for most configurations (3.8 test loss on average). At first, I seemed
that the smaller the learning rate,
the better, but 0.1 performed worse than 0.01 (4.7 test loss on average). I´m guessing that in the learning rate
scheduler configuration could
play a big role here (but I did not have time to test it).

Final metric also played a big role in the results. The neural metric performed better (3.8 test loss on average) than
cosine similarity (6.2 test loss on average) in most cases. What surprised me was that there were no good combinations
with the cosine similarity metric, I thought that with good embeddings, the cosine similarity would perform better than
it did.

Adam optimizer performed better (4.1 test loss on average) than SGD (6.0 test loss on average vs 4.7 test loss on average). I´m guessing that the
momentum in Adam helped the model to converge faster.

The batch size also played a big role in the results. I originally tested only batch size 1000, but had trouble hitting
test loss under 2. Lowering the batch size to 100 helped a lot. As I mentioned earlier, It would be interesting to test
even smaller batch sizes.

I would expect that the embeddings would play a big role as well since without training the random embeddings don´t make
much sense. But the overall results were quite similar, non-random embeddings performed just a bit better (4.9 test loss
on average vs 5.0 test loss on average). The same applies to the embedding training itself, the results were quite
similar (4.9 test loss on average vs 5.1 test loss on average). This goes against my intuition, I would expect way more
distinct results.

Lastly the embedding projection layer did not help the model that much. The results showed that the model performed
better with the projection layer (4.9 test loss on average vs 5.0 test loss on average). Which is quite surprising since
the best results were achieved without the projection layer. On the other hand, I would expect that the projection layer
would help the model to learn better.

I also think that it would be interesting to test more vocab sizes since they could be quite
impactful on the results as well.

# To Think About:

## Practical Question

1. Compare both similarity measures (Fully-connected, cosine):

    - speed of convergence,

    - behaviour in first epochs,

    - accuracy.

2. Does the size of the effective vocabulary affect the results? How?

3. Have you sped up debugging the system somehow?

4. Can we save some processing time in the final test (10 runs) without
   affecting the results?

5. What is the role of UNK and PAD token in both models?

6. Can you name some hints for improvement of our models?

7. Can we count UNK and PAD into sentence representation average
   embedding? Does it affect the model?

8. What is the problem with output scale with the neural network?

9. What is the problem with output scale with cosine?

10. What is the best vocab size? Why? Task8

11. What is the best learning rate? Why?

12. Which hyper-parameters affect memory usage the most?

## Theoretical Questions

1. Is it better to train embeddings or not? Why?

2. Is it important to randomly shuffle the train data, test data? Why?
   When?

3. What is the reason for comparing MSE on train dataset or testing
   dataset with Mean of training data or mean of testing data?

4. Can you name similar baselines for other tasks (Sentiment
   classification, NER, Question Answering)?

                                            

