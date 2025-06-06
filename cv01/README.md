# KIV/ANLP assignment 01

## Implement Training Loop and Experiment with Hyper-Parameters

## Prerequisities

1. Instal PyTorch (https://pytorch.org/)
2. Run Hello world (cpu/gpu)
3. Create account on MetaCentrum

## Tasks

## Our team work  [0pt]

Complete missing parts and design clear interface for experimenting.

1. Use python argparser
2. Use wandb and log everything
3. For easy login and testing use environment variable WANDB_API_KEY
4. Run minimalistic hello world on MetaCentrum

## Individual work **[13pt in total]**

### Dataset Analysis **[1pt]**

Create histogram of classes in the dataset.
![img.png](img.png)

### Baseline analysis **[1.5pt]**

How would look accucary metric for **random model** and **majority class model**(returns only majority class as an
output)

_For the random model, the accuracy would be approximately 1/number_of_classes, thus approximately 0.1 since the classes
are balanced._

_For the majority class model, the accuracy would be the number of samples of the majority class divided by the total
number of samples.
The number of samples of the majority class is 1135, so the accuracy would be 1135/10000 = 0.1135 on the test dataset._

For the implementation see the code in the file `random_and_majority_model.py`.

Is there any drawback? Can we use something better, why?

_The drawback of the majority class model is that it is not able to predict any other class than the majority class.
Thus not providing much information about the data.
Same with the random model, it doesn't provide any useful information about the data.
However, they can serve as a good baseline for the model to beat._

_Pretty much any model would be better than these two models, as long as it takes the features into account._

1. Implement missing fragments in template main01.py
2. Implement 3-layer MLP with ReLU activation function **CF#Dense**
3. Run Experiments **[3pt]**
    1. Run at least 5 experiments with all possible combinations of following hyper-parameters
    2. Draw parallel coordinates chart and add image output into output section in this README.md

            `model: ["dense", "cnn"]`
            `lr: [0.1, 0.01, 0.001, 0.0001, 0.00001]`
            `optimizer: ["sgd","adam"]`
            `dp: [0, 0.1, 0.3, 0.5]`

Each experiment train at least for 2 epochs.

4. Utilize MetaCentrum **[3pt]**

   For HP search modify attached scripts and utilize cluster MetaCentrum.
   https://metavo.metacentrum.cz/

# My results

## Parallel Coordinate Chart with Appropriate Convenient Setup **[0.5pt]**

Draw parallel coordinate chart with all tunned hyper parameters

1. Show all your runs **[0.5pt]**
   ![W&B Chart 28. 9. 2024 13_51_18.svg](W%26B%20Chart%2028.%209.%202024%2013_51_18.svg)

2. Show only runs better than random baseline. **[0.5pt]**
   ![W&B Chart 28. 9. 2024 13_59_35.svg](W%26B%20Chart%2028.%209.%202024%2013_59_35.svg)
   _- used random model accuracy as a baseline (10%)_
   ![W&B Chart 4. 10. 2024 14_27_12.svg](W%26B%20Chart%204.%2010.%202024%2014_27_12.svg)
   _- models with accuracy higher than 90% shown_

## Table of my results **[1pt]**

1. show 2 best HP configuration for dense and cnn model
   (both configurations run 5 times and add confidence interval to the table)
2. add random and majority class models into the result table
3. mark as bold in the table

| model    |    dp |       lr | optimizer   | accuracy -+ 95% confidence   |
|:---------|------:|---------:|:------------|:-----------------------------|
| dense    |   0   |   0.001  | adam        | 96.95 +- 1.03                |
| dense    |   0.3 |   0.001  | adam        | 96.83 +- 1.05                |
| cnn      |   0   |   0.0001 | adam        | 98.51 +- 0.73                |
| cnn      |   0.5 |   0.001  | adam        | 98.81 +- 0.66                |
| random   | nan   | nan      |             | **cca 10.00**                |
| majority | nan   | nan      |             | **cca 11.00**                |

There weren't exactly 5 runs for each configuration, but concretely 10 9 8 8.

## Present all konvergent runs **[0.5pt]**

![W&B Chart 4. 10. 2024 16_27_54.svg](W%26B%20Chart%204.%2010.%202024%2016_27_54.svg)
Not all runs are shown, only the ones with accuracy higher than 99% - approximately 50 runs.
Most of those runs (45) are with the cnn model.

One of the best models is morning-sound-761 with the cnn model, lr=0.0001, dp=0, optimizer=adam, test accuracy=100.
![W&B Chart 4. 10. 2024 17_09_59.svg](W%26B%20Chart%204.%2010.%202024%2017_09_59.svg)

## Present all divergent runs **[0.5pt]**

![W&B Chart 4. 10. 2024 17_15_29.svg](W%26B%20Chart%204.%2010.%202024%2017_15_29.svg)
Above about 50 runs with accuracy lower than 20% are shown.
What is interesting is that there are many models with accuracy around 10% - the random model and majority model
accuracy.
Therefore it is possible that these models have similar behaviour to the random model or majority model.
Another interesting behaviour is that there were some models that had at some point accuracy above 10% but then it
dropped to 10%,
such as dark-capybara-590 and still-capybara-743.
![W&B Chart 4. 10. 2024 17_15_29(1).svg](W%26B%20Chart%204.%2010.%202024%2017_15_29%281%29.svg)
This accuracy decrease can generally be seen for all dense models.
There are also a few models at the bottom with low accuracy but a good accuracy trend, maybe they just need more epochs
to reach a higher accuracy.
![W&B Chart 4. 10. 2024 17_15_29(2).svg](W%26B%20Chart%204.%2010.%202024%2017_15_29%282%29.svg)
If grouped by learning rate, it can be seen that the models with learning rate 0.00001 are those slowly learning models,
so
maybe with bigger learning rate they would reach a higher accuracy faster.
The models with learning rate 0.0001 are probably the ones with decreasing accuracy, as mentioned before.
Maybe these models are overfitting the data.

## Discussion **[1pt]**

- Discuss the results.
- Try to explain why specific configurations work and other not.
- Try to discuss the most interesting points in your work.
- Is there something that does not make any sense? Write it here with your thoughts.

The task was successfully solved, the best models are shown in the table above.
It seems that the learning rate has the most influence both on test accuracy and test loss.
![img_1.png](img_1.png)![img_2.png](img_2.png)
And in deed higher learning rates seem to outperform lower learning rates (in general).
This can be due to the fact that the models with lower learning rates converge too slowly.
Learning rate 0.1 seems to be too high for the models, it is possible that is overshoots the minimum.
![W&B Chart 5. 10. 2024 15_37_15.svg](W%26B%20Chart%205.%2010.%202024%2015_37_15.svg)
Other important hyperparameters were model type and optimizer.
From the images below it can be seen that the cnn model outperforms the dense model and
the adam optimizer is generally better than the sgd optimizer.
![W&B Chart 5. 10. 2024 15_37_15(1).svg](W%26B%20Chart%205.%2010.%202024%2015_37_15%281%29.svg)
![W&B Chart 5. 10. 2024 15_37_15(2).svg](W%26B%20Chart%205.%2010.%202024%2015_37_15%282%29.svg)
That cnn models work best with adam (in this case) was also confirmed in the table of the best models.
![W&B Chart 5. 10. 2024 15_37_15(3).svg](W%26B%20Chart%205.%2010.%202024%2015_37_15%283%29.svg)
![W&B Chart 5. 10. 2024 15_37_15(4).svg](W%26B%20Chart%205.%2010.%202024%2015_37_15%284%29.svg)
We can conclude that adam works best with learning rate 0.001 and sgd with learning rate 0.1.
And that the cnn model works best with learning rate 0.001 and the dense model with learning rate 0.01 or 0.001 also.

## Try to explain why specific configurations works better than others.

From the analysis above it seems clear that best working combination is cnn model with adam optimizer and learning rate
0.001,
which is also the best model in the table above.
It is possible that the cnn model works better because the task was image classification and the cnn model is better
suited for this task.
Some dense models exhibited divergence, with accuracies dropping below 20%.
The patterns showing a brief rise in accuracy followed by a decline suggest these models might be overfitting early in
the training process or suffering from vanishing gradients in deeper layers.

As expected, models with lower learning rates exhibited slower learning. While some of these models demonstrated
a positive accuracy trend, they did not reach optimal performance in the given time and could benefit from additional
training.
On the other hand, too high of a learning rate (e.g., 0.1) likely caused models to "overshoot" the optimal minima during
gradient descent, which could explain the poor performance for some configurations.

In the end it is always about the hyperparameter combination as a whole, as the hyperparameters interact with each
other.

# Something to think about

1. How to estimate the batch size?
2. What are the consequences of using a larger/smaller batch size?
3. What is the impact of batch size on the final accuracy of the system?
4. What are the advantages/disadvantages of calculating the test on a smaller/larger number of data samples?
5. When would you use such a technique?
6. How to set the number of epochs when training models?
7. Why do the test and train loss start with similar values? Can initial values have any special significance?
8. Is there any reason to set batch_size differently for train/dev/test?
9. When is it appropriate to use learning rate (LR) decay?

                                                                   

