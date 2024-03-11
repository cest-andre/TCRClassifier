# **CSC 7343 - Project Report**

Group: Muhammad Hussain and André Longon

## **Pretraining**

To deviate from the predictive pretraining of GPT and BERT, we decided to formulate a contrastive learning self-supervised objective. Similar to contrastive learning in vision, we generated augmented data points from each batch. For each datapoint (sequence of tokens), we randomly replace one of the non-reserved tokens which represent the antigen and TCR sequences with the special &lt;mask&gt; token. We perform this augmentation twice to get two projections of each data point.

We then pass the two projections through the model and extract the first vector of the output sequence. We then compute the loss using the contrastive loss function used in SimCLR. Where _sim(u,v)_ is the dot product between two normalized vectors.

![download](https://github.com/wahaj-47/TCRClassifier/assets/8758774/8707a103-3a29-4825-be50-66eea2273d4e)

The loss function is dependent on the batch size because the batch size determines how many pairs of positive and negative samples are considered in each iteration of training. When you have a larger batch size, the average has the effect of considering the agreement or disagreement across more pairs. To do this on a limited resource budget we used gradient accumulation. We average the gradient of N steps and then update the model, instead of every forward-backward pass.

This approach was inspired by contrastive learning in computer vision where a model is encouraged to embed an image and its noised/blurred counterpart similarly. Also of inspiration is the intuition that we are somewhat tolerant to omitted letters in reading text. By training the network to be resilient to augmentations and to distance different points, we hope the network will build some basic knowledge of the data before its supervised fine-tuning. Let us see if we accomplish this.

## **Results**

The model with no pre-training, achieves an average accuracy of 52.4% after training for 3 epochs with a batch size of 1024. With pre-training, the model was able to achieve an accuracy of 62.4% after 3 epochs of pre-training and 3 epochs of fine tuning with a batch size of 1024. The optimizer used for pretraining was SGD with a learning rate of 1e-3 and the optimizer used for supervised fine-tuning was AdamW with a learning rate of 1e-5.

|     | Epochs | Learning rate | Batch size | Avg Accuracy % |
| --- | --- | --- | --- | --- |
| No pre-training | 3   | 1e-5 | 1024 | 52.4 |
| Pre-training | 3, 3 (Fine-tuning) | 1e-3, 1e-5 (Fine-tuning) | 1024 | 64.7 |

_Avg accuracy reported is from 3 fold cross validation_

|     | Iteration #1 | Iteration #2 | Iteration #3 |
| --- | --- | --- | --- |
| No pre-training | 57.9 | 56.9 | 42.1 |
| Pre-training | 60.3 | 66.5 | 67.4 |

_Accuracy (%) each iteration of 3 fold cross validation_

## **Discussion**

The approach poses challenges in hyperparameter tuning, demands considerable computational resources and makes it necessary to use large batch sizes. The exploration of an effective set of hyperparameters, such as choice of optimizer, temperature, batch size and gradient accumulation iterator are crucial for successful implementation. The reliance on large batch sizes adds to the computational overhead influencing the scalability of the approach.

While we achieved a performance improvement, it may not be significant enough to justify the additional computation, especially when scaled up on a larger dataset. Further work could explore better augmentation schemes then the masking we presented. Compared with images, it was difficult to conceptualize augmentations on tokens that made intuitive sense from an embedding similarity standpoint. The slight performance boost is encouraging to pursue alternative augmentations and even different contrastive loss objectives.

## **References**

_A Simple Framework for Contrastive Learning of Visual Representations_, arxiv.org/pdf/2002.05709v3.pdf. Accessed 4 Dec. 2023.

Nikolas Adloglou, _Implementing SimCLR with pytorch lightning,_ [theaisummer.com/simclr](https://theaisummer.com/simclr) Accessed 4 Dec. 2023.

Konkle, T., Alvarez, G.A. _A self-supervised domain-general learning framework for human ventral stream representation_. _Nat Commun_ 13, 491 (2022).
