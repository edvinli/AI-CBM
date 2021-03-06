# AI-CBM
## Deep learning for circular economy

In this repo we open source models trained to predict auction end prices using online auction data from Sweden.

The data used in this study was collected from the clothing category from a Swedish online auction site (Tradera). The whole dataset contains 88,511 items and was randomly split into a training set of size 70,807 and a test set of size 17,704. Each item in the dataset has an end price, a title description, a text description, and an image, all uploaded by users themselves. The training set was further split into 67,267 (95%) training samples and 3541 (5%) validation samples.

For the text descriptions, we have experimented with three different types of representations: unigrams, bigrams and [Swedish CLIP embeddings](https://github.com/FreddeFrallan/Multilingual-CLIP). We use the pre-trained CLIP vision model (a ResNet RN50x4) to create image representations.

## Models

Three models are available in this repo:

* A text model with bigram input representations (dim=100,000)
* A vision model with pre-trained ResNet RN50x4 images as input (dim=640)
* A text+vision model which uses both of the above representations as input (dim=100,640)

The model is a one layer MLP, which either takes bigram input representations of Swedish text (DIM_IN=100,000) a [pre-trained CLIP RN50x4](https://github.com/openai/CLIP) image representation (DIM_IN=640) or a concatenation of them both, text + img (DIM_IN=100,640).

Pytorch models (.pth) can be found [here](https://github.com/edvinli/AI-CBM/tree/main/models). Read more about how to load Pytorch models [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html). 

Below follows Pytorch code for the model.


    DIM_IN = 640 #640, 100000 or 100640
    n_classes = 9
    class CbmClassifier(nn.Module):

    def __init__(self, n_classes, DIM_IN):
        super(CbmClassifier, self).__init__()

        self.input = nn.Linear(DIM_IN, 256)
        self.fc1 = nn.Linear(256,n_classes)
        self.activation = nn.ReLU()

    def forward(self, input_rep):
        
        x = self.input(input_rep)
        x = self.activation(x)
        x = self.fc1(x)
        return x


## Results

| Representation | Accuracy | Top 2 accuracy  |
| ----------- | ----------- | ----------- |
| Clip image  | 30.25       | 47.56 |
| Clip text   | 32.83       | 51.80 |
| Clip text+image | 33.86 | 53.06 |
| Bigram | 36.08 | 56.03 |
| Bigram + clip image | 37.2 | 57.77 |

This can be compared to mean human accuracy of 18.75%, where 32 humans tried to estimate the end prices using only images.


| Class | Price range (SEK) | 
| ----------- | ----------- | 
| 0 | 1-34 | 
| 1 | 35-49 | 
| 2 | 50 | 
| 3 | 51-79 |
| 4 | 80-103 |
| 5 | 104-154 |
| 6 | 155-249 |
| 7 | 250-400 |
| 8 | 400+ |

Confusion matrix of the best perfoming model normalized over predictions. Classes are approximately equal in sizes.

![confusion_matrix_9classes](https://user-images.githubusercontent.com/16919172/162403537-261fa561-817b-4c6b-a887-b305e71b0e2a.png)
