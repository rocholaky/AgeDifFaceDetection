
# Facial Recognition with  Age Robustness

Now a days Facial Recogition systems have great accuracies on different datasets, with precisions near 90%. Specifically on the Facial Recognition task, high precision is very important because of the type of solutions that can be solved using this type of models. This solutions may: be access to private facilites using just the face, phone unlocking and recognition of people in grocerie stores. 

The use of Age difference makes the recognition of faces much more difficult for the model. So if a model is robust in detecting faces with age difference, the features it uses for the prediction must be much better than the ones other type of models use, because this features endure time. 


<img src="FacialRecog.png" alt= "Example" width=800 height=600 class="center">

## Methology:
One of the many ways to solve this type of problems is using a Siamise Network Architecture, in this type of networks we make use of a feature backbone, in this case we use the <a href=https://keras.io/api/applications/vgg/>VGG16</a> as the model from which we will extract the features from, by doing this we skip the need to train the model over a big dataset, instead we take advantage that this model has already been trained on and is open-source. By just adding a few Dense layers after the backbone we can train just this two layers in order to apply the Siamise Network Architecture.  
<img src="SiamiseNetwork.png" alt="Model" width=800 height=400 class="center">

This model has a lot of advantages, first of all we can train it with a very little amount of resources with the use of feature extraction, second, as the model tries to find a similarity function for the problem, we can train this type of structures just with binary cross-entropy (avoiding class imbalance and shortage).   

## Installation

This repo can be runned just by running the following command

```bash
  conda create --name AgeDif python=3.8
  conda activate AgeDif
  cd <path of repo>
  pip install -r requirements.txt
```
    
## Authors

- [@rocholaky](https://www.github.com/rocholaky)

