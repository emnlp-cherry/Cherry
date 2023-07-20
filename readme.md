# Cherry: On Detecting Cherry-picking in News Coverage Using Large Language Models
## Abstract: 
Cherry-picking refers to the deliberate selection of evidence, statements, or facts that favor a particular viewpoint while ignoring or distorting evidence that supports an opposing perspective. Manually identifying instances of cherry-picked statements in news stories can be challenging, particularly when the opposing viewpoint's story is absent. This study introduces Cherry, an innovative approach for automatically detecting cherry-picked statements in news articles by finding missing important statements in the target news story. Cherry utilizes the analysis of news coverage from multiple sources to identify instances of cherry-picking. Our approach relies on language models that consider contextual information from other news sources to classify statements based on their importance to the event covered in the target news story. Furthermore, this research introduces a novel dataset specifically designed for cherry-picking detection, which was used to train and evaluate the performance of the models. Our best performing model achieves an F-1 score of about 89\% in detecting important statements when tested on unseen set of news stories. Moreover, results show the importance incorporating external knowledge from alternative unbiased narratives when assessing a statement importance.
## contentes:
This repo contains three main directories:
1. cherry_baseline_CUDA: contains the BERT variant of the model.
2. cherry_baseline_CUDA_nocontext: contains the BERT variant of the model that accepts only the statement, without context in th einput sequence.
3. cherry_longformer_CUDA: contains the Longformer variant of the model.
4. cherry_picking_detection: contains the scripts required to run the full-end-to-end cherry-picking detection pipeline.

## Install and run:
### Training
1. make sure to have the following packages installed on your environment with python 3.10.4: </br>
huggingface-hub       0.10.0rc0 </br>
matplotlib            3.6.0 </br>
nltk                  3.7 </br>
numpy                 1.23.1 </br>
pandas                1.4.3 </br>
scikit-learn          1.1.2 </br>
scipy                 1.9.1 </br>
sentence-transformers 2.2.2 </br>
torch                 1.12.1 </br>
tqdm                  4.64.0 </br>
torchmetrics          0.9.3 </br>
transformers          4.22.1 </br>
2. To run traning for any of the model variants above, run the following command from inside the variant's main directory: </br>
python main.py </br>
3. To modify the paramters, adjust the values in the main.py files under "paramters".
Each of the variants directories contians the code and the data sets used in the four different classification configurations. You do not need to reset data paths, just choose the classification configuration number in the paramter list in main.py.

### End-to-end cherry-picking detection:
To run the end-to-end detection pipleine using the top-performing model, install the large files from [here](https://drive.google.com/drive/folders/1bJTSS5HJdb2GGEmfnOciIHnn9U6qOFg4?usp=sharing) and place them under the cherry_picking_detection directory, then run the following command in the directory: </br>
python spot_cherry_picking.py </br>
The script will run the pipline on the clustered and prepared data file "bias_analysis_events_clustered_wpredictions.json"
This file contains all the inference data comprised of the 2453 unseen events preprocessed, clustered, and also conatains the results of inference from the top-performing model.


