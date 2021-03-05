## Template Generation with Deep Transfer Neural Network
This repository provides codes and scripts for the experiments of log template generation with deep transfer learning

If you publish material based on LogDTL, then, help others to obtain the same data sets and replicate your experiments
by citing LogDTL original paper. Here is the suggested format for referring to LogDTL:

```code 
T. Nguyen, S. Kobayashi, K. Fukuda. "LogDTL: Network Log Template Generation with Deep Transfer Learning". 6 pages.
AnNet 2021. Bordeaux, France. 2021.
```



### Dataset 
```code 
Since our dataset is sensitive and proprietary, we use open-source dataset (https://github.com/logpai/logparser) to demonstrate the model is working. 
Assumption that
    + The Linux data is open-source dataset (data/open_source/linux/)
    + The Windows data is proprietary dataset (data/proprietary/windows/)

+ Some representation of log data:
    + Raw Log: user daniel log in at 7:09 AM, server unknown.
    + Template: user <*> log in at <*> <*>, server <*>.
        + The Special Character (the character represent the variable in the log) is: <*>
    + Event (or Clusters): This is a group of log that produce by the same template.
    + Event String: is the sequence of encoded words, for example
        + With template: user <*> log in at <*> <*>, server <*>.
        + EventStr is: DES VAR DES DES DES VAR VAR DES VAR 

If you want to test with your data, please look at our sample dataset (Linux and Mac), there are 3 type of files:
    + log_raw: file is the raw log file.
    + log_events.csv: file is the log templates for the whole log_raw file. This file is use for reference only. 
For open-source code, you can get it by looking at the source-code. For proprietary software, we have to manually 
make the templates for each raw log. Make sure you have 2 columns EventId and EventTemplate
    + log_templates.csv: will be used for training and testing the model. Make sure you have 3 columns: 
        + Content: the raw message in the log 
        + EventId and EventTemplate. The EventId make sure it matches with the file log_events.csv

So first, we make training and testing files for both linux and windows: 
    + linux/2k_log_train_test.csv
    + windows/2k_log_train_test.csv
including only 4 columns: Content, EventId, EventTemplate, EventStr
```

### Models
- Located at: models/transfer/...
- There are 7 models have been developed including
    + crf.py (simplest model)
    + gru.py (2nd simplest model, Word Embedding + Word-level GRU)
    + gru_crf.py (Word Embedding + Word-level GRU + CRF)
    + dgru.py (Word Embedding + Word-level GRU + Char-level GRU)
    + dgru_crf.py (word Embedding + Word-level GRU + Char-level GRU + CRF)
    + cnn_dgru.py (Word Embedding + Word-level GRU + Char Embedding-based CNN + Char-level GRU)
    + cnn_dgru_crf.py (Word Embedding + Word-level GRU + Char Embedding-based CNN + Char-level GRU + CRF)
    + The last model cnn_dgru_crf.py is the model: dtnn.py (deep transfer neural network)
    + model_util.py: is the abstract code for all above models. All above models inherit some methods from this class.
    + sample.py: is the code for taking ramdom samples from dataset for training and testing set.

### Main files (Running files)
- crf.py
    + The simple CRF model, test dataset is Linux or Windows 
- dnn.py
    + The Deep Neural Network model, test dataset is Linux or Windows
- dtnn.py
    + Our proposed model Deep Transfer Neural Network, test dataset is Windows in which source-task is Linux and 
      target-task is Windows.
    + DTNN model learning knowledge from Linux dataset to generate a model, then using this model to re-train and 
      test with Windows dataset.
- dtnn_0.py
    + This is still DTNN model but without training-set of target task. 
    + In this case, DTNN model learning knowledge from Linux dataset to generate a model, then using this model to 
      predict the testing dataset of Windows dataset. 

- The configuration for all models stay in config.py 

- There are some preprocessing stuffs need to be done before using calling there models. Depend on your dataset, 
  just prepare the required format as we talk in above section (Dataset). For our simple test case (Linux and 
  Windows), we do some pre-processing in the file: preprocessing_dataset.py


### Helper files 
- Dataset util: located at: models/utils/dataset_util.py
    + Including Word Embedding based on Gensim library 
    + Including all preprocessing stuffs from above processing files 

- Measurement Methods: located at: models/utils/measurement_util.py
    + Including all the errors, scores functions using in the paper.

### Results files
- Located at: data/results/dataset_name/ including: 
    + *-template_prediction.csv: Prediction template for all testing dataset
    + *-metrics.csv: The performance metrics based on groundtruth labels and predicted labels 

    
## How to run
1. Execute on normal computer without GPU:
    + python preprocessing_dataset.py
    + python crf.py
    + python dnn.py
    + python dtnn.py
   
2. Execute on GPU server:
    + python preprocessing_dataset.py
    + python crf.py  
    + THEANO_FLAGS='device=cuda0,floatX=float32' python dnn.py
    + THEANO_FLAGS='device=cuda1,floatX=float32' python dtnn.py



   
