# 1. Project Title
Multi-Label-Classifiction-NLP-LSTM-BBC

# 2. Project Description
Text documents are one of the richest sources of data for businesses. We’ll use a public dataset from the BBC comprised of 2315 articles, each labeled under one of 5 categories: business, entertainment, politics, sport or tech

# 3. Data Description
Contains the ![data](https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv) to be used for model building and model testing. 

The dataset is broken into 70/30 records for training & testing. The goal will be to build a system that can accurately classify previously unseen news articles into the right category.

Model is evaluated using Accuracy as a metric.

# 4. How to Install and Run the Project
This project was run in Conda environment using Spyder IDE (Interactive Development Environment). Several essential libraries required to be installed prior to running the code. 

For computer that does not have GPU, you might want to use external workspace such as ![Google Colab](https://colab.research.google.com/?utm_source=scs-index) for running your scripts which no additional modules installation are required.

# 5. How to Use the Project
The full ![code](https://github.com/GeoLai/Multi-Label-Classifiction-NLP-LSTM-BBC/blob/main/bbc-articles-nlp.py) can be viewed here as a reference. For cleaner code construction, I have written Long Short Term Memory Deep Learning architecture in a separate ![module](https://github.com/GeoLai/Multi-Label-Classifiction-NLP-LSTM-BBC/blob/main/bbc_nlp_module.py) file which some of tuning can be done during model training.

Visuals are provided which were generated from data visualization of the data, training curves which displayed in Tensorboard, snippet of training scores, learning architecture, model parameters where located in the ![image](https://github.com/GeoLai/Multi-Label-Classifiction-NLP-LSTM-BBC/tree/main/images) folder.

# 6. Include Credits
Credits to owner of the dataset and the provider.

Data provided by ![Susan Li](https://github.com/susanli2016). Purpose for solution ![PyCon Canada 2019](https://github.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial)

# 7. Add a License
No license

# 8. Badges
### These codes are powered by
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

### If you found this beneficial and help you in way or another. You may want to
![BuyMeAWanTanMee](https://img.shields.io/badge/Buy%20Me%20a%20Wan%20Tan%20Mee-ffdd00?style=for-the-badge&logo=buy-me-a-wantanmee&logoColor=black)

