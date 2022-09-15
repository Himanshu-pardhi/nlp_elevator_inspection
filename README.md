# Imbalance Multiclass Text Classification using Machine Learning Models


### Description  
This project is implemented based on Machine learning Models. Using a `LinearSVC`, `Random Forest` and `Logistic Regrission` classifiers to predict the correct class for the given text.    

### Repo structure  

* `nlp.ipynb`: `jupyter notebook` file which has scripts and instructions on exploring the dataset, preprocessing, identifying the models and training them.  

* `functions.py`: All functions used to implement the `nlp.ipynb`.  

* `app.py`: Scripts of streamlit application for deploying the model.  

* `requirements.txt`: file that has all packages for running program.


### Pipeline Review
![workflow](https://github.com/Himanshu-pardhi/nlp_elevator_inspection/blob/Rana/assets/nlp_workflow.png)

### Installation
#### step 1/ clone the repo
`git clone 'https://github.com/Himanshu-pardhi/nlp_elevator_inspection'`

#### step 2/ Create Virtual Environment

#### step 3/ Install Required Packages
Use `pip install requirements.txt` for installation.

### Usage  
Follow these steps to run the program:

1. Go to `nlp.ipynb` and load the dataset then run and follow the steps in the file.

2. For deployment the web-application for this ML-Model with `streamlit` go to file `app.py`  
in the terminal run this command `streamlit run app.py`.

![webapp](https://github.com/Himanshu-pardhi/nlp_elevator_inspection/blob/Rana/assets/streamlit1_img.png)


### Future Improvements  
1. Trian more data with several classes.  
2. Train a Deep Learning Model.  
3. Deploy the application to the server.


