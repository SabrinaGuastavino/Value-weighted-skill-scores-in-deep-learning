# Value-weighted-skill-scores-in-deep-learning

Major softwares: Keras 2.4.3, Tensorflow 2.3.0, Python 2.7.15

We provide the experiments shown in [1] concerning four applications:

(1) Pollution forecasting problem (pollution folder)

(2) Solar flare forecasting problem (solar_flare folder)

(3) Stock prize foreacsting problem (stock_prize folder)

(4) IoT data stream forecasting problem (iot_datastream folder)

Each folder contains:
- a demo file .ipynb
- an utilities folder 
- a data folder
- a prediction folder

In detail

(1) pollution folder: 

    - pollution_experiments_demo.ipynb includes the experiments concerning the problem of predicting whether PM2.5 concentration will exceed a fixed
      threshold associated to a condition of severely polluted air. 
      The demo shows results obtained by using the following classical machine learning methods:
      
      - Logistic Regression (LR)
      
      - Support Vector Machine (SVM)
      
      - a Fully connected Neural Network (NN)
      
      Furthermore, the demo shows results obtained by applying the ensemble strategy proposed in [1] and the early stopping strategy when an LSTM network
      is trained:
      
      - ensemble strategy based on TSS optimization
      
      - ensemble strategy based on wTSS optimization
      
      - early stopping strategy based on TSS optimization
      
      - early stopping strategy based on wTSS optimization
      
    - data folder includes the data file pollution.csv (the data set comes from the University of California at Irvine (UCI), available at
      https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data via the data interface released by the US embassy in Beijing)
      
    - utilities folder includes utilities_forecasting.py and save_variables folder. The utilities_forecasting.py file mainly includes all the necessary
      functions for implementing the ensemble strategy based on the optimization of quality-based and value-weighted skill scores and for the slection of
      the patience parameter of the early stopping strategy. The save_variables folder includes the variables which are saved during the ensemble
      procedures.
      
    - prediction folder includes the predictions on the test set obtained by LR (named y_pred_LR.npy), by SVM (named y_pred_SVM.npy) and by NN (named
      y_pred_nn.npy), the checkpoints folder which includes the LSTM models saved during the training process on 100 epochs and the final LSTM models
      saved during the training process when the early stopping strategy is applied with the keyword patience equal to 10,20,30,40 and 50.
      
 (2) solar_flare folder: 
 
    - solar_flare_prediction_experiments_demo.ipynb includes the experiments concerning the problem of forecasting solar flares of GOES class C1 and
      above (C1+ flares), M1 and above (M1+ flares) and X1 and above (X1+ flares) 
      The demo shows results obtained by applying the ensemble strategy proposed in [1] and the early stopping strategy when a deep multi-layer
      perceptron is trained:
      
      - ensemble strategy based on TSS optimization
      
      - ensemble strategy based on wTSS optimization
      
      - early stopping strategy based on TSS optimization
      
      - early stopping strategy based on wTSS optimization
      
    - data folder includes the data panels .pkl files which include 23 features extracted from magnetogram images of active regions (as the ones recorded
      by the Helioseismic and Magnetic Imager (HMI)) in the time range between 09/15/2012 to 09/07/2017 and the the label panels .pkl files which include
      the labels associated to each feature vector.
      
    - utilities folder includes utilities_solar_flare_forecasting.py and save_variables folder. The utilities_solar_flare_forecasting.py file mainly
      includes all the necessary functions for implementing the ensemble strategy based on the optimization of quality-based and value-weighted skill
      scores and for the slection of the patience parameter of the early stopping strategy. The save_variables folder includes the variables which are
      saved during the ensemble procedures.
      
    - prediction folder includes the checkpoints folder which includes the deep multi-layer perceptron models saved during the training process on 100
      epochs and the final deep multi-layer perceptron models saved during the training process when the early stopping strategy is applied with the
      keyword patience equal to 10,20,30,40 and 50.
      
  (3) stock_prize folder: 
  
    - stock_prize_prediction_experiments_demo.ipynb includes the experiments concerning the problem of forecasting the "down" movements in stock prizes
      relying on information concerned the daily closure prizes in the database put at disposal by Yahoo Finance.
      The demo shows results obtained by applying the ensemble strategy proposed in [1] and the early stopping strategy when a deep multi-layer
      perceptron is trained:
      
      - ensemble strategy based on TSS optimization
      
      - ensemble strategy based on wTSS optimization
      
      - early stopping strategy based on TSS optimization
      
      - early stopping strategy based on wTSS optimization
      
    - utilities folder includes utilities_stock_prize_forecasting.py and save_variables folder. The utilities_stock_prize_forecasting.py file mainly
      includes all the necessary functions for implementing the ensemble strategy based on the optimization of quality-based and value-weighted skill
      scores and for the slection of the patience parameter of the early stopping strategy. The save_variables folder includes the variables which are
      saved during the ensemble procedures.
      
    - prediction folder includes the checkpoints folder which includes the LSTM models saved during the training process on 100
      epochs and the final LSTM models saved during the training process when the early stopping strategy is applied with the
      keyword patience equal to 10,20,30,40 and 50.
      
  (4) iot_datastream folder: 
  
    - iot_datastream_prediction_experiments_demo.ipynb includes the experiments concerning the problem of predicting the usage of light from IoT data
      stream
      The demo shows results obtained by applying the ensemble strategy proposed in [1] and the early stopping strategy when a CNN-LSTM
      is trained:
      
      - ensemble strategy based on TSS optimization
      
      - ensemble strategy based on wTSS optimization
      
      - early stopping strategy based on TSS optimization
      
      - early stopping strategy based on wTSS optimization
      
    - data folder includes the data file iot_telemetry_data.csv, the dataset is available at 
       https://www.kaggle.com/garystafford/environmental-sensor-data-132k
       
    - utilities folder includes utilities_iot_datastream_forecasting.py and save_variables folder. The utilities_iot_datastream_forecasting.py file
      mainly includes all the necessary functions for implementing the ensemble strategy based on the optimization of quality-based and value-weighted
      skill scores and for the slection of the patience parameter of the early stopping strategy. The save_variables folder includes the variables which
      are saved during the ensemble procedures.
      
    - prediction folder includes the checkpoints folder which includes the CNN-LSTM models saved during the training process on 100
      epochs and the final CNN-LSTM models saved during the training process when the early stopping strategy is applied with the
      keyword patience equal to 10,20,30,40 and 50.
      

