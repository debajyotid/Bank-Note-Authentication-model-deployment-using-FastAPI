# Deploying a Bank Note Authentication model using FastAPI
## This problem is part of Kaggle: https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data
### The objective of the problem is to distinguish between forged bank notes and authentic notes. The data comprises of 4 features and 1 target columns, and then create a FastAPI web-app which can be invoked via localhost.

#### The features are: 'variance', 'skewness', 'kurtosis', and 'entropy'. The target column is labelled as 'class'.

The features were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera, usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object, gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.

This is a simple classification problem without much requirement for feature engineering. With a default RandomForest Classifier model, we were able to achieve >99% validation accuracy. The trained model is then saved as a .pkl file (classifier.pkl) for later use.

We create a app.py file, which can be run from localhost. The app requires 4 inputs, as the model has been trained on 4 features. Similar to a Flask App, we create a FastApi app, which when invoked using uvicorn (a ASGI server, different from the typical WSGI Flask server) will load the necessary method of the FastAPI app. We will finally use OpenAPI/Swagger for rendering a proper web-api interface, but it is also possible to invoke this FastAPI app locally.

The entire app file is invoked by uvicorn, on CLI, using uvicorn.run(app, host='127.0.0.1', port=8000) which instructs that uvicorn should run the app.py file on host:127.0.0.1 and on port:8000.

We have defined 2 methods: root (/) and (/predict), which can be invoked using http://127.0.0.1:8000/ and http://127.0.0.1:8000/predict or by simply using http://127.0.0.1:8000/docs or http://127.0.0.1:8000/predict?variance=2&skewness=3&curtosis=2&entropy=1. The /predict is used to invoke the saved classification model for prediction. If defines a function predict_banknote, which leverages a pydantic library based class BankNote, for accepting user-input in a .json format, and passes these values to the classification model for prediction.
