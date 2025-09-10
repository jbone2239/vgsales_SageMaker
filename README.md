# This project analyzes 15,000+ structured video game sales records to predict top-selling games by region. Using Random forest, the model identifies which features (genre, platform, publisher, etc.) best predict regional success. The trained model is deployed on Heroku for interactive predictions.

- This repository contains the files for deploying the Video Game Sales prediction model on AWS SageMaker using a custom SKLearn container.
- The trained Random Forest model is supported by label_encoder.pkl for categorical feature encoding and consistent predictions.
- The main notebook, vgsales_SM.ipynb, walks through the end-to-end SageMaker workflow including training, containerization, and deployment.
- The script vgsales_model.py provides helper functions to load the model and handle prediction logic inside the container.
- A screenshot endpoints.PNG documents the SageMaker endpoints created during deployment, confirming that the app was successfully served in AWS.
