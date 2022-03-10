# TabTransformer in Predicting Defaults of Telco Customers

This repo contains the code necessary to train and test a TabTransformer model on whether a customer will default on their telco payments. 

Model: [TabTransformer](https://arxiv.org/abs/2012.06678)
- Paper: https://arxiv.org/pdf/2012.06678.pdf
- Github: https://github.com/lucidrains/tab-transformer-pytorch

Library: [pytorch-widedeep](https://pytorch-widedeep.readthedocs.io/en/latest/index.html)

## Quick Start

1. Clone the repository
2. Ensure you have Docker Desktop and Postman installed on your machine
3. Open a terminal in the directory of the cloned repo
4. run ```docker build -t <image_name> .```
5. run ```docker run -d --name <container_name> -p 80:80 <image_name>```
6. Open Docker Desktop and navigate to Containers/Apps. Make sure the container with ```<container_name>``` is running
7. Open Postman
8. Send a GET request to ```http://0.0.0.0:80/accuracy``` to run the model and retrieve its accuracy
   - You may select the depth/number of transformers by specifying an integer, otherwise the recommended depth of 6 is used
   - E.g. To select depth = 10, ```http://0.0.0.0:80/accuracy/10```

## Data Cleaning and Feature Engineering

```TabTransformer.ipynb``` is a Jupyter Notebook that includes more comments and visualisations alongside the code explaining some of my thought process.
You should be able to view them on GitHub without having to run the notebook

```DataCleaning&FeatureEngineering.py``` includes code that generates the pickle files necessary for the predictive model. Does not need to be run before using the API

## Other References

https://github.com/antecessor/TabTransformer
https://github.com/jrzaurin/tabulardl-benchmark
https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b![image](https://user-images.githubusercontent.com/65173208/157682865-491ebfb2-8bcb-4d43-a841-c457a8e60ab7.png)
