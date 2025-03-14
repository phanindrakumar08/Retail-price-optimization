# Retail Price Optimization

In today's competitive retail market, setting the right price for products is crucial. This project focuses on retail price optimization using machine learning techniques to predict customer satisfaction scores. This is a crucial part of developing dynamic pricing strategies, leading to increased sales and customer satisfaction.

## Problem Statement

Our task is to develop a model that predicts the optimal price for a product based on various factors. This prediction would enable us to make an informed decision when pricing a product, leading to maximized sales and customer satisfaction.

The dataset's diverse features like product details, order details, review details, pricing, competition, time, and customer details provide a comprehensive view for our price optimization task.

By analyzing this information, we aim to predict the optimal price for retail products. This would aid in making strategic pricing decisions, thereby optimizing retail prices effectively.

## Sample Dataset

The sample dataset includes various details about each order, such as:

- **Product details**: ID, category, weight, dimensions, and more.
- **Order details**: Approved date, delivery date, estimated delivery date, and more.
- **Review details**: Score and comments.
- **Pricing and competition details**: Total price, freight price, unit price, competitor prices, and more.
- **Time details**: Month, year, weekdays, weekends, holidays.
- **Customer details**: ZIP code, order item ID.

## 🐍 Python Requirements

Let's jump into the Python packages you need. Within the Python environment of your choice, run:

```bash
git clone https://github.com/phanindrakumar08/Reatail-price-optimization.git
pip install -r requirements.txt
```

Starting with ZenML 0.20.0, ZenML comes bundled with a React-based dashboard. This dashboard allows you to observe your stacks, stack components, and pipeline DAGs in a dashboard interface. To access this, you need to launch the ZenML Server and Dashboard locally. First, install the optional dependencies for the ZenML server:

```bash
pip install "zenml[server]"
zenml up
```

If you are running the `run_cid_pipeline.py` script, you will also need to install some integrations using ZenML:

```bash
zenml integration install mlflow -y
zenml integration install bentoml
```

## 🛠 Configuring the ZenML Stack

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and BentoML model deployer as a component. Configuring a new stack with the two components is as follows:

```bash
zenml experiment-tracker register mlflow_tracker_mlops --flavor=mlflow
zenml model-deployer register bentoml_deployer --flavor=bentoml
zenml stack register local_bentoml_stack \
    -a default \
    -o default \
    -d bentoml_deployer \
    -e mlflow_tracker_mlops --set

zenml stack set local_bentoml_stack
```

## 🚀 Training Pipeline

Our standard training pipeline consists of several steps:

1. **ingest**: Ingests the data from the database into the ZenML repository.
2. **categorical_encoder**: Encodes the categorical features of the dataset.
3. **feature_engineer**: Creates new features from the existing features.
4. **split**: Splits the dataset into train and eval splits.
5. **train**: Trains the model on the training split.
6. **evaluate**: Evaluates the model on the eval split.
7. **deploy**: Deploys the model to a BentoML endpoint.

This structured approach ensures efficient price optimization and strategic decision-making.

---

🎯 **Happy Coding!** 🚀
