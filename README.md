# Ecommerce-Product-Price-Prediction-Engine
Built an e-commerce product price prediction engine using Transformer embeddings, engineered catalog features, TF-IDF, LightGBM and stacking to estimate optimal product prices from titles, descriptions, pack size and unit data, trained with cross-validation and optimized using SMAPE.

E-commerce Product Price Prediction Engine:

A production-grade machine learning system that predicts the optimal price of e-commerce products using catalog text, pack size, and unit information. This project was built for the Smart Product Pricing Challenge 2025, where the task was to predict prices for 75,000 unseen products based only on their metadata.

Problem Statement:

E-commerce marketplaces must continuously price millions of products, including newly listed SKUs that have no historical sales data. Pricing depends on factors such as product type, brand, quantity, pack size, unit volume (ml, grams, liters), and whether the product is a bundle or offer.

The challenge was to build a system that could infer this information from raw catalog text and accurately predict product prices. The evaluation metric used was SMAPE (Symmetric Mean Absolute Percentage Error), which measures relative pricing accuracy and penalizes both overpricing and underpricing.

Solution Overview:

This project implements a multi-model ensemble pricing engine that combines natural language understanding, feature engineering, and gradient-boosted decision trees. Instead of relying on a single model, the system learns from three different perspectives: keyword-level signals, semantic meaning, and quantitative product attributes. These models are combined using stacked generalization to produce a final calibrated price prediction.

Data:

Each product contains:

sample_id – unique identifier

catalog_content – product title, description and pack information

image_link – URL of product image

price – target variable (only in training set)

The training set contains 75,000 labeled products and the test set contains 75,000 unlabeled products.

Feature Engineering:

The system extracts both semantic and numerical pricing signals.

Text-based features include:

Transformer embeddings using MPNet

TF-IDF unigrams and bigrams

Token frequency patterns

Out-of-fold target encoding for high-impact keywords

Numeric and structural features include:

Pack quantity (for example, Pack of 6 or 2x)

Unit size (ml, grams, liters, kg, oz)

Digit count

Word count

Text length

Offer and bundle indicators

Punctuation density

These features allow the model to distinguish between small items, bulk packs, premium products, and discounted bundles.

Model Architecture:

The system uses a two-model ensemble with stacking.

TF-IDF with LightGBM:
This model captures keyword-level pricing signals such as “combo”, “refill”, “premium”, and “value pack”.

Transformer embeddings combined with engineered features and LightGBM:
This model learns the semantic meaning of product descriptions together with numerical product attributes.

Ridge regression stacking:
Predictions from both models are combined using Ridge regression to produce a single calibrated price estimate.

All models are trained using 5-fold cross-validation to prevent data leakage and ensure robustness.

Training Objective:

Prices are log-transformed during training and optimized using RMSE. Final evaluation is done using SMAPE, which ensures the model is accurate for both low-priced and high-priced products.

Why This Works:

This engine mirrors how real marketplaces price products. It understands what the product is using NLP, how much the customer gets using quantity and unit extraction, and market value patterns from similar listings. By combining multiple models, it achieves more stable and accurate predictions.

How to Run:

Install the required dependencies

Place train.csv and test.csv in the working directory

Run the main pipeline script

A submission file with predicted prices will be generated

Output:

The final output is a CSV file with two columns:
sample_id and price, containing one predicted price for every test product.

Applications:

This system can be used for:

New product pricing

Seller price recommendations

Catalog quality checks

Inventory valuation

Automated marketplace pricing engines
