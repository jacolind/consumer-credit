# Consumer-credit

## Problem statement

This is a binary classification problem. 
The company issues short term consumer credit 
(people buy shoes today but pay back later).
If consumers do not pay back, the company will have higher costs. 
If consumers do not buy, the company will have lower revenue.
The company must balance these two sides in the classification algo of who gets approved.

## Method

The approach is the following:

1. Select variables, usually all of them.
2. Fit different models.
3. See which one performs best wrt AUC.
4. Choose a winning model, and tune this model.
5. Then take that tuned winning model, and make final predictions.

Machine learning algorithms I used (models in step 2)

- logistic regression
- decision trees
- KNN
- neural networks

Language: Python.

## Important files

- `do2.py` was the file I submitted in the case. It tries reg, knn, nnet. It uses knn for final predictions.
- `do3.py` is a continuation of `do2.py`. It tries reg, trees, knn. It uses reg for final predicitons.
- `do4.py` is a continuation of `do3.py`. It introduces a threshold so that I can balance the tradeoff between type I and type II error.

## Folders

Here is what each folder is supposed to contain.

- `code` the important code. Not all code is stored here.
- `notes` useful code snippets, things I need to keep in mind, ideas on modeling.
- `dataset` the raw datasets being used.
- `html-output` is what other people should read. Usually it the jupyter notebook exported as HTML. It can also be slides or a report / executive summary.
- `output-predictions` the predicted probabilities/classes
- `output` all other output, such as tables, graphs, etc. 

## Status

This repo is a bit messy due to the creation of many files, because I worked here and there rarely dedicated long hours in a row.

I want to do more with this data. See `TODO.txt`
