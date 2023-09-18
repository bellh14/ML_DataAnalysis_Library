
# Maching Learning and Data Analaysis Library

The goal of this is to simplify the time needed to develop and test data cleaning and maching learning models. The classes act as a simplified abstraction layer to implement my most componly used methods from pandas, plotly, scikit-learn, and tensorflow to start testing and analyzing different models performance for any new dataset.

Having to load, clean, analyze, train, and then evaluate different models performance is exteremely time consuming if you do not already know the right imputation, encodings, and model to use (which you often do not of course).

# Need to Finish writing out the rest

I wrote most of this in the last day (17/09/2023), but it took longer to finish other projects and I ran out of time. Should be able to finish the rest by the 19th of September. It is mainly just calling already written methods.

## Implementation Overview

- Data is loaded into a Dataset object. This object takes in both traning and test data pathes, as well as other file/folder information for later storing the data.

- This Dataset object is then passed into the other classes.

- The idea is that the FullPipeLine class will be called with specified, imputation method, encoding scheme, visualizations, and model type which then runs through the whole pipeline, returning the results.

- Though the fullpipeline might be too simplifed, so the user still has full freedom to call any methods they want. Since every class takes in the Dataset object, this is extremly simple and fast to accomplish.
