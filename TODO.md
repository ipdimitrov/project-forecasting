# TODO: Future Improvements

## API Enhancements
- Add more testing not only on datapoint
- Use the API for training, testing, generating, etc., not only for getting the forecasts
- Make everything work out of docker

## Data and Model Improvements

### Data Expansion
- Test with more data, more sites
- Experiment with a model that combines the two sites, so learns from both of them

### Feature Engineering
- Experiment with other time features, as lag features add noise and make the forecast go low because end of december we have low consumption and the model uses those low datapoints to forecast more low datapoints. Therefore lag features perform very bad on iterative forecasting
- Think how to handle the very low saled in the End of the year that are mentioned in the previous point
- More experimentation with grid search not only by model parameters, but also by features used
- Weather data (currently we don't know the exact location of the sites)
- Maybe insert some macroeconomic data about the production
- Public holidays data (days that are non-working)

### Model Architecture
- I would experiment with a stacked model - one for forecasting daily values, and one for the distribution of the daily values on 1-hour intervals.
- Maybe try out also other ways of splitting the 1 hour data into 15-minute intervals, such as smoothing the data
- I would like to train the best performing model up until the end of the year, and then use it to forecast the next year. Now we train only up to end of November
- Make everything one click, so that we can just click on a button and we train models, generate forecasts, combine them for all sites
- When doing grid search we don't need to calculate baseline metrics for each model, we can just calculate it once.

### Alternative Model Types
- LSTM
- GRU
- TiRex (By Sepp Hochreiter)
- Mamba / SSMs
- Random Forest
- XGBoost
- Catboost
- LightGBM
- XGBoost + NN Embeddings