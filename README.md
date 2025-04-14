# Abstract

Solar flares are among the most powerful and dynamic events in the solar system, resulting from the sudden release of magnetic energy stored in the Sun’s atmosphere. These energetic bursts of electromagnetic radiation can release up to 10^32 erg of energy, impacting space weather and posing risks to technological infrastructure and therefore require accurate forecasting of solar flare occurrences and intensities.

This study evaluates the predictive performance of three machine learning algorithms—Random Forest, k-Nearest Neighbors (KNN) and Extreme Gradient Boosting (XGBoost)—for classifying solar flares into 4 categories (B, C, M, X). Using the dataset created by Liu et al. (2016) with 13 SHARP parameters, we compared the models' effectiveness in both binary and multiclass classification tasks, utilizing 8 principal components (PC) capturing 95% of data variance and 100 PC capturing 97.5% of variance.

Our approach uniquely combines binary and multiclass classification with different levels of dimensionality reduction, an innovative methodology not previously explored in the context of solar flare prediction. Employing stratified 10-fold cross-validation and grid search for hyperparameter tuning ensured robust model evaluation.

Our findings indicate that Random Forest and XGBoost consistently demonstrate strong performance across all metrics, benefiting significantly from increased dimensionality. This study's insights enhance future research by optimizing dimensionality reduction techniques and informing model selection for astrophysical tasks. By integrating this newly acquired knowledge in future research, we can develop more accurate space weather forecasting systems and a deeper understanding of solar physics.
