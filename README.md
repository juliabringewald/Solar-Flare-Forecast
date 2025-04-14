# Solar Flare Forecast - A Comparative Analysis of Machine Learning Algorithms for Predicting Solar Flare Classes 

Solar flares are among the most powerful and dynamic events in the solar system, resulting from the sudden release of magnetic energy stored in the Sun’s atmosphere. These energetic bursts of electromagnetic radiation can release up to 10^32 erg of energy, impacting space weather and posing risks to technological infrastructure and therefore require accurate forecasting of solar flare occurrences and intensities.

This study evaluates the predictive performance of three machine learning algorithms—Random Forest, k-Nearest Neighbors (KNN) and Extreme Gradient Boosting (XGBoost)—for classifying solar flares into 4 categories (B, C, M, X). Using the dataset created by Liu et al. (2016) with 13 SHARP parameters, we compared the models' effectiveness in both binary and multiclass classification tasks, utilizing 8 principal components (PC) capturing 95% of data variance and 100 PC capturing 97.5% of variance.

Our approach uniquely combines binary and multiclass classification with different levels of dimensionality reduction, an innovative methodology not previously explored in the context of solar flare prediction. Employing stratified 10-fold cross-validation and grid search for hyperparameter tuning ensured robust model evaluation.

Our findings indicate that Random Forest and XGBoost consistently demonstrate strong performance across all metrics, benefiting significantly from increased dimensionality. This study's insights enhance future research by optimizing dimensionality reduction techniques and informing model selection for astrophysical tasks. By integrating this newly acquired knowledge in future research, we can develop more accurate space weather forecasting systems and a deeper understanding of solar physics.


# Dataset

For this study, we utilized a pre-processed dataset of solar flares based on Space-weather HMI Active Region Patches (SHARP) data, originally created by Liu et al. (2016), publicly available at https://iopscience.iop.org/article/10.3847/1538-4357/aa789b/meta#apjaa789bapp1. The SHARP dataset, developed by the SDO/HMI team (Bobra et al., 2014), provides magnetic field measurements and derived parameters for automatically identified and tracked active regions (ARs). The data covers a ~6.5-year period (May 2010–December 2016), covering the peak of Solar Cycle 24. The dataset includes 13 key magnetic field 7 parameters, identified by Bobra & Couvidat (2015) as strong predictors of solar flare activity.

| **PARAMETER** | **DESCRIPTION** |
|---------------|-----------------|
| ABSNJZH       | Absolute value of net current helicity |
| AREA_ACR      | Area of strong field pixels in active region (AR) |
| EPSZ          | Sum of z-component of normalized Lorentz force |
| MEANPOT       | Mean photospheric magnetic free energy |
| R_VALUE       | Sum of flux near polarity inversion line |
| SAVNCPP       | Sum of the modulus of the net current per polarity |
| SHRGT45       | Fraction of area with shear \( >45^\circ \) |
| TOTBSQ        | Total magnitude of Lorentz force |
| TOTFZ         | Sum of z-component of Lorentz force |
| TOTFOT        | Total photospheric magnetic free energy density |
| TOTUSJH       | Total unsigned current helicity |
| TOTUSJZ       | Total unsigned vertical current |
| USFLUX        | Total unsigned flux |
