# CS342-Machine-Learning-Assignment-1

Classification and Prediction problems

Hi! This is my first machine learning project using Python with the package sckit-learn. Below is a brief description of the project. 
Please find detailed methodology and codes in the files "answer.pdf", "predictClass.py" and "predictTNA.py".
"classA.csv", "probeA.csv" and "probeB.csv" are the given data sets.

Context:

Two space probes have recently arrived at the planet Nereus to collect data on the existing extraterrestrial life-forms living 
under water. The first probe (probe A) collected data on 1000 life-forms measuring 4 chemical compounds for each lifeform
{cryptonine, mermaidine, posidine, neraidine} each at 3 different chemical resolutions plus a further genetic attribute called TNA. 
The second probe (probe B) unfortunately malfunctioned during data transmission but before doing so we received a further dataset 
on 1000 life-forms without the TNA measurements.

Task 1 "predictClass.py" is to give predictions (probabilities for class 1) for the probeB data.

Task 2 "predictTNA.py" is to give the genetic attribute (TNA) predictions for the probeB data.

The models tested are OLS, Lasso Regression, Ridge Regression, Decision Trees Classifier/Regressor, k-Nearest Neighbour 
Classifier/Regressor and Random Forest. Please note my codes only output the optimal model in terms of Area under ROC curve and 
Rsquared. Data sets are pre-processed and model parameters are tested by performing cross-validations. 
