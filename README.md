# DATA-230-Midterm
## Motivation
### Project Goal
* Identify key factors affecting passenger satisfaction and explore how service quality, customer loyalty, and operational reliability influence airline passenger experience

### Project Question
What attributes on the passenger/flight experience are impacting passenger satisfaction?

Which of the three chosen ml models, Hist Gradient Boosting, Multi-layer Perceptron, and Linear Discriminant Analysis, best classify airline passenger satisfaction?

## Dataset
Dataset: Airline Passenger Satisfaction (Kaggle) https://www.kaggle.com/datasets/johndddddd/customer-satisfaction/data 
* Contains real airline passenger survey responses

Dataset Size
* ~129,000 passenger records
* 24 variables including demographics, service ratings, and operational data
* Sufficiently complex and appropriate for exploratory data analysis

Key Features
* Passenger demographics (Age, Gender)
* Travel characteristics (Class, Type of Travel, Flight Distance)
* Service ratings (Seat comfort, WiFi, Food, Boarding, etc.)
* Operational metrics (Departure delay, Arrival delay)
* Target variable: Passenger Satisfaction

## EDA Findings
* Passenger satisfaction is strongly influenced by service quality, especially cleanliness, baggage handling, and inflight services
* Flight delays have a clear negative impact, with longer delays significantly reducing satisfaction across all age groups
* Passengers who are loyal and in business class have abnormally high satisfaction
* Passenger satisfaction is not strongly dependent on flight distance, suggesting operational and service factors are more important
* We found that passenger satisfaction is primarily driven by service experience rather than demographic factors or flight distance

## ML Findings
* HGB model performs the best out of the three models used
* LDA is the worse performing model that showing that airline satisfaction is a non linear relationship
* Seat comfortability matters the most for the models to classify if the passenger is satisfied or not
* Based on eda, we thought passenger class and delays have more weight but that was not the case
* There is more importance on the quality of inflight services compared to delays or customer demographic in determining if a passenger is satisfied or not 

## Future Work
* Compare with well known model performance to see if these well known models are more well known because they perform better
* Find a better way to deal with skewness of arrival and departure delay bc departure delay has a lot of zeros and log can’t normalize the spike

## Google doc version history collaboration
https://docs.google.com/document/d/1V8uieZSqqPLVf4e8f-LDnXZNKiSqfqdikTKCa9P-mwY/edit?usp=sharing 

https://docs.google.com/document/d/11w3HzEDLCD0tBJZzcifGrpvNRgk150tNNYOkjA4Pwuc/edit?usp=sharing 

## Tableau Dashboard
Our Tableau Dashboard shows passenger demographics & satisfaction. It explores how passenger characteristics influence overall satisfaction focusing on passenger age and gender. The link to the dashboard can be found on:

https://public.tableau.com/app/profile/angeli.faith.deanon/viz/mid_presentation_final_dash/Dashboard1?publish=yes

## Plotly Dashboard
Our Plotly dashboard shows flight experience & services. It analyzes how flight characteristics and services impact satisfaction focusing on passenger class and loyalty. A demo of this dashboard can be found via the following link:

https://drive.google.com/file/d/1mtJtXdJKyIqOu4LHay-Z2kO8m-mDGFNM/view?usp=sharing

To run Plotly, make sure the following requirements are installed:
```
pip install dash plotly
```
To run the Plotly dashboard:
```
python app.py
```

## Streamlit Dashboard
The Streamlit dashboard goes over the machine learning results of training the ml models to classify is a passenger is satisfied or not. The three models used are: Hist Gradient Boosting, Multi-layer Perceptron, and Linear Discriminant Analysis.

Make sure the following requirements are installed before running the streamlit dashboard
```
pip install lime
pip install streamlit scikit-learn imbalanced-learn 
```
To run the streamlit dashboard
```
streamlit run streamlit/app.py 
```

