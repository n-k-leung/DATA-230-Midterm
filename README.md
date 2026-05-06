# DATA-230-Midterm
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

ML Findings
* HGB model performs the best out of the three models used
* LDA is the worse performing model that showing that airline satisfaction is a non linear relationship
* Seat comfortability matters the most for the models to classify if the passenger is satisfied or not
* There is more importance on the quality of inflight services compared to delays or customer demographic in determining if a passenger is satisfied or not 


## Google doc version history collaboration
https://docs.google.com/document/d/1V8uieZSqqPLVf4e8f-LDnXZNKiSqfqdikTKCa9P-mwY/edit?usp=sharing 

## Tableau Dashboard
https://public.tableau.com/app/profile/angeli.faith.deanon/viz/mid_presentation_final_dash/Dashboard1?publish=yes

## Plotly Dashboard
https://drive.google.com/file/d/1mtJtXdJKyIqOu4LHay-Z2kO8m-mDGFNM/view?usp=sharing

## Streamlit Dashboard
Make sure the following requirements are installed before running the streamlit dashboard
```
pip install lime
pip install streamlit scikit-learn imbalanced-learn 
```
To run the streamlit dashboard
```
streamlit run streamlit/app.py 
```

