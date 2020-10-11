# Police-Killings-Research
Doing Research to Push for the Greater Good


# Introduction
The people of the United States have becoming increasingly more aware of the country’s police force being both undertrained and underprepared to handle the immense amounts of stress that go along with the job, as well as the implicit (and sometimes explicit) biases that rise to the surface when police are put in high-intensity situations. A combination of these two factors, as well as a multitude of other internal and external forces, has led to a very large number of situations where civilians have been and are continuing to be killed by police who are using excessive force. 

There is an existing divide between people who over-appreciate law enforcement and those who detest it, and the already-polarizing topic has been used by politicians as a pawn in gathering votes, all while the problem at hand continues to grow. Serious measures need to be taken in providing law enforcement with the resources that its departments need, while still punishing its members that abused their power within society. 

My goal is to help expose some more of the lack of accountability for the police, as well as maybe dive more into what external factors can lead towards more or less accountability within these police departments. The model should identify cases in which police are likely to have their body cameras turned off with the available data on police killings. Research like this could and hopefully will be very helpful in the aiding the movement towards a more peaceful and transparent relationship between law enforcement and civilians. 


# Data Description 
(After cleaning and merging)
6199 Observations of 25 Variables
-	Police Killing Data from 2013-2020 = MappingPoliceViolence.org = FatalEncounters.org, the U.S. Police Shootings Database, and KilledbyPolice.net
-	Census Data (2015) = MedianHouseHoldIncome2015, PercentOver25CompletedHighSchool, PercentagePeopleBelowPovertyLevel, ShareRaceByCity
VARIABLES:
"Victim_Age" - Age of Victim   (Categorical)                           
"Victim_Gender" - Gender of Victim   (Categorical)       
"Victim_Race" - Race of Victim   (Categorical)            
"City" - City of Death    (Categorical)                 
"State" - State of Death   (Categorical)                 
"Zipcode" - Zipcode of Death   (Categorical)              
"County" - County of Death    (Categorical)              
"PD_Responsible" - Agency Responsible for Death    (Categorical)      
"Cause_of_Death" - Reported Cause of Death    (Categorical)       
"Official_Disposition_of_Death" - Official Disposition of Death   (Categorical)
"Criminal_Charges" - Criminal Charges (if any)   (Categorical)    
"Symptoms_of_Mental_Illness" - Any symptoms of mental illness   (Categorical)
"Armed_or_Unarmed" - Whether or not victim was reported as armed    (Categorical)  
"Alleged_Weapon" - Alleged weapon victim was reported as using   (Categorical)     
"Alleged_Threat_Level" - Alleged threat level officer reported feeling   (Categorical)
"Fleeing_or_Not" - Whether or not victim was reported as fleeing   (Categorical) 
"BodyCamBinary" - RESPONSE VARIABLE  Body Cam On/Off  (Categorical, Binary) 
(5739 Off, 460 On)  
"Median_Income" - Median Income of City of Death    (Numeric)     
"Percent_Completed_HS" - Percent Completed High School in City of Death    (Numeric)  
"Poverty_Rate" - Poverty Rate in City of Death    (Numeric)     
"Share_White" - White Percent of Population in City of Death    (Numeric)       
"Share_Black" - Black Percent of Population in City of Death    (Numeric)        
"Share_Native_American" - Native American Percent of Population in City of Death   (Numeric)
"Share_Asian" - Asian Percent of Population in City of Death    (Numeric)         
"Share_Hispanic" - Hispanic Percent of Population in City of Death   (Numeric)


The pre-processing data required the merging of multiple datasets into one master dataset. First, the dataset from Mapping Police Violence was loaded into R. Then, “NA.” values were removed and a “BodyCamBinary” variable was created as a binary duplicate of the variable containing body camera data. Lastly, that variable was converted to a factor for plotting purposes before adding in the census datasets to the project.

Variable names in the 4 census datasets were changed before merging them into one larger census dataset. The strings within the larger census dataset were cleaned up a bit more, and then a fuzzy match was performed using a ‘for’ loop in order to match City and State names from the two datasets together for the final merge of the preprocessing steps. Afterwards, some exploratory visualizations were done on the data. Please see Appendix for data visualizations. 

After the visualizations, a random number generator seed was set and then the data was split using the stratified() function, dividing the data on the “BodyCamBinary” variable and ensuring that the proportions of 1/0 values in both the training and test datasets were the same. The training dataset was made with 20% of the data and the test dataset was made with 80% of the data. 

# Methods

A binary classification model was created for this project using the CatBoost package installed from GitHub. It is very similar to the XGBoost package, but instead it is optimized for handling categorical data that has large amount of unique values. It seemed to be the best fit for this project as the “City” variable, alone, had over 1000 unique values in it. CatBoost takes all of the categories within the variables and converts them to numbers automatically without any extra steps in data formatting and cleaning before running the model. Another strength in this model is its prediction time. A major weakness in the model is the time it takes to optimize the model (i.e. cross-validation). 

# Results

Rather than use the Area Under the Curve (AUC) to measure model success in training, the CatBoost model was set to measure Logarithmic Loss (LogLoss). LogLoss is actually seen as more important or explanatory than AUC in many classification problems. The LogLoss value decreases as the model is more confident on incorrect predictions. This, along with some other characteristics of the metric, make it more relevant in problems such as this one (where there is an imbalanced dataset in regard to the response variable). The model also instantly calculated the feature importance for each variable. 

When used on the training dataset, the accuracy of the model was 0.6911, with the sensitivity being 0.71739. Find the confusion matrix below:

         0   1
     
     0 791  26
  
     1 357  66

So, the model performed alright on the test data. The sensitivity, or the True Positive Rate, was higher than the overall accuracy of the model. That is important, as it was important that the model found as many correct cases in which a body camera would be on as possible. It was worse than initially expected, but this makes sense as predicting something that involves a lot of human interaction can be pretty difficult in some scenarios, especially in high-pressure situations that all end in someone getting killed. The most important takeaway from this model was the feature importance, and ultimately the SHAP values that were extracted from the results.  

The opportunity to optimize and tune parameters through cross-validation may have allowed for a more accurate or sensitive model. Otherwise, there were not any serious encountered issues in the project or model fitting process. CatBoost is a very fast and powerful package. 

# Discussion

After the model was run, there were a few insight functions that were run in order to be able to extract some more information. SHAP values were calculated for the variables in the dataset in order to find variable importance. Through some more calculations, a few insight graphs were created in order to better understand the model and its results. As stated earlier, one of the most important points of this project was to find which factors (variables) were significant in determining the outcome of one of the scenarios at hand. 
The model found that the “Share_White”, “Share_Hispanic”, “Median_Income”, and “Poverty_Rate” variables were the most “important” to the model outcome, based on their feature importance values. That being said, they were harder to interpret once plotted based by their actual feature values against their SHAP values, some more insights were possible. When “Median_Income” had a high feature value, it was found to negatively impact the model outcome (higher probability of the body camera being turned off). The same goes for the “Share_Black” variable. When the “Percent_Completed_HS” variable had a high feature value, it was found to positively impact the model outcome (higher probability of the body camera being turned off). This was interesting to me, as the researcher, as it helped match up some of my insights from the pre-processing visualizations. But, one of my most interesting graphs from the pre-processing stage that mapped out the densities of body cameras being turned on or off against the “Share_White” variable using a facet_wrap function to visualize the relationship by the race of the victim. I performed a very similar visualization using the SHAP values of the “Share_White” variable, and found that as the “Share_White” value increases, the SHAP values tend to become negative around 68% white. This means that as a city’s white population accounts for roughly 68% or more, the police are more likely to have their body cameras off during an interaction that ends in a killing. 
Action that can be taken after realizing these insights includes becoming more educated about social issues affecting local, national, and global communities, learning about where taxpayer money goes, as well as the money spent at private companies, and registering to vote (when and where one is able to). Another relevant move could be demanding action on ending police violence from a local representative, senator, or governor. 

# Conclusion and Future Work

Police violence is a prevalent issue in today’s society, particularly in the United States. This comes from a variety of factors, some of which are able to be tracked, others are not. One aspect of the police violence is a lack of accountability from police officers and law enforcement when it comes to taking responsibility for their actions that result in excessive force and the killing of the country’s citizens. Throughout this project, I performed multiple data exploration techniques and visualizations, and then created a predictive model that would predict whether or not a police officer’s body camera was on or off during a killing. The model was 69% accurate, and the most important variable in determining the response was the percent white portion of a city’s population, as the probability of the body camera being off would dramatically increase as the white percentage increased. 

There are always more ways to explore the data and find connections, and this project is not completed. I will continue exploring the data and trying to find more meaningful insights to pull from it. 
