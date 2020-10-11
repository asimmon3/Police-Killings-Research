##' Ariel Simmons #################################################
##' 
##' Machine Learning Fall 2020
##' Professor Martin Barron
##' 
##' Making Meaningful Predictions on Police Killings & Census Data 
##' Data all found on *www.kaggle.com* and *www.mappingpoliceviolence.com*
##' 

#################### DATA EXPLORATION ###############################


#'install.packages("xlsx")
library(xlsx)
library(ggplot2)
library(stringr)
library(dplyr)
library(tidyverse)
library(ggmap)

#' Read in the MappingPoliceViolence Dataset that we found
PoliceViolence <- read.xlsx("MPVDatasetDownload.xlsx", 
                            sheetName = "2013-2020 Police Killings")
#' See how many off duty killings there were to decide if it is worth it to keep
#' that variable in the data
summary(as.factor(PoliceViolence$Off.Duty.Killing.))

PoliceViolence2 <- PoliceViolence #' Create a second DF just in case

PoliceViolence2 <- select(PoliceViolence2, -c(starts_with("NA."))) #' remove NAs

summary(as.factor(PoliceViolence$Body.Camera..Source..WaPo.))                        
#' See how many killings had body camera footage

BodyCamBinary <- rep(0, nrow(PoliceViolence2)) #'Create Response Variable vector
BodyCamBinary #' View response variable vector (filled with zero values now)
BodyCamBinary[which(PoliceViolence2$Body.Camera..Source..WaPo. == "Yes")] <- 1 
#' Take all of the values in which the Body Cameras were turned on and assign 
#' them a '1' value in the response variable vector

PoliceViolence2$BodyCamBinary <- as.factor(BodyCamBinary) #' Convert to factor
summary(PoliceViolence2$BodyCamBinary) #' Summarize response variable

#' Read in the Census data in order to merge datasets
MedianHouseholdIncome2015 <- (read.csv("MedianHouseholdIncome2015.csv"))
PercentOver25CompletedHighSchool <- 
  (read.csv("PercentOver25CompletedHighSchool.csv"))
PercentagePeopleBelowPovertyLevel <- 
  (read.csv("PercentagePeopleBelowPovertyLevel.csv"))
ShareRaceByCity <- (read.csv("ShareRaceByCity.csv"))

#' Change column names in order to make merging more seamless
colnames(ShareRaceByCity)[colnames(ShareRaceByCity) == "Geographic.area"] <- 
  "State"
colnames(PercentOver25CompletedHighSchool)[colnames(PercentOver25CompletedHighSchool) 
                                           == "Geographic.Area"] <- "State"
colnames(PercentagePeopleBelowPovertyLevel)[colnames(PercentagePeopleBelowPovertyLevel) 
                                            == "Geographic.Area"] <- "State"
colnames(MedianHouseholdIncome2015)[colnames(MedianHouseholdIncome2015) == 
                                      "Geographic.Area"] <- "State"

#' Join the all 4 census datasets together
BigDataFrame <- inner_join(MedianHouseholdIncome2015, 
                           PercentOver25CompletedHighSchool, 
                           by = c("City","State"))
BigDataFrame <- inner_join(BigDataFrame, PercentagePeopleBelowPovertyLevel, 
                           by = c("City","State"))
BigDataFrame <- inner_join(BigDataFrame, ShareRaceByCity, 
                           by = c("City","State"))

#' Clean up merged dataframe before fuzzy matching to MPV Dataset
BigDataFrame['City']
str_which(BigDataFrame$City, 'CDP')
str_which(BigDataFrame$City, 'Township')
str_which(PoliceKillingsUS$City, 'City')
BigDataFrame$City <- str_replace(BigDataFrame$City, ' CDP', '')
BigDataFrame$City <- str_replace(BigDataFrame$City, 'city', 'City')
BigDataFrame$City <- str_replace(BigDataFrame$City, 'town', 'Town')
BigDataFrame$City <- str_replace(BigDataFrame$City, 'City City', 'City')
BigDataFrame$City <- str_replace(BigDataFrame$City, 'village', 'Village')

#'Just removed that fake wannabe city off of my glorious dataset
BigDataIndex <- rep(NA, nrow(PoliceViolence2)) 
#'Start with a vector full of NA Values
for (i in 1:nrow(PoliceViolence2)) {  #Loop through the cities
  for (j in c(0, 0.0001, 0.001, 0.01, 0.1)) { 
    #'Loop through different 'max distance' values
    if (is.na(BigDataIndex[i])) { 
      #'Checks if the BigDataIndex[i] vector is still NA
      temp <- agrep(paste(PoliceViolence2$City[i],
                          PoliceViolence2$State[i]), 
                    paste(BigDataFrame$City,BigDataFrame$State), 
                    ignore.case = TRUE, value = FALSE, max.distance = j) 
      #'Tries to fuzzy match at max distance 'j'
      print(temp) #'Print the indices we have
      if (length(temp) != 0) { 
        #'If fuzzy match is found, assign to the next corresponding row on 
        #'BigDataIndex
        BigDataIndex[i] <- temp[1] 
        #'Assigns the printed indices to the result vector
      }
    }
  }
}
BigDataIndex

ResData <- as.data.frame(matrix(NA, nrow = nrow(PoliceViolence2), ncol = 10))
for (i in 1:nrow(PoliceViolence2)) { #
  if (!is.na(BigDataIndex[i])) { #'Loop through each row with an index 
    #'found from previous loop
    ResData[i,] <- BigDataFrame[BigDataIndex[i],] #Assign indexed rows to new DF
  }
}
ResData #' The resulting, fuzzy-matched, 
#' dataframe to be combined with PoliceKillingsUS

names(ResData)[1:2] <- c("MatchedState", "MatchedCity")

THE_FINAL_DATA_FRAME <- cbind.data.frame(PoliceViolence2, ResData) 
#'Create a final *clean* DataFrame

names(THE_FINAL_DATA_FRAME)[33:40] <- names(BigDataFrame)[3:10]
#'Rename the V1-10 columns to their field names

save(THE_FINAL_DATA_FRAME, file = "PoliceKillingsPlusCensusData.rda") 

#####################'VISUALIZATION TIME ###################################

load("PoliceKillingsPlusCensusData.rda")

#' Change all instances where 'Body Camera' is listed as 'NA' because we will
#' treat all 'NA' values as 'No' values. This is because we are assuming that 
#' if the body camera was on, the police department would have either shared the 
#' footage, or is not sharing the footage because it is too incriminating. 
#' Therefore, if the 'Body Camera' value is not a 'Yes', then it will be treated
#' as a 'No'.
THE_FINAL_DATA_FRAME$Body.Camera..Source..WaPo.[is.na(THE_FINAL_DATA_FRAME$Body.Camera..Source..WaPo.)] <- "No"

plot_data <- 
  THE_FINAL_DATA_FRAME[is.na(THE_FINAL_DATA_FRAME$Off.Duty.Killing.),] 
#' Create duplicate DF for visualizations without the Off Duty killings

sum(is.na(plot_data$MatchedCity)) #' See how many NA values there are in the
#' ShootingsCity variable

colSums(is.na(plot_data)) #' See where NA values pop up
colnames(plot_data) #' Check column names

summary(as.factor(plot_data$Alleged.Threat.Level..Source..WaPo.)) 
#' Summarize alleged threat level variable
plot_data$Alleged.Threat.Level..Source..WaPo.[is.na(plot_data$Alleged.Threat.Level..Source..WaPo.)] <- 
  "undetermined" #' Change 'NA' values to undetermined for better understanding
table(as.factor(plot_data$Fleeing..Source..WaPo.), plot_data$BodyCamBinary,
      useNA = "always") 
plot_data$Fleeing..Source..WaPo.[is.na(plot_data$Fleeing..Source..WaPo.)] <-
  "Unknown" #' Change 'NA' values to 'Unknown'

head(plot_data, 10)
plot_data <- plot_data[c(2:4, 8:12, 14, 16:17, 19:23, 30:40)] 
#' select important/needed variables from dataset
plot_data <- na.omit(plot_data) #' Remove the NA values from the dataframe

table(plot_data$race, plot_data$armed) 
#' plot the Armed/weapon and race variables

#' Convert explanatory variables to numeric/factor and response variable to 
#' factor for plotting
names(plot_data)
str(plot_data)
summary(as.factor(plot_data$Victim.s.age))
plot_data$Victim.s.age[plot_data$Victim.s.age == "40s"] <- "45"
plot_data$Victim.s.age <- as.numeric(plot_data$Victim.s.age)
plot_data$Victim.s.gender <- as.factor(plot_data$Victim.s.gender)
plot_data$Victim.s.race <- as.factor(plot_data$Victim.s.race)
plot_data$City <- as.factor(plot_data$City)
plot_data$State <- as.factor(plot_data$State)
plot_data$Zipcode <- as.factor(plot_data$Zipcode)
plot_data$County <- as.factor(plot_data$County)
plot_data$Agency.responsible.for.death <- 
  as.factor(plot_data$Agency.responsible.for.death)
plot_data$Cause.of.death <- as.factor(plot_data$Cause.of.death)
plot_data$Official.disposition.of.death..justified.or.other. <-
  as.factor(plot_data$Official.disposition.of.death..justified.or.other.)
plot_data$Criminal.Charges. <- as.factor(plot_data$Criminal.Charges.)
plot_data$Symptoms.of.mental.illness. <- 
  as.factor(plot_data$Symptoms.of.mental.illness.)
plot_data$Unarmed.Did.Not.Have.an.Actual.Weapon <-
  as.factor(plot_data$Unarmed.Did.Not.Have.an.Actual.Weapon)
plot_data$Alleged.Weapon..Source..WaPo.and.Review.of.Cases.Not.Included.in.WaPo.Database. <-
  as.factor(plot_data$Alleged.Weapon..Source..WaPo.and.Review.of.Cases.Not.Included.in.WaPo.Database.)
plot_data$Alleged.Threat.Level..Source..WaPo. <- 
  as.factor(plot_data$Alleged.Threat.Level..Source..WaPo.)
plot_data$Fleeing..Source..WaPo. <- as.factor(plot_data$Fleeing..Source..WaPo.)
plot_data$BodyCamBinary <- as.factor(plot_data$BodyCamBinary)
plot_data$poverty_rate <- as.numeric(plot_data$poverty_rate)
plot_data$Median.Income <- as.numeric(plot_data$Median.Income)
plot_data$share_white <- as.numeric(plot_data$share_white)
plot_data$share_black <- as.numeric(plot_data$share_black)
plot_data$share_hispanic <- as.numeric(plot_data$share_hispanic)
plot_data$share_native_american <- as.numeric(plot_data$share_native_american)
plot_data$share_asian <- as.numeric(plot_data$share_asian)
plot_data$percent_completed_hs <- as.numeric(plot_data$percent_completed_hs)


colSums(is.na(plot_data))
colnames(plot_data)
plot_data <- plot_data[,-c(18:19)]
plot_data <- na.omit(plot_data)

names(plot_data) <- c("Victim_Age","Victim_Gender","Victim_Race","City","State",
                      "Zipcode","County","PD_Responsible","Cause_of_Death",
                      "Official_Disposition_of_Death","Criminal_Charges",
                      "Symptoms_of_Mental_Illness","Armed_or_Unarmed",
                      "Alleged_Weapon","Alleged_Threat_Level","Fleeing_or_Not",
                      "BodyCamBinary","Median_Income","Percent_Completed_HS",
                      "Poverty_Rate","Share_White","Share_Black",
                      "Share_Native_American","Share_Asian","Share_Hispanic")

CitiesWherePoliceShootPpl <- as.data.frame(table(plot_data$City))
names(CitiesWherePoliceShootPpl) <- c("City","freq")
CitiesWherePoliceShootPpl$freq <- as.numeric(CitiesWherePoliceShootPpl$freq)
CitiesWherePoliceShootPpl

MoreThan10DeathsByOfficerIndex <- 
  CitiesWherePoliceShootPpl[CitiesWherePoliceShootPpl$freq >= 40,]
MoreThan10DeathsByOfficerDF <- plot_data[plot_data$City %in%
                                           MoreThan10DeathsByOfficerIndex$City,]

#' First Visualization
#' Mapping the density plot for the poverty rate vs body camera on

PovRate_BodyCam_Viz <- ggplot(plot_data, aes(x = log(Poverty_Rate) +1, 
                                        fill = BodyCamBinary)) +
  # Set x as poverty_rate and fill as class
  geom_density(alpha = 0.5) + # Select density plot and set transperancy (alpha)
  theme_set(theme_bw(base_size = 22) ) + # Set theme and text size
  theme(title = element_text(size = 15),
        panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) + # Remove grid 
  labs(x = "Log(Poverty Rate) + 1", 
       title = "Poverty Rate vs Body Camera Footage",
       fill = "Camera On") # Set labels
PovRate_BodyCam_Viz

#' From the looks of it, poverty rate is not a good predictor of body camera
#' footage

#' Second Visualization
#' Now let's see the bar graph for the race of the victim vs body camera on

RaceVsBodyCamOn <- ggplot(plot_data, aes(x = Victim_Race,fill = BodyCamBinary))+
  geom_bar(alpha = 0.5) +
  #scale_x_discrete(labels = c("Asian","Black","Hispanic","Native American",
                              #"Other","White")) + #' Change names
  theme_set(theme_bw(base_size = 18)) +
  theme(axis.text.x = element_text(size = 12, angle=45, hjust=1),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) +
  labs(x = "Race", y = "Number Deaths by Police",
       title = "Race vs Body Camera Footage",
       fill = "Camera On")
RaceVsBodyCamOn

#' it looks as though the body camera footage that we have seen online is a rare
#' occurrence, as most police killings do not include it.... Seems odd to me

#' Third Visualization
#' Now let's see if the share_white variable will be a good predictor

PctWhite_BodyCam_Viz <- ggplot(plot_data, aes(x = Share_White,fill = BodyCamBinary))+
  # Set x as share_white and fill as class
  geom_density(alpha = 0.5) + # Select density plot and set transperancy (alpha)
  theme_set(theme_bw(base_size = 22) ) + # Set theme and text size
  theme(title = element_text(size = 12),
        axis.title.x = element_text(size = 15),
        panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) + # Remove grid 
  labs(x = "Percent of Population that is White", 
       title = "Percent White vs Body Camera Footage",
       fill = "Camera On") # Set labels
PctWhite_BodyCam_Viz

#' alas, it looks as though share_white is not a good predictor :-(
#' BUT WAIT! there may be a small coincidence.... There is a large spike in 
#' Body camera being off where the percentage of population being white is the
#' highest.... we should see where some of those killings line up with the race
#' of the victims..... to be continued


#' Fourth Visualization
PctBlack_BodyCam_Viz <- ggplot(plot_data, aes(x = log(Share_Black) + 1, 
                                         fill = BodyCamBinary)) +
  geom_density(alpha = 0.5) + # Select density plot and set transperancy (alpha)
  theme_set(theme_bw(base_size = 22) ) + # Set theme and text size
  theme(title = element_text(size = 15),
        panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) + # Remove grid 
  labs(x = "log(Percent Black) + 1", 
       title = "Percent Black vs Body Camera Footage",
       fill = "Camera On") # Set labels

PctBlack_BodyCam_Viz
#' some areas where it doesn't overlap.... this may be a better predictor..
#' only time will find out.

#' Fifth Visualization
#' Now, let's investigate our results of the PctWhitePredViz visualization...
#' We saw that the body cameras were more turned off more often in areas that 
#' were predominantly white... We will do the same visualization, but use the 
#' facet_wrap function to split the graphs by Race

PctWhiteByVicRace_BodyCam_Viz <- ggplot(plot_data, aes(x = Share_White, 
                                                  fill = BodyCamBinary)) +
  # Set x as share_white and fill as class
  geom_density(alpha = 0.5) + # Select density plot and set transperancy (alpha)
  facet_wrap(vars(Victim_Race)) + #' wrap by race
  theme_set(theme_bw(base_size = 15) ) + # Set theme and text size
  theme(title = element_text(size = 10),
        strip.text.x = element_text(size=8),
        panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) + # Remove grid 
  labs(x = "Percent White", 
       title = "Percent White vs Body Cam Footage by Race of Victim",
       fill = "Camera On") # Set labels
PctWhiteByVicRace_BodyCam_Viz

#' HMMMM INTERESTING!!!!!! It seems as though my earlier *prediction* was 
#' correct! It seems as though, when grouped by race, non-white victims 
#' (specifically NA, Black, Hispanic, and Native American victims) had different
#' peaks in density of body cameras being turned off during killings when the 
#' city had a higher percentage of white people.... very intriguing...
#' Correlation cannot imply causation, but it is definitely worth investigating 
#' further.

#' Sixth Visualization
PctHispanic_BodyCam_Viz <- ggplot(plot_data, aes(x = log(Share_Hispanic) + 1,
                                            fill = BodyCamBinary)) +
  # Set x as share_hispanic and fill as class
  geom_density(alpha = 0.5) + # Select density plot and set transperancy (alpha)
  theme_set(theme_bw(base_size = 18) ) + # Set theme and text size
  theme(title = element_text(size = 12),
        panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) + # Remove grid 
  labs(x = "Log(Percent Hispanic) + 1", 
       title = "Percent Hispanic vs Body Camera Footage",
       fill = "Camera On") # Set labels
PctHispanic_BodyCam_Viz

#' This visualization does not necessarily show that the hispanic percentage
#' of city population is a good predictor for whether or not the body cameras
#' are turned on.

#' Seventh Visualization
#' Let's look at the city vs body camera footage graph in the form 
#' of a bar graph

CityVsBodyCam_Viz <- ggplot(MoreThan10DeathsByOfficerDF, aes(BodyCamBinary, 
                                                             fill = 
                                                               BodyCamBinary)) +
  geom_bar(alpha = 0.5) +
  facet_wrap(~ City) + #' wrap so that there is one graph per city
  theme_set(theme_bw(base_size = 18)) +
  theme(strip.text.x = element_text(size=8),
        axis.text.x = element_text(size = 12, angle=45, hjust=1),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) +
  labs(x = "Cities", y = "Number Deaths by Police",
       title = "City vs Body Camera Footage",
       fill = "Camera On")
CityVsBodyCam_Viz

#' The bar graph is interesting to look at, as we see that many more cases occur
#' where the police have their body cameras turned off during killings 
#' (specifically in cities with the most deaths by police), which is a very 
#' interesting, yet scary, observation, as the body cams would provide
#' a more objective view and accurate representation of the events that 
#' transpired. 


####' *MOST INTERESTING VISUALIZATIONS* 
####' - CityVsCamOnPredViz (tells us about lack of accountability for officers)
####' - PctwhitePredViz (because it led into the next one on the list)
####' - PctWhiteByVicRacePredViz (tells us about lack of accountability in
####'                        and potential biases in predominantly white areas..
####'                        AKA maybe race and share_white are going to make a
####'                        strong interaction term? Only time will tell...)
####' - RaceVsBodyCamOn (tells us about lack of accountability for officers)


###################### PREDICTION TIME ###################################
#' Install and load catBoost
#' install.packages('devtools')
#' BINARY_URL="https://github.com/catboost/catboost/releases/download/v0.24.1/catboost-R-Darwin-0.24.1.tgz"
#' devtools::install_url(BINARY_URL,args = c("--no-multiarch"))
#' install.packages("splitstackshape")
library(xgboost)
library(OptimalCutpoints)
library(pROC)
library(splitstackshape)
library(catboost)
library(caret)
library(data.table)
library(tibble)

colnames(plot_data)
summary(plot_data$Fleeing_or_Not)

# plot_data <- plot_data[,c(4:13,17:25)] #' extract needed data

# plot_data$manner_of_death <- as.factor(plot_data$manner_of_death)

set.seed(69696969) # Set random number generator seed for reproducability
#data <- stratified(indt=plot_data, group = c("manner_of_death", "gender",
#                                             "race", "Geographic.Area",
#                                             "signs_of_mental_illness",
#                                             "threat_level", "flee", 
#                                             "class"),
#                   size = 0.2, bothSets = TRUE) #' keeps both sets
#' split data so that the proportion of 1's and 0's in test and train datii
data <- stratified(indt=plot_data, group = c(
  "BodyCamBinary"),
  size = 0.2, bothSets = TRUE) #' keeps both sets
# Extract training and test dat
train_data <- data[[2]]
test_data <- data[[1]]
colnames(train_data)
train_use <- train_data[,c(1:16,18:25)]
test_use <- test_data[,c(1:16,18:25)]

summary(train_use)

train_data$BodyCamBinary <- as.numeric(train_data$BodyCamBinary) - 1

colnames(train_use)
# Create training data
train_pool <- catboost.load_pool(data = train_use, # Dataset
                                 label = as.double(train_data$BodyCamBinary), 
                                 # Label
                                 cat_features = c("Victim_Gender","Victim_Race",
                                                  "City", "State","Zipcode",
                                                  "County","PD_Responsible",
                                                  "Cause_of_Death", 
                                                "Official_Disposition_of_Death",
                                                "Criminal_Charges",
                                                "Symptoms_of_Mental_Illness",
                                                "Armed_or_Unarmed",
                                                "Alleged_Weapon",
                                                "Alleged_Threat_Level",
                                                "Fleeing_or_Not"))
                                                # Name categorical features

test_pool <- catboost.load_pool(data = test_use, # Dataset
                                cat_features = c("Victim_Gender","Victim_Race",
                                                 "City", "State","Zipcode",
                                                 "County","PD_Responsible",
                                                 "Cause_of_Death", 
                                                 "Official_Disposition_of_Death",
                                                 "Criminal_Charges",
                                                 "Symptoms_of_Mental_Illness",
                                                 "Armed_or_Unarmed",
                                                 "Alleged_Weapon",
                                                 "Alleged_Threat_Level",
                                                 "Fleeing_or_Not"))
                                                # Name categorical features

#' Set the seed
set.seed(69696969)
# Fit model
model <- catboost.train(train_pool,  test_pool = test_pool,
                        params = list(loss_function = 'Logloss',
                                      iterations = 1000)
)
model
#' Create predictions for the model
prediction <- catboost.predict(model, test_pool, 
                               prediction_type = "Probability")
print(prediction)
summary(prediction)

#' Convert predictions to classes using a cutoff value
pred_class <- rep(0, length(prediction))
pred_class[prediction >= 0.049] <- 1 


u <- union(pred_class,  test_data$BodyCamBinary) # Join factor levels
t <- table(factor(pred_class, u), 
           factor(test_data$BodyCamBinary, u)) # Create table
confusionMatrix(t, positive = "1") # Produce confusion matrix

# Check accuracy
table(prediction, test_data$BodyCamBinary)



################################################
########## Actionable Helper Insights ##########
################################################
#
#
#
#
#

# Note: The functions shap.score.rank, shap_long_hd and plot.shap.summary were 
# originally published at:
# https://liuyanguu.github.io/post/2018/10/14/shap-visualization-for-xgboost/
# All the credits to the author.

shap.score.rank_modified <- function(shap_matrix){
  require(data.table)
  shap_contrib <- shap_matrix
  shap_contrib <- as.data.table(shap_contrib)
  shap_contrib[,BIAS:=NULL]
  cat('make SHAP score by decreasing order\n\n')
  mean_shap_score <- 
    colMeans(abs(shap_contrib))[order(colMeans(abs(shap_contrib)), 
                                      decreasing = T)]
  return(list(shap_score = shap_contrib,
              mean_shap_score = (mean_shap_score)))
}

## functions for plot
# return matrix of shap score and mean ranked score list
shap.score.rank <- function(xgb_model = xgb_mod, shap_approx = TRUE, 
                            X_train = mydata$train_mm){
  require(xgboost)
  require(data.table)
  shap_contrib <- predict(xgb_model, X_train,
                          predcontrib = TRUE, approxcontrib = shap_approx)
  shap_contrib <- as.data.table(shap_contrib)
  shap_contrib[,BIAS:=NULL]
  cat('make SHAP score by decreasing order\n\n')
  mean_shap_score <- 
    colMeans(abs(shap_contrib))[order(colMeans(abs(shap_contrib)), 
                                      decreasing = T)]
  return(list(shap_score = shap_contrib,
              mean_shap_score = (mean_shap_score)))
}

# a function to standardize feature values into same range
std1 <- function(x){
  return ((x - min(x, na.rm = T))/(max(x, na.rm = T) - min(x, na.rm = T)))
}


# prep shap data
shap.prep <- function(shap  = shap_result, X_train = mydata$train_mm, top_n){
  require(ggforce)
  # descending order
  if (missing(top_n)) top_n <- dim(X_train)[2] # by default, use all features
  if (!top_n%in%c(1:dim(X_train)[2])) stop('supply correct top_n')
  require(data.table)
  shap_score_sub <- as.data.table(shap$shap_score)
  shap_score_sub <- shap_score_sub[, names(shap$mean_shap_score)[1:top_n], 
                                   with = F]
  shap_score_long <- melt.data.table(shap_score_sub, 
                                     measure.vars = colnames(shap_score_sub))
  
  # feature values: the values in the original dataset
  fv_sub <- as.data.table(X_train)[, names(shap$mean_shap_score)[1:top_n], 
                                   with = F]
  # standardize feature values
  fv_sub_long <- melt.data.table(fv_sub, measure.vars = colnames(fv_sub))
  fv_sub_long[, stdfvalue := std1(value), by = "variable"]
  # SHAP value: value
  # raw feature value: rfvalue; 
  # standarized: stdfvalue
  names(fv_sub_long) <- c("variable", "rfvalue", "stdfvalue" )
  shap_long2 <- cbind(shap_score_long, fv_sub_long[,c('rfvalue','stdfvalue')])
  shap_long2[, mean_value := mean(abs(value)), by = variable]
  setkey(shap_long2, variable)
  return(shap_long2) 
}

plot.shap.summary <- function(data_long){
  x_bound <- max(abs(data_long$value))
  require('ggforce') # for `geom_sina`
  plot1 <- ggplot(data = data_long)+
    coord_flip() + 
    # sina plot: 
    geom_sina(aes(x = variable, y = value, color = stdfvalue)) +
    # print the mean absolute value: 
    geom_text(data = unique(data_long[, c("variable", "mean_value"), with = F]),
              aes(x = variable, y=-Inf, label = sprintf("%.3f", mean_value)),
              size = 3, alpha = 0.7,
              hjust = -0.2, 
              fontface = "bold") + # bold
    # # add a "SHAP" bar notation
    # annotate("text", x = -Inf, y = -Inf, vjust = -0.2, hjust = 0, size = 3,
    #          label = expression(group("|", bar(SHAP), "|"))) + 
    scale_color_gradient(low="#FFCC33", high="#6600CC", 
                         breaks=c(0,1), labels=c("Low","High")) +
    theme_bw() + 
    theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(), 
          legend.position="bottom") + # remove axis line
    geom_hline(yintercept = 0) + # the vertical line
    scale_y_continuous(limits = c(-x_bound, x_bound)) +
    # reverse the order of features
    scale_x_discrete(limits = rev(levels(data_long$variable)) 
    ) + 
    labs(y = "SHAP value (impact on model output)", x = "", 
         color = "Feature value") 
  return(plot1)
}






var_importance <- function(shap_result, top_n=10)
{
  var_importance=tibble(var=names(shap_result$mean_shap_score), 
                        importance=shap_result$mean_shap_score)
  
  var_importance=var_importance[1:top_n,]
  
  ggplot(var_importance, aes(x=reorder(var,importance), y=importance)) + 
    geom_bar(stat = "identity") + 
    coord_flip() + 
    theme_light() + 
    theme(axis.title.y=element_blank()) 
}

# Variable imptance is even more important. 

# Extract importance
set.seed(69696969)
imp <- catboost.get_feature_importance(model, pool= train_pool,
                                       type = 'ShapValues',
                                       thread_count = -1)
imp <- as.data.frame(imp) #' Convert importance matrix to dataframe
names(imp) <- names(train_use) #' Change variable names to match data
imp #"print importance


######################## INSIGHTS #######################

RaceVector <- unique(train_use$Victim_Race) #' Create vector of races

avg_SHAPbyRace <- rep(NA, length(RaceVector)) 
#' Create empty vector length RaceVector

for (i in 1:length(RaceVector)) { #' Loop through RaceVector
  avg_SHAPbyRace[i] <- 
    #' Assign average SHAP value for each race to appropriate race in RaceVector
    mean(imp$Victim_Race[which(train_use$Victim_Race == RaceVector[i])],
         na.rm = TRUE)
}

AverageRaceShapDf <- cbind.data.frame(RaceVector, avg_SHAPbyRace)
#' Create DF of results
AverageRaceShapDf


names(imp)
imp <- imp[c(1:24)] #' Remove NA Variable

shap_contrib <- as.data.table(imp[,c(17:24)])
mean_shap_score <- 
  colMeans(abs(shap_contrib))[order(colMeans(abs(shap_contrib)), 
                                    decreasing = T)]
mean_shap_score #' Find mean Shap Score for each numeric variable in dataset

shap_result <- list(shap_score = shap_contrib,
     mean_shap_score = (mean_shap_score))

var_importance(shap_result, top_n = 8) 
#' Plot Variable importance for numeric variables

shap_long = shap.prep(shap = shap_result,
                      X_train = train_use[,c(17:24)], 
                      top_n = 8)
shap <- shap_result
X_train <- train_use[,c(17:24)]
top_n <- 8
shap_score_sub <- as.data.table(shap$shap_score)
shap_score_sub <- shap_score_sub[, names(shap$mean_shap_score)[1:top_n], 
                                 with = F]
shap_score_long <- melt.data.table(shap_score_sub, 
                                   measure.vars = colnames(shap_score_sub))

# feature values: the values in the original dataset
fv_sub <- as.data.table(X_train)[, names(shap$mean_shap_score)[1:top_n], 
                                 with = F]
# standardize feature values
fv_sub_long <- melt.data.table(fv_sub, measure.vars = colnames(fv_sub))
fv_sub_long[, stdfvalue := std1(value), by = "variable"]
# SHAP value: value
# raw feature value: rfvalue; 
# standarized: stdfvalue
names(fv_sub_long) <- c("variable", "rfvalue", "stdfvalue" )
shap_long2 <- cbind(shap_score_long, fv_sub_long[,c('rfvalue','stdfvalue')])
shap_long2[, mean_value := mean(abs(value)), by = variable]
setkey(shap_long2, variable)


plot.shap.summary(data_long = shap_long) #' Plot Shap Variable Importance


############# Final Insight Visualizations ######################
imp2 <- imp
names(imp2) <- paste("SHAP_",names(imp),sep = "")

Pdata1 <- cbind.data.frame(train_data, imp2)

PctWhitevsBodyCamPredictions <- ggplot(Pdata1, aes(x = Share_White, 
                                                   y = SHAP_Share_White)) +
  geom_point(alpha = 0.5) +
  facet_wrap(vars(Victim_Race)) + 
  #' wrap by race
  theme_set(theme_bw(base_size = 15) ) + # Set theme and text size
  theme(title = element_text(size = 10),
        strip.text.x = element_text(size=8),
        panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) + # Remove grid 
  labs(x = "Percent White", 
       title = "Percent White vs Body Cam Footage by Race of Victim",
       fill = "Camera On") # Set labels
PctWhitevsBodyCamPredictions

RacevsRaceShap <- ggplot(AverageRaceShapDf, aes(x = RaceVector, 
                                                y = avg_SHAPbyRace, 
                                                fill = RaceVector)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Race", y = "Average SHAP Value by Race",
       title = "Race of Victim vs Average SHAP Value by Race") # Set labels
  #' use the actual y value we feed it for the y-axis. 
  #' "dodge" unstacks the bars
RacevsRaceShap


summary(plot_data$BodyCamBinary)


summary(THE_FINAL_DATA_FRAME)




### THE END. I hope you enjoyed it. Machine Learning has been [a] class. :-)

## See you in Sports Analytics!! 