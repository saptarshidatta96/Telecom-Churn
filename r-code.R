setwd("C:\\Users\\Saptarshi Datta\\Desktop\\R PROGRAMMING")
data=read.csv("Cellphone.csv",header=TRUE)
str(data)
attach(data)
hist(data)
boxplot(data)
cor(data)
cor(Churn,ContractRenewal)
data$Churn_fact=as.factor(data$Churn)
data$ContractRenewal_fact=as.factor(data$ContractRenewal)
data$DataPlan_fact=as.factor(data$DataPlan)
summary(data)
str(data)

data_fact=data[,-c(1,3,4)]
data_num=data[,-c(12,13,14)]

str(data_fact)
str(data_num)

###Basic EDA
library(funModeling)
library(tidyverse) 
library(Hmisc)

basic_eda <- function(data)
{
  summary(data)
  df_status(data)
  freq(data) 
  profiling_num(data)
  hist(data)
  describe(data)
  
}
basic_eda(data)

#Univariate analysis
summary(AccountWeeks)
boxplot(AccountWeeks)
summary(DataUsage)
boxplot(DataUsage)
summary(CustServCalls)
boxplot(CustServCalls)
summary(DayMins)
boxplot(DayMins)
summary(DayCalls)
boxplot(DayCalls)
summary(MonthlyCharge)
boxplot(MonthlyCharge)
summary(OverageFee)
boxplot(OverageFee)
summary(RoamMins)
boxplot(RoamMins)

summary(Churn)


hist(DataUsage)
hist(CustServCalls)
hist(MonthlyCharge)


plot(DataUsage,MonthlyCharge)
plot(MonthlyCharge,DayMins)
boxplot(data)
is.na(data)
data=na.omit(data)
names(data)

#collinearity

str(data)
library(corrplot)
cor_data=cor(data[,-c(12,13,14)])
corrplot(cor_data , method='number')
library(car)

lm(Churn~.,data=data_num)
vif(lm(Churn~.,data=data_num))


vif(glm(Churn_fact~.,data=data_fact,family=binomial))
summary(glm(ContractRenewal_fact~.,data=data_fact,family=binomial))

summary(glm(Churn_fact~AccountWeeks+DataUsage+CustServCalls+DayMins+DayCalls+
              MonthlyCharge+OverageFee+RoamMins+ContractRenewal_fact,
            data=data_fact,family=binomial))

vif(glm(Churn_fact~AccountWeeks+DataUsage+CustServCalls+DayMins+DayCalls+
          MonthlyCharge+OverageFee+RoamMins+ContractRenewal_fact,
        data=data_fact,family=binomial))



cor(data_num)



library(caTools)
sample = sample.split(data_fact$Churn_fact, SplitRatio = .70)
train = subset(data_fact, sample == TRUE)
test  = subset(data_fact, sample == FALSE)



####Logistic Regression
set.seed(1234)
# 6 steps in logistic 

# Step 1: Overall Validity of the Model #

# 1) Log likelihood Test

logit=glm(Churn_fact~AccountWeeks+CustServCalls+DayCalls+RoamMins+ContractRenewal_fact,
          data=train,family=binomial)
summary(logit)

library(car)
vif(logit)
library(lmtest)
lrtest(logit)

#step 2: McFadden Rsq
options(scipen=999)
library(pscl)
print(pR2(logit))

#step 3: Indiviudal Slopes Significance Test
summary(logit)
#Estimates for the variables
print(logit)

#Step 4: Explanatory Power of odds

round(exp(coef(logit)),2)

probability_Scores=format(exp(coef(logit))/(exp(coef(logit))+1),scientific = FALSE)

print(probability_Scores)

#step 5 : Classification / Confusion Matrix

Pred.logit=predict(logit,type = "response",data=train)
summary(Pred.logit)
summary(train$Churn_fact)
plot(train$Churn_fact,Pred.logit)
Pred.logit=ifelse(Pred.logit<0.14,0,1)
table(Actual=train$ContractRenewal_fact,Pred.logit)


#step 6: ROC Curve

library(Deducer)
rocplot(logit)

library(caret)
##Testing the model on the Test Data

Pred.logit.test=predict(logit,type = "response",newdata = test)
summary(Pred.logit.test)
summary(test$Churn_fact)
plot(test$Churn_fact,Pred.logit.test)
Pred.logit.test.factor=ifelse(Pred.logit.test<0.20,0,1)
confusionMatrix(table(Actual=test$Churn_fact,Pred.logit.test.factor))



library(blorr) # to build and validate binary logistic models

blr_step_aic_both(logit, details = FALSE)


#ModelPerformanceParameter
#Train
train$prediction = predict(logit, train, type="response")
library(ROCR)
library(ineq)
predObj = prediction(train$prediction, train$Churn_fact)
perf = performance(predObj, "tpr", "fpr")
plot(perf)
KS = max(perf@y.values[[1]]-perf@x.values[[1]])
auc = performance(predObj,"auc"); 
auc = as.numeric(auc@y.values)
gini = ineq(train$prediction, type="Gini")

#Test
test$prediction = predict(logit, test, type="response")
library(ROCR)
library(ineq)
predObj = prediction(test$prediction, test$Churn_fact)
perf = performance(predObj, "tpr", "fpr")
plot(perf)
KS = max(perf@y.values[[1]]-perf@x.values[[1]])
auc = performance(predObj,"auc"); 
auc = as.numeric(auc@y.values)
gini = ineq(test$prediction, type="Gini")




#Use KNN Classifier 
#normalize the test & train data

norm=function(x){(x-min(x))/(max(x)-min(x))}
norm.data=as.data.frame(lapply(data_fact[,-c(9,10,11)],norm))
norm.data_fact=cbind(data_fact[,c(9,10,11)],norm.data)

#split the normalized dataset
library(caTools)
sample = sample.split(norm.data_fact$Churn_fact, SplitRatio = .70)
norm.train = subset(norm.data_fact, sample == TRUE)
norm.test  = subset(norm.data_fact, sample == FALSE)


library(class)
knn.pred = knn(norm.train[-c(1)], norm.test[,-c(1)], norm.train[,1], k = 19) 
table.knn = table(norm.test$Churn_fact, knn.pred)
table.knn
sum(diag(table.knn)/sum(table.knn)) 
confusionMatrix(table.knn)


#Naive Bayes

library(e1071)
NB = naiveBayes(Churn_fact~AccountWeeks+CustServCalls+DayCalls+RoamMins+ContractRenewal_fact, data = train)
predNB = predict(NB, test, type = "class")
tab.NB = table(test[,9], predNB)
sum(diag(tab.NB)/sum(tab.NB))
confusionMatrix(tab.NB)
tab.NB


