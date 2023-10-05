# BankMarketing
Machine Learning Model

## Reading the dataset
```{r}
Bank = read.table("bank-additional.csv", header = TRUE,sep=";")
```

## To view the dataset
```{r}
View(Bank)
```

```{r}
names(Bank)
```

```{r}
str(Bank)
```
```{r}
summary(Bank)
```
## To check whether the dataset have the null values

```{r}
sum(is.na(Bank))
```
## Explorating Data Analysis

# Color code based on Type of job variable

```{r}
ggplot(data=Bank)+ geom_bar(mapping = aes(x = y, fill = job), position = "dodge")
```

# Interpretation:
we can see an unknown factor 


# Color code based on marital status
```{r}
ggplot(data=Bank)+ geom_bar(mapping = aes(x = y, fill = marital), position = "dodge")
```



# Color code based on education level

```{r}
ggplot(Bank,aes(education))+geom_bar(aes(fill= y), position = position_dodge())+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
```



## Age Variable
```{r}
ggplot(data = Bank, mapping = aes(x = y, y = age))+ geom_boxplot()
```


## Housing Variable

```{r}
ggplot(Bank,aes(housing))+geom_bar(aes(fill= y), position = position_dodge())+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
```


## Contact variable
```{r}
ggplot(data=Bank)+ geom_bar(mapping = aes(x = y, fill = contact), position = "dodge")
```



## Removing unknow values 

```{r}
Bank = subset(Bank, !(default == "unknown"))
Bank = subset(Bank, !(housing == "unknown"))
Bank = subset(Bank, !(loan == "unknown"))
Bank = subset(Bank, !(job == "unknown"))
Bank = subset(Bank, !(marital == "unknown"))
Bank = subset(Bank, !(education == "unknown"))
```

## Checkning if unknow values are removed
```{r}
table(Bank$default)
```


## As recommended removing duration variable as it is influential varaible
```{r}
Bank = dplyr::select(Bank,-duration)
```

## Checking whether duration variable is removed
```{r}
names(Bank)
```


## Making character variables in to factors
```{r}
Bank$job = as.factor(Bank$job)
Bank$marital = as.factor(Bank$marital)
Bank$education = as.factor(Bank$education)
Bank$default = as.factor(Bank$default)
Bank$housing  = as.factor(Bank$housing)
Bank$loan = as.factor(Bank$loan)
Bank$contact = as.factor(Bank$contact)
Bank$month = as.factor(Bank$month)
Bank$day_of_week = as.factor(Bank$day_of_week)
Bank$y = as.factor(Bank$y)
Bank$poutcome = as.factor(Bank$poutcome)
```


# Checking the levels of the the dependant variable

```{r}
levels(as.factor(Bank$y))
```

## Check for correlation
```{r}
Bank_num = dplyr::select_if(Bank, is.numeric)
C = cor(Bank_num)
```


# Plot the correlation matrix
```{r}
corrplot(C, method = "number")
```


## Interpretation:
It is observed that euribor3m and emp.var.rate are highly correlated



# Overview of the data distribution

## Computing scatter plots for each variable

```{r}
pairs(Bank_num)
```


# Ploting the dependent variable

```{r}
ggplot(data=Bank)+ geom_bar(mapping = aes(x = y))
```


# Interpretation:
The above graph indicate the y variable in terms of count it is observed that, number of no's are more than number of yes and it indicates that the data is imbalance.



## Spliting the data in to Training & Testing

```{r}
set.seed(1)
```

```{r}
tr_ind = sample(nrow(Bank),0.8*nrow(Bank), replace = F)
banktrain = Bank[tr_ind,]
banktest = Bank[-tr_ind,]
```


## Build logistic regression model
```{r}
r1 = glm(y ~ ., data = banktrain, family = binomial)
```

```{r}
summary(r1)
```

## Removing least significant variables - Model Selection - Backward Elimination

```{r}
r2 = step(r1, direction = "backward")
```


# Interpretation: 
Removing the variables which are least significant by using AIC - Backward elimination

```{r}
summary(r2)
```

## Checking the Multicollinearity

```{r}
vif(r2)
```




## Make predictions for logistic regressions

```{r}
predprob = predict.glm(r2, newdata = banktest, type = "response")
```

```{r}
predclass_log = ifelse(predprob >= 0.5, "yes", "no")
```

```{r}
caret::confusionMatrix(as.factor(predclass_log), as.factor(banktest$y), positive = "yes")
```

## ROC Curve

```{r}
pred <- prediction(predict(r2, banktest, type = "response"),
                   banktest$y) #Predicted Probability and True Classification

auc <- round(as.numeric(performance(pred, measure = "auc")@y.values),3)

```
```{r}
false.rates <-performance(pred, "fpr","fnr")
accuracy <-performance(pred, "acc","err")
perf <- performance(pred, "tpr","fpr")
plot(perf,colorize = T, main = "ROC Curve")
text(0.5,0.5, paste("AUC:", auc))
```

```{r}
plot(unlist(performance(pred, "sens")@x.values), unlist(performance(pred, "sens")@y.values),
     type="l", lwd=2,
     ylab="Sensitivity", xlab="Cutoff", main = paste("Maximized Cutoff\n","AUC: ",auc))
par(new=TRUE)
plot(unlist(performance(pred, "spec")@x.values), unlist(performance(pred, "spec")@y.values),
     type="l", lwd=2, col='red', ylab="", xlab="")
axis(4, at=seq(0,1,0.2))
mtext("Specificity",side=4, padj=-2, col='red')

min.diff <-which.min(abs(unlist(performance(pred, "sens")@y.values) - unlist(performance(pred, "spec")@y.values)))
min.x<-unlist(performance(pred, "sens")@x.values)[min.diff]
min.y<-unlist(performance(pred, "spec")@y.values)[min.diff]
optimal <-min.x #this is the optimal points to best trade off sensitivity and specificity

abline(h = min.y, lty = 3)
abline(v = min.x, lty = 3)
text(min.x,0,paste("optimal threshold=",round(optimal,2)), pos = 4)
```


## Making predictions with optimal cutoff

```{r}
predprob = predict.glm(r2, newdata = banktest, type = "response")
```


```{r}
predclass_log = ifelse(predprob >= 0.07, "yes", "no")
```

```{r}
caret::confusionMatrix(as.factor(predclass_log), as.factor(banktest$y), positive = "yes")
```


# Linear Discriminate analysis - Comparision for logistic regression model


```{r}
m1.lda = lda(as.factor(y) ~ ., data = banktrain)
```


## Make predictions for LDA

```{r}
predclass_lda = predict(m1.lda, newdata = banktest)
```

```{r}
caret::confusionMatrix(as.factor(predclass_lda$class), as.factor(banktest$y), positive = "yes")
```












