library(data.table)
library(ggplot2)
library(glmnet)
library(boot)
library(randomForest)
library(gbm)
library(Boruta)
library(xgboost)
library(class)

data_train <- fread('train.csv')
data_test <- fread('test.csv')

dt <- rbind(data_train, data_test, fill=T)

################## Feature formatting and engineering ####################
dt1 <- data.table(dt)

dt1 <- dt1[, 'Id':=NULL]
dt1 <- dt1[, 'SalePrice':=NULL]
dt1$dummy <- 0

dt1$MSSubClass <- as.character(dt1$MSSubClass)

map_1 <- c('Ex'=4, 'Gd'=3, 'TA'=2, 'Fa'=1, 'Po'=0)
map_2 <- c('Gd'=4, 'Av'=3, 'Mn'=2, 'No'=1)
map_3 <- c('Min1'=5, 'Min2'=4, 'Maj1'=3, 'Maj2'=2, 'Sal'=1)
map_4 <- c('Fin'=3, 'RFn'=2, 'Ufn'=1)

assign_level <- function(x, map=map_1){
  map_qual <- c('Ex'=4, 'Gd'=3, 'TA'=2, 'Fa'=1, 'Po'=0)
  x <- sapply(x, function(y){map[y]})
  x[is.na(x)] <- 0
  return(x)
}

dt1$ExterQual <- assign_level(dt$ExterQual)
dt1$ExterCond <- assign_level(dt$ExterCond)
dt1$BsmtQual <- assign_level(dt$BsmtQual)
dt1$BsmtCond <- assign_level(dt$BsmtCond)
dt1$HeatingQC <- assign_level(dt$HeatingQC)
dt1$KitchenQual <- assign_level(dt$KitchenQual)
dt1$FireplaceQu <- assign_level(dt$FireplaceQu)
dt1$GarageQual <- assign_level(dt$GarageQual)
dt1$GarageCond <- assign_level(dt$GarageCond)
dt1$PoolQC <- assign_level(dt$PoolQC)
dt1$BsmtExposure <- assign_level(dt$BsmtExposure, map_2)
dt1$Functional <- assign_level(dt$Functional, map_3)
dt1$GarageFinish <- assign_level(dt1$GarageFinish, map_4)

na_sum <- apply(dt1, 2, function (x){sum(is.na(x))})
for (col in which(na_sum>0)){
  if (class(dt1[[col]])=='character') {
    dt1[is.na(dt1[[col]]), col] <- 'NONE'
  } else {dt1[is.na(dt1[[col]]), col] <- min(dt1[[col]], na.rm = T)}
}
na_row <- apply(dt1, 2, function(x){which(is.na(x))})

##################### Explorative Analysis using the plot window ######################
# numerical <- which(lapply(dt1, class)!='character')
# dt1.num <- dt1[, names(numerical), with=F]
# colnames(dt1.num) <- sapply(colnames(dt1.num), function(x){paste0('xx', x)})
# for (fac in colnames(dt1.num)){
#   plot.new()
#   par(mfrow=c(2,1))
#   plot(dt1.num[[fac]], dt$SalePrice, xlab=fac)
#   hist(dt1.num[[fac]], xlab=fac, breaks=20)
#   readline(prompt="Press [enter] to continue")
# }
# 
# charac <- which(lapply(dt1, class)=='character')
# dt1.chr <- dt1[, names(charac), with=F]
# colnames(dt1.chr) <- sapply(colnames(dt1.chr), function(x){paste0('xx', x)})
# for (fac in colnames(dt1.chr)){
#   data1 <- data.frame(fac = dt1.chr[[fac]], 'SalePrice'=dt$SalePrice)
#   boxplot(SalePrice~fac, data=data1, xlab=fac)
#   readline(prompt="Press [enter] to continue")
# }
# 
# hist(dt$SalePrice)
#################### Prepare train set #######################
x <- model.matrix(dummy~., data=dt1)[, -1]
y <- as.double(dt$SalePrice)
train_x <- x[which(!is.na(dt$SalePrice)),]
train_y <- y[!is.na(dt$SalePrice)]
train_y <- log(train_y)+1
train_xy <- data.frame(train_x, 'SalePrice'=train_y)
test_x <- x[which(is.na(dt$SalePrice)),]
# 
##################### Boruta analysis and new train/test sets #######################
# # boruta.train <- Boruta(SalePrice~., data=train_xy)
# boruta.train <- readRDS('boruta.rds')
# 
# impHist <- data.table(boruta.train$ImpHistory)
# impHist[abs(impHist)==Inf] <- NA
# 
# impMean <- apply(impHist, 2, function(x){mean(x, na.rm=T)})
# impMean <- data.table('fac'=names(impMean), 'impMean'=impMean)
# impMean <- impMean[order(impMean, decreasing = T)]
# impMean <- impMean[, 'decision':=boruta.train$finalDecision[fac]]
# 
# confirmed <- impMean[decision=='Confirmed', fac]
# x.bor <- x[, which(colnames(x) %in% confirmed)]
# train_x.bor <- x.bor[which(!is.na(dt$SalePrice)),]
# train_xy.bor <- data.frame(train_x.bor, 'SalePrice'=train_y)
# test_x.bor <- x.bor[which(is.na(dt$SalePrice)),]

################# XGB ####################
log_xgb <- c()
for (eta_ in seq(0.02, 0.4, 0.02)){
  for (max_depth in seq(6, 26, 2)) {
    model.xgbcv <- xgb.cv(data=train_x, label=train_y,
                          nround=1000, 
                          nfold=5, 
                          eta=eta_,
                          objective='reg:linear', 
                          booster='gbtree',  
                          max_depth=max_depth,
                          metrics='rmse', 
                          verbose=T)
    xgb.rmse <- min(model.xgbcv$evaluation_log$test_rmse_mean)
    # plot(model.xgbcv$evaluation_log$train_rmse_mean, type='l', col='red', add=T)
    # lines(model.xgbcv$evaluation_log$test_rmse_mean, col='blue')
    log_xgb <- rbind(log_xgb, c(eta_, max_depth, xgb.rmse))
  }
}
best.xgb.index <- which(log_xgb == min(log_xgb[,3]), arr.ind = T)[1]

xgb <- xgboost(data = train_x, label = train_y, nrounds = 1000, eta=log_xgb[best.xgb.index, 1], max_depth=log_xgb[best.xgb.index, 2])
plot(xgb$evaluation_log$train_rmse, type='l', col='red')


# ################## Random Forest ###################
log_rf <- c()
for (mtry in seq(40, 160, 20)){
  model.rfcv <- randomForest(train_x, train_y, ntree=2500, mtry=mtry)
  log_rf <- rbind(log_rf, c(mtry, min(model.rfcv$mse)))
}
best.rf.index <- which(log_rf==min(log_rf[, 2]), arr.ind = T)[1]
model.rf <- randomForest(train_x, train_y, ntree=2500, mtry=log_rf[best.rf.index, 1])

# ################# GLMnet ####################
log_glmnet <- c()
for (alpha in seq(0, 1, 0.05)){
  model.glmnetcv <- cv.glmnet(train_x.bor, train_y, alpha=alpha, nfolds = 5, lambda = 10^seq(-10, 10, length=200), type.measure = 'mse')
  log_glmnet <- rbind(log_glmnet, c(alpha, model.glmnetcv$lambda.min, min(model.glmnetcv$cvm)))
}
best.glmnet.index <- which(log_glmnet==min(log_glmnet[, 3]), arr.ind = T)[1]
model.glmnet <- glmnet(train_x, train_y, alpha=log_glmnet[best.glmnet.index, 1], lambda=log_glmnet[best.glmnet.index, 2])

#################### GLM ########################
model.glm <- glm(SalePrice~., data=train_xy, family = 'gaussian')
model.glmcv <- cv.glm(model.glm, data=train_xy.bor, K=10)
# glm.rmse <- sqrt(model.glmcv$delta)[2]

#################### Prepare for Ensemble ########################
pred.xgb <- predict(xgb, train_x)
pred.rf <- predict(model.rf, train_x)
pred.glmnet <- predict(model.glmnet, train_x)
pred.glm <- predict(model.glm, data.frame(train_x))

train_ensemble <- data.frame(pred.xgb, pred.rf, pred.glmnet, pred.glm)
train_ensemble <- as.matrix(train_ensemble)

#################### glmnet ensemble ########################
log_glmnet2 <- c()
for (alpha in seq(0, 1, 0.05)){
  model.glmnetcv2 <- cv.glmnet(train_x.bor, train_y, alpha=alpha, nfolds = 5, lambda = 10^seq(-10, 10, length=200), type.measure = 'mse')
  log_glmnet2 <- rbind(log_glmnet2, c(alpha, model.glmnetcv2$lambda.min, min(model.glmnetcv2$cvm)))
}
best.glmnet2.index <- which(log_glmnet2==min(log_glmnet2[, 3]), arr.ind = T)[1]
model.glmnet2 <- glmnet(train_x, train_y, alpha=log_glmnet[best.glmnet2.index, 1], lambda=log_glmnet[best.glmnet2.index, 2])


#################### XGB Ensemble ########################
train_ensemble <- data.frame(pred.xgb, pred.rf, pred.glmnet, pred.glm)
train_ensemble <- as.matrix(train_ensemble)
# train_ensemble <- exp(train_ensemble-1)
# train_y2 <- exp(train_y-1)

log_xgb2 <- c()
for (eta_ in seq(0.02, 0.5, 0.03)){
  for (max_depth in c(1:10)) {
    model.xgbcv2 <- xgb.cv(data=train_ensemble, label=train_y,
                          nround=1000,
                          nfold=5,
                          eta=0.1,
                          objective='reg:linear',
                          booster='gbtree',
                          max_depth=max_depth,
                          metrics='rmse',
                          verbose=T)
    xgb2.rmse <- min(model.xgbcv2$evaluation_log$test_rmse_mean)
    log_xgb2 <- rbind(log_xgb2, c(eta_, max_depth, xgb2.rmse))
  }
}
 best.xgb2.index <- which(log_xgb2 == min(log_xgb2[,3]), arr.ind = T)[1]
xgb2 <- xgboost(data = train_x, label = train_y, nrounds = 1000, eta=log_xgb2[best.xgb2.index, 1], max_depth=log_xgb2[best.xgb2.index, 2])

#################### Submission ########################
submit <- function(model, newdata=test_x){
  prediction <- predict(model, newdata)
  prediction <- exp(prediction-1)
  submission <- data.frame(data_test$Id, prediction)
  names(submission) <- c('Id', 'SalePrice')
  write.csv(submission, 'House Price Submission.csv', row.names=F)
}
