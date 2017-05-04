library(data.table)
library(ggplot2)
library(stringr)
library(caret)
library(dplyr)
library(boot)

dt1 <- fread('train.csv')
dt2 <- fread('test.csv')
dt <- merge(dt1, dt2, all = T)
dt$Survived <- as.factor(dt$Survived)
answer <- fread('gender_submission.csv')

####### Split the names into different parts #######
name_split <- strsplit(dt$Name, ', ', fixed = TRUE)
first_name_split <- strsplit(sapply(name_split, '[[', 2), '. ', fixed=T)

dt$surname <- sapply(name_split, '[[', 1)
dt$surname <- sapply(strsplit(dt$surname, '-', fixed=T),  '[[', 1)

dt$title <- sapply(first_name_split, '[[', 1)
dt_firstname <- sapply(first_name_split, '[[', 2)
re1 <- "\\(([^()]+)\\)"
dt <- dt[grepl('(', `Name`, fixed=T), 'b_name':=gsub(re1, "\\1", str_extract_all(`Name`, re1))]
dt <- dt[!is.na(b_name), 'b_surname':=sapply(strsplit(b_name, ' '), tail, 1)]

re2='\\"([^""]+)\\"'
dt <- dt[grepl('"', `Name`, fixed=T), 'nickname':=gsub(re2, "\\1", str_extract_all(`Name`, re2))]

########Get Cabin block with available#######
dt <- dt[!is.na(Cabin), 'CabinBLK':= substr(Cabin, start=1, stop=1)]
dt[CabinBLK=='', 'CabinBLK'] <- 'Unknown'
dt$CabinBLK <- as.factor(dt$CabinBLK)

##########Get Family size###########
dt$fam_size <- dt$SibSp + dt$Parch + 1

##########Create a custom Family Identifier to compare with fam_size#########
dt$FamGrp <- as.integer('')
dt$FamGrpSize <- as.integer(1)

##########Filter out all singletons#########
dt[, 'Alone'] <- ifelse(dt$fam_size>1, 'Alone', 'Group')
mask1 <- dt$fam_size>1

##########Assume all fams embarked from same location, share same cabin class#########
dt <- dt[mask1, 'Grp1':=.GRP, by=c('Embarked', 'Pclass', 'surname')]
dt <- dt[mask1, 'Grp1Size':=.N, by='Grp1']
dt <- dt[mask1, 'FamGrp':=Grp1]
dt <- dt[mask1, 'FamGrpSize':=.N, by='FamGrp']

##########See the ones we haven't matched properly###########
mask2 <- as.logical(mask1 * (dt$FamGrpSize<dt$fam_size))

######Froup by Ticket number to enhance accuracy##########
dt <- dt[mask2, 'Grp2':=.GRP, by='Ticket']
dt <- dt[mask2, 'Grp2Size':=.N, by='Grp2']
dt$Grp2 <- dt$Grp2+as.integer(1000)
dt <- dt[mask2 & Grp2Size>=fam_size, 'FamGrp':=Grp2]
dt <- dt[, 'FamGrpSize':=.N, by='FamGrp']

#########Check again what's left from the last classification##########
mask3 <- as.logical(mask1 * (dt$FamGrpSize<dt$fam_size))

#########Group people in same cabin together########
dt <- dt[mask3 & Cabin!='', 'Grp3':=.GRP, by='Cabin']
dt$Grp3 <- dt$Grp3+as.integer(2000)
dt <- dt[mask3 & Cabin!='', 'Grp3Size':=.N, by='Grp3']
dt <- dt[mask3 & Cabin!='' & Grp3Size>=fam_size, 'FamGrp':=Grp3]
dt <- dt[, 'FamGrpSize':=.N, by='FamGrp']

#########Check again what's left from the last classification##########
mask4 <- as.logical(mask1 * (dt$FamGrpSize<dt$fam_size))

#########Check in the in-laws, Richards is a special case...##########
cousins <- dt[mask4 & (surname %in% dt$b_surname) & !surname %in% c('Richards')]
match <- dt[!(PassengerId %in% cousins$PassengerId) & mask1, list('FamGrp'=FamGrp[1]), by='b_surname']
cousins <- merge(cousins, match, by.x='surname', by.y='b_surname', all.x=T)
dt <- merge(dt, cousins[, list('PassengerId'=PassengerId, 'Grp4'=FamGrp.y)], by='PassengerId', all.x=T)
dt <- dt[!is.na(Grp4), 'FamGrp':=Grp4]
dt <- dt[, 'FamGrpSize':=.N, by='FamGrp']

#########Check again what's left from the last classification##########
mask5 <- as.logical(mask1 * (dt$FamGrpSize<dt$fam_size))

#########match the surnames in COUSINS with surnames in dt[!COUSINS]##########
cousins2 <- dt[mask5]
match <- dt[!(PassengerId %in% cousins2$PassengerId) & mask1, list('FamGrp'=FamGrp[1]), by='surname']
cousins2 <- merge(cousins2, match, by='surname', all.x=T)
dt <- merge(dt, cousins2[, list('PassengerId'=PassengerId, 'Grp5'=FamGrp.y)], by='PassengerId', all.x=T)
dt <- dt[!is.na(Grp5), 'FamGrp':=Grp5]
dt <- dt[, 'FamGrpSize':=.N, by='FamGrp']

#########Check again what's left from the last classification##########
mask6 <- as.logical(mask1 * (dt$FamGrpSize<dt$fam_size))

#########Take the lazy way and group the ticket numbers by resemblance##########
cousins3 <- dt[mask6]
cousins3$TicHead <- substr(cousins3$Ticket, start=1, stop=3)
cousins3 <- cousins3[, 'Grp6':=.GRP, by=TicHead]
cousins3$Grp6 <- cousins3$Grp6+as.integer(3000)
dt <- merge(dt, cousins3[, c('PassengerId','Grp6')], by='PassengerId', all.x=T)
dt <- dt[!is.na(Grp6), 'FamGrp':=Grp6]
dt <- dt[, 'FamGrpSize':=.N, by='FamGrp']

########Impute Age########
impute <- dt[!is.na(Age), list('mean age'=mean(Age)), by=c('title')]
impute$`mean age` <- as.numeric(impute$`mean age`)
dt[is.na(Age), 'Age'] <- sapply(dt[is.na(Age)]$title, function(x) impute$`mean age`[match(x, impute$title)])

########Isolate the single-moms########
dt[, 'LadyCount'] <- sapply(dt$FamGrp, function (x) {dt[FamGrp==x, sum(Age>12, Sex=='female')]})
dt[, 'KidsCount'] <- sapply(dt$FamGrp, function (x) {dt[FamGrp==x, sum(Age<=12)]})
dt[, 'GrownManCount'] <- sapply(dt$FamGrp, function (x) {dt[FamGrp==x, sum(Sex=='male' & Age>=18)]})
dt[, 'ElderlyCount'] <- sapply(dt$FamGrp, function (x) {dt[FamGrp==x, sum(Age>=55)]})

#########Make a couple plots to explore the family stats########
Fam <- dt[mask1 & !is.na(dt$Survived)]
Fam$FamGrpSize <- as.integer(Fam$FamGrpSize)

g1 <- ggplot(Fam) + 
  geom_density(aes(x = FamGrpSize, fill = Survived), adjust=2, alpha=0.5) + 
  scale_x_continuous(breaks = c(1:max(Fam$FamGrpSize)))

g2 <- ggplot(Fam, aes(x=Survived, y=Age, fill=Survived)) +
  geom_violin()+
  geom_jitter(aes(color=Sex))

dt_g3 <- dt[!is.na(Survived), list('Count'=.N), by=c('Alone', 'Sex', 'Survived')]
g3 <- ggplot(dt_g3, aes(x=Alone, y= Count, fill=Survived)) +
  geom_bar(position='fill', stat='identity') +
  facet_grid(Sex~.)

g4 <- ggplot(dt[!is.na(Age) & !is.na(Survived)], aes(x=Age, fill=Survived)) +
  geom_density(alpha=0.5) +
  facet_grid(Sex~Alone)

Kids <- dt[FamGrpSize>1 & !is.na(Survived), list('SurvivalRate'=mean(as.numeric(as.character(Survived)))), by=c('KidsCount')]
g5 <- ggplot(Kids, aes(x=KidsCount, y=SurvivalRate)) +
  geom_line()
g5

ManvKidsMrs <- dt[FamGrpSize>1 & !is.na(Survived), list('SurvivalRate'=mean(as.numeric(as.character(Survived)))), by=c('GrownManCount')]
g6 <- ggplot(ManvKidsMrs, aes(x=GrownManCount, y=SurvivalRate)) +
  geom_line()

Lady <- dt[FamGrpSize>1 & !is.na(Survived), list('SurvivalRate'=mean(as.numeric(as.character(Survived)))), by=c('LadyCount')]
g7 <- ggplot(Lady, aes(x=LadyCount, y=SurvivalRate)) +
  geom_line()

Elderly <- dt[FamGrpSize>1 & !is.na(Survived), list('SurvivalRate'=mean(as.numeric(as.character(Survived)))), by=c('ElderlyCount')]
g8 <- ggplot(Elderly, aes(x=ElderlyCount, y=SurvivalRate)) +
  geom_line()
#########Run a few models#########
train_set <- dt[PassengerId %in% dt1$PassengerId]
test_set <- dt[PassengerId %in% dt2$PassengerId]
train_control <- trainControl(method='cv', number=5)

#Logistic Regression
# Reg <- train(Survived~poly(Age, 2)+Sex+Pclass+poly(FamGrpSize, 2)+poly(LadyCount, 3)+poly(KidsCount, 2)+poly(GrownManCount, 2)+poly(ElderlyCount, 2), data=train_set, trControl=train_control, method='glm', family = 'binomial')
# test_fit <- predict(Reg, newdata = test_set)

#Random Forest
RF <- train(Survived~poly(Age, 2)+Sex+Pclass+poly(FamGrpSize, 2)+poly(LadyCount, 3)+poly(KidsCount, 2)+poly(GrownManCount, 2)+poly(ElderlyCount, 2), data=train_set, trControl=train_control, method='rf')
test_fit <- predict(RF, newdata = test_set)
 
submission = data.frame('PassengerId'=test_set$PassengerId, 'Survived'=test_fit)
write.csv(submission, 'submission.csv', row.names = F)
# 






