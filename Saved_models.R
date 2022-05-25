library(yardstick)
library(tidyverse)
library(tidymodels)
library(ISLR)
library(rpart.plot)
library(vip)
library(janitor)
library(randomForest)
library(xgboost)
library(corrr)
library(corrplot)

tidymodels_prefer()

pokemon <- read_csv("Pokemon.csv")

set.seed(777)

titanic$survived =  factor(titanic$survived, levels = c("Yes", "No")) 
# Note can use parse_factor() in order to give a warning when there is a value not in the set

titanic$pclass =  factor(titanic$pclass)


class(titanic$survived)
class(titanic$pclass)



# Q1 SPLIT
titanic_split <- initial_split(titanic, strata = survived, prop = 0.8)

titanic_train <- training(titanic_split)
titanic_test <- testing(titanic_split)


titanic_recipe <- recipe(survived ~ pclass + sex + age + sib_sp + parch + fare, data = titanic_train) %>% 
  step_impute_linear(age, impute_with = imp_vars(sib_sp)) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_interact(~ starts_with("sex"):age + age:fare)



# Q2 
# Creating tuned recipe for use later in workflows

titanic_tuned_rec <- titanic_recipe %>%
  step_poly(pclass, sex,age , sib_sp, parch, fare, degree = tune()) # polynomial regression

# rsample object of the cross-validation resamples

titanic_folds <- vfold_cv(titanic_train, v = 10) # Here ISLR uses v instead of k but they are interchangeable, common values include 5 or 10
titanic_folds # creates the k-Fold data set

# tibble with hyperparameter values we are exploring
degree_grid <- grid_regular(degree(range = c(1, 10)), levels = 10)
degree_grid 

# Q4

# Logistic regression
# We will use the recipe created in Question 2 to create workflows

# Logistic regression
log_reg <- logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")

titanic_log_wf <- workflow() %>%  # log workflow
  add_recipe(titanic_recipe) %>%
  add_model(log_reg)

# LDA
# Linear discriminant analysis
lda_mod <- discrim_linear() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

titanic_lda_wf <- workflow() %>%  # lda workflow
  add_recipe(titanic_recipe) %>%
  add_model(lda_mod)

# QDA
# Quadratic discriminant analysis
qda_mod <- discrim_quad() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

titanic_qda_wf <- workflow() %>%  # qda workflow
  add_model(qda_mod) %>% 
  add_recipe(titanic_recipe)

# Q5 FIT MODELS

# LOG REG
fit_log <- fit_resamples(  
  titanic_log_wf,
  titanic_folds,
  control = control_resamples(verbose = TRUE))


# fit_resamples fits computes performance metrics across our specified resamples
# the control option prints out progress which helps with models that take a long time to fit


# LDA
fit_lda <- fit_resamples(  
  titanic_lda_wf,
  titanic_folds,
  control = control_resamples(verbose = TRUE))



# QDA

fit_qda <- fit_resamples(  
  titanic_qda_wf,
  titanic_folds,
  control = control_resamples(verbose = TRUE))



# SAVING ALL THE FITS
save(fit_log,fit_lda, fit_qda,file = "fittedmodels.rda")
rm(fit_lda,fit_log,fit_qda)
load(file = "fittedmodels.rda")