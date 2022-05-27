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

clean_pokemon <- clean_names(pokemon)

# question 1
# Filter entire dataset
filtered_pokemon <- filter(clean_pokemon, type_1 == "Bug" | type_1 == "Fire" 
                           | type_1 == "Grass" | type_1== "Normal" | 
                             type_1 == "Water" | type_1 == "Psychic")

# Convert type_1 and legendary to factors
filtered_pokemon$type_1 <- factor(filtered_pokemon$type_1)
filtered_pokemon$legendary <- factor(filtered_pokemon$legendary)


set.seed(777)

# Initial split
pokemon_split <- initial_split(filtered_pokemon, strata = type_1, prop = 0.8)

pokemon_train <- training(pokemon_split)
pokemon_test <- testing(pokemon_split)

# Folding Training Data
set.seed(777)
pokemon_fold <- vfold_cv(pokemon_train, v = 5, strata = type_1)

# Recipe
pokemon_recipe <- recipe(type_1 ~ legendary + generation + sp_atk + attack 
                         + speed + defense + hp + sp_def, data = pokemon_train) %>%
  step_dummy(all_nominal_predictors()) %>%   # creates dummy variables
  step_normalize(all_predictors())    # Centers and Scales all variables


  # Q3 

# general decision tree specification
tree_spec <- decision_tree() %>%
  set_engine("rpart")

# classificaition decision tree engine/model
class_tree_spec <- tree_spec %>%
  set_mode("classification")

# Workflow tuning cost complexity
class_tree_wf <- workflow() %>%
  add_model(class_tree_spec %>% set_args(cost_complexity = tune())) %>%
  add_recipe(pokemon_recipe)

# setup grid
set.seed(777)

param_grid <- grid_regular(cost_complexity(range = c(-3, -1)), levels = 10)

tune_res_tree <- tune_grid(
  class_tree_wf, 
  resamples = pokemon_fold, 
  grid = param_grid, 
  metrics = metric_set(roc_auc)
)

  # Q5
#fit model
best_model <- select_best(tune_res_tree)

class_tree_final <- finalize_workflow(class_tree_wf, best_model)

class_tree_final_fit <- fit(class_tree_final, data = pokemon_train)

# Random forest model and workflow
rf_mod <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(pokemon_recipe) %>%
  add_model(rf_mod)

# grid
param_grid <- grid_regular(mtry(range = c(2, 6)), 
                           trees(range = c(2, 5)), 
                           min_n(), levels = c(8,8,8)) # I am not sure what a good range for min_n would be

  # Q6
tune_res_rf <- tune_grid(
  rf_wf, 
  resamples = pokemon_fold, 
  grid = param_grid, 
  metrics = metric_set(roc_auc)
)

  # Q9
boost_spec <- boost_tree(trees = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(pokemon_recipe) %>%
  add_model(boost_spec)

# grid
param_grid <- grid_regular(trees(range = c(10, 2000)), levels = 10)

tune_res_boost <- tune_grid(
  boost_wf, 
  resamples = pokemon_fold, 
  grid = param_grid, 
  metrics = metric_set(roc_auc)
)



# SAVING ALL THE FITS
save(tune_res_tree,tune_res_rf,tune_res_boost, file = "tunedmodels.rda")
rm(tune_res_tree,tune_res_rf,tune_res_boost)
load(file = "tunedmodels.rda")
