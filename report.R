library("tidyverse")

data <- read.csv("https://www.maths.dur.ac.uk/users/hailiang.du/assignment_data/heart_failure.csv")

#################################### DATA EXPLORATION ##########################
df <- data %>%
  mutate(
    anaemia = factor(anaemia, levels = c(0, 1), labels = c("No", "Yes")),
    diabetes = factor(diabetes, levels = c(0, 1), labels = c("No", "Yes")),
    high_blood_pressure = factor(high_blood_pressure, levels = c(0, 1), labels = c("No", "Yes")),
    sex = factor(sex, levels = c(0, 1), labels = c("Female", "Male")),
    smoking = factor(smoking, levels = c(0, 1), labels = c("No", "Yes")),
    fatal_mi = factor(fatal_mi, levels = c(0, 1), labels = c("No", "Yes")),
    log_cpk = log1p(creatinine_phosphokinase)
  )

library("skimr")
skim(df)

num_vars <- c(
  "age",
  "creatinine_phosphokinase",
  "ejection_fraction",
  "platelets",
  "serum_creatinine",
  "serum_sodium"
)

df %>%
  select(all_of(num_vars)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30) +
  facet_wrap(~ variable, scales = "free") +
  labs(
    title = "Histograms of numerical predictors",
    x = "Value",
    y = "Count"
  ) +
  theme_minimal()

df %>%
  select(all_of(num_vars), fatal_mi) %>%
  pivot_longer(cols = all_of(num_vars), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = fatal_mi, y = value)) +
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free") +
  labs(
    title = "Boxplots of numerical predictors by fatal_mi outcome",
    x = "Fatal MI",
    y = "Value"
  ) +
  theme_minimal()

cat_vars <- c(
  "anaemia",
  "diabetes",
  "high_blood_pressure",
  "sex",
  "smoking"
)

df %>%
  select(all_of(cat_vars), fatal_mi) %>%
  pivot_longer(cols = all_of(cat_vars), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value, fill = fatal_mi)) +
  geom_bar(position = "stack") +
  facet_wrap(~ variable, scales = "free_x") +
  labs(
    title = "Bar plots of categorical predictors by fatal_mi outcome",
    x = "Category",
    y = "Count",
    fill = "Fatal MI"
  ) +
  theme_minimal()


library(GGally)
ggpairs(df[, num_vars])

cor_matrix <- cor(df[, num_vars], use = "complete.obs")
print(cor_matrix)

cor_df <- as.data.frame(as.table(cor_matrix))

ggplot(cor_df, aes(x = Var1, y = Var2, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = round(Freq, 2)), size = 3) +
  labs(
    title = "Correlation Matrix of Numerical Predictors",
    x = "",
    y = "",
    fill = "Correlation"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#################################### MODEL FITTING ##############################
library(mlr3verse)
library(mlr3pipelines)

model_df <- df %>%
  select(-time, -creatinine_phosphokinase)

set.seed(1)

hf_task <- TaskClassif$new(
  id= "heart_failure",
  backend = model_df,
  target="fatal_mi",
  positive = "Yes"
  
)
hf_task

cv5 <- rsmp("cv", folds =5)
cv5$instantiate(hf_task)

lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
lrn_tree <- lrn("classif.rpart", predict_type="prob")
lrn_rf <- lrn("classif.ranger", predict_type="prob")
lrn_xgb <- lrn("classif.xgboost", predict_type = "prob")
pl_xgb <- po("encode") %>>%
  po(lrn_xgb)

res <- benchmark(
  benchmark_grid(
    tasks = hf_task,
    learners = list(lrn_log_reg,lrn_tree,lrn_rf,pl_xgb),
    resamplings = cv5
  ),
  store_models = TRUE
)


res_performance <- res$aggregate(list(
  msr("classif.acc"),
  msr("classif.ce"),
  msr("classif.auc"),
  msr("classif.fpr"),
  msr("classif.fnr")
))

modified_perf <- res_performance[,7:11]
modified_perf$model <- c("Logistic regression", "CART", "Random Forest", "XGBoost")



#################################### MODEL TUNING##############################
library(mlr3tuning)
library(paradox)

set.seed(1)

rf_base <- lrn(
  "classif.ranger",
  predict_type = "prob"
)

search_space <- ps(
  mtry = p_int(lower=1, upper = ncol(model_df)-1),
  min.node.size = p_int(lower=1, upper =20),
  num.trees = p_int(lower =100, upper =800)
)

inner_cv <- rsmp("cv", folds=3)
outer_cv <- rsmp("cv", folds=5)

at_rf <- auto_tuner(
  tuner=tnr("random_search"),
  learner = rf_base,
  resampling = inner_cv,
  measure = msr("classif.auc"),
  search_space = search_space,
  terminator = trm("evals", n_evals =20)
)

final_res <- benchmark(
  benchmark_grid(
    tasks = hf_task,
    learners = list(lrn_log_reg, at_rf),
    resamplings = outer_cv
  ),
  store_models = TRUE
)


final_table <- final_res$aggregate(list(
  msr("classif.acc"),
  msr("classif.ce"),
  msr("classif.auc"),
  msr("classif.fpr"),
  msr("classif.fnr")
))

final_table

modified_final_table <- final_table[,7:11]
modified_final_table$model <- c("Logistic regression", "Random Forest")

pred_rf <- final_res$resample_result(2)$prediction()
pred_rf

table(
  Truth = pred_rf$truth, 
  Predicted = pred_rf$response
)

pred_rf_new <- pred_rf$clone(deep = TRUE)
pred_rf_new$set_threshold(0.3)

table(
  Truth = pred_rf_new$truth,
  Predicted = pred_rf_new$response
)

pred_rf_new$score(list(
  msr("classif.acc"),
  msr("classif.auc"),
  msr("classif.fpr"),
  msr("classif.fnr")
))






