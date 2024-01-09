# create recipe
dhi_recipe <- recipe(diabetes ~ ., data = dhi_training)

# create glm model
dhi_glm <- logistic_reg() %>%
  set_engine("glm")

# create workflow
dhi_workflow <- workflow() %>%
  add_recipe(dhi_recipe) %>%
  add_model(dhi_glm)

# fit model
dhi_fit <- fit(dhi_workflow, data = dhi_training)

# predict class and prob on validation set
dhi_pred_class <- predict(dhi_fit, dhi_validation) 
dhi_pred_prob <- predict(dhi_fit, dhi_validation, type = "prob")

# create predictions dataframe
dhi_predictions <- dhi_validation %>%
  select(diabetes) %>%
  bind_cols(dhi_pred_class) %>%
  bind_cols(dhi_pred_prob)

# # create yardstick metrics set with roc_auc, and accuracy
# dhi_metrics_set <- metric_set(roc_auc, accuracy)

# create metrics set with all metrics
dhi_metrics_set <- metric_set(roc_auc, pr_auc, accuracy, f_meas, mcc, sens, spec, ppv, npv)

# # calculate metrics
# dhi_metrics <- dhi_predictions %>%
#   dhi_metrics_set(truth = diabetes, estimate = .pred_class, .pred_1, event_level = "second")

# calculate all metrics
dhi_metrics <- dhi_predictions %>%
  dhi_metrics_set(truth = diabetes, estimate = .pred_class, .pred_1, event_level = "second")

# print metrics
dhi_metrics

# confusion matrix
cm <- dhi_predictions %>%
  conf_mat(diabetes, .pred_class)

cm

summary(cm)


# plot ROC curve
dhi_predictions %>%
  roc_curve(diabetes, .pred_1, event_level = "second") %>%
  autoplot() +
  theme_minimal() +
  theme(legend.position = "none") +
  ggtitle("ROC Curve") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank()) +
  theme(axis.text = element_text(size = 8)) +
  theme(text = element_text(size = 8)) +
  theme(plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm"))

# plot PR curve
dhi_predictions %>%
  pr_curve(diabetes, .pred_1, event_level = "second") %>%
  autoplot() +
  theme_minimal() +
  theme(legend.position = "none") +
  ggtitle("PR Curve") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank()) +
  theme(axis.text = element_text(size = 8)) +
  theme(text = element_text(size = 8)) +
  theme(plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm"))

# plot confusion matrix
dhi_predictions %>%
  conf_mat(diabetes, .pred_class) %>%
  autoplot() +
  theme_minimal() +
  theme(legend.position = "none") +
  ggtitle("Confusion Matrix") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank()) +
  theme(axis.text = element_text(size = 8)) +
  theme(text = element_text(size = 8)) +
  theme(plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm"))


# transform metrics dataframe
dhi_metrics_t <- as.data.frame(t(dhi_metrics))
colnames(dhi_metrics_t) <- dhi_metrics$.metric
dhi_metrics_t <- dhi_metrics_t[3,]
dhi_metrics_t <- dhi_metrics_t %>% mutate_if(is.character, as.numeric)
dhi_metrics <- dhi_metrics_t
remove(dhi_metrics_t)


# store results in dhi_results
dhi_results <- dhi_results %>%
  add_row(model = "glm",
          accuracy = dhi_metrics$accuracy,
          auc = dhi_metrics$roc_auc,
          pr_auc = dhi_metrics$pr_auc,
          f1 = dhi_metrics$f_meas,
          mcc = dhi_metrics$mcc,
          sensitivity = dhi_metrics$sens,
          specificity = dhi_metrics$spec,
          ppv = dhi_metrics$ppv,
          npv = dhi_metrics$npv)

# save model
save(dhi_fit, file = "models/dhi_fit_glm.RData")