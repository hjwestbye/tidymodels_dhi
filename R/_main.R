# load libraries
source("R/libraries.R")

# load data
df <- read_csv("data/diabetes_binary_health_indicators_BRFSS2015.csv")

# clean data with janitor
df <- df %>%
  clean_names()

# rename outcome column
df <- df %>%
  rename(diabetes = diabetes_binary)

glimpse(df)

# outcome column to factor
df <- df %>%
  mutate(diabetes = as.factor(diabetes))

# split data into training, validation and test sets
set.seed(6987)
df_split <- initial_validation_split(df, strata = diabetes)
dhi_training <- training(df_split)
dhi_validation <- validation(df_split)
dhi_test <- testing(df_split)

# save data
save(dhi_training, dhi_validation, dhi_test, file = "data/dhi_split.RData")

# load data
load("data/dhi_split.RData")

# Check observations in each set
nrow(dhi_training)
nrow(dhi_validation)
nrow(dhi_test)

# Check correlation in training data
dhi_training %>%
  select(-diabetes) %>%
  cor(use = "pairwise.complete.obs") %>%
  ggcorrplot(type = "lower", lab = TRUE, lab_size = 2) + #plot lower triangle
  theme_minimal() + #make background white
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) + #rotate x-axis labels
  ggtitle("Correlation in Training Data") + #add title
  theme(plot.title = element_text(hjust = 0.5)) + #center title
  theme(axis.title.x = element_blank(), axis.title.y = element_blank()) + #remove axis titles
  scale_fill_gradient(low = "white", high = "steelblue") + #change color gradient
  theme(legend.position = "none") #remove legend

# create dataframe to store model results as a global variable
dhi_results <- tibble(model = character(),
                      accuracy = double(),
                      auc = double(),
                      mcc = double(),
                      pr_auc = double(),
                      f1 = double(),
                      sensitivity = double(),
                      specificity = double(),
                      ppv = double(),
                      npv = double())


source("R/fit_glm.R")

# save results dataframe
save(dhi_results, file = "results/dhi_results.RData")

