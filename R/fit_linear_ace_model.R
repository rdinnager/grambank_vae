library(tidyverse)
library(fibre)
library(phyf)
library(conflicted)

####### Load Data ########################
gb_pf <- read_rds("data/gb_pf.rds")
grambank <- read_tsv("data/GB_cropped_for_missing.tsv")
## reshape to a matrix with GB parameters in columns, languages in rows
## and replace ? with NA
# grambank <- grambank %>%
#   select(Language_ID, Parameter_ID, Value) %>%
#   pivot_wider(names_from = Parameter_ID, values_from = Value) %>%
#   mutate(across(-Language_ID, .fns = ~ ifelse(.x == "?", NA_integer_, as.integer(.x))))

lang_ids <- grambank$Language_ID

grambank_pf <- gb_pf %>%
  left_join(grambank, by = c("label" = "Language_ID")) %>%
  mutate(gb_codes = as.matrix(pick(starts_with("GB"))))

form <- paste0(paste(colnames(grambank_pf$gb_codes), collapse = " + "),
               " ~ bre_brownian(phlo)")

gb_fit <- fibre(as.formula(form),
                data = grambank_pf,
                family = "binomial",
                verbose = 2)

gb_fit <- fibre(as.formula(form),
                data = grambank_pf,
                verbose = 2,
                engine_options = list(control.family = list(hyper = list(hyper = list(prec = list(prior = "pc.prec", initial = 4, fixed = TRUE))))))

rates <- gb_fit$random$phlo %>%
  select(ID, mean) %>%
  separate(ID, c("var", "node"), ":") %>%
  pivot_wider(names_from = var, values_from = mean) %>%
  select(node, starts_with("GB"))

ace_preds <- predict(gb_fit)
names(ace_preds) <- paste0(names(ace_preds), "_pred")
ace_means <-  map(ace_preds, ".pred_mean") %>%
  as_tibble()

gb_w_preds <- grambank_pf %>%
  bind_cols(ace_means) %>%
  mutate(time = pf_flow_sum(phlo))

write_rds(gb_w_preds, "data/gb_w_preds_linear_3dim.rds")

gb_edge_trajs <- pf_ends(gb_w_preds$phlo) %>%
  mutate(isna = is.na(end),
         end = ifelse(isna, start, end),
         start = ifelse(isna, NA, start)) %>%
  left_join(gb_w_preds %>%
              select(start = label,
                     ends_with("_pred")) %>%
              rename_with(~ paste0(.x, "_start"),
                          ends_with("_pred"))) %>%
  left_join(gb_w_preds %>%
              select(end = label,
                     ends_with("_pred")) %>%
              rename_with(~ paste0(.x, "_end"),
                          ends_with("_pred"))) %>%
  left_join(gb_w_preds %>%
              select(start = label,
                     start_time = time)) %>%
  left_join(gb_w_preds %>%
              select(end = label,
                     end_time = time)) %>%
  mutate(start_time = ifelse(isna, 0, start_time),
         across(ends_with("_start"), ~ ifelse(is.na(.x), 0, .x)))

write_rds(gb_edge_trajs, "data/gb_edge_trajs_linear_3dim.rds")

gb_rates <- gb_edge_trajs %>%
  mutate(vecs = pick(ends_with("_end")) - pick(ends_with("_start")),
         time_len = end_time - start_time) %>%
  mutate(vecs = vecs / time_len) %>%
  select(label = end, vecs) %>%
  unnest(vecs)

colnames(gb_rates) <- gsub("_pred_end", "", colnames(gb_rates))

write_rds(gb_pf, "data/gb_pf_linear_3dim.rds")
write_csv(gb_rates, "data/init_rates_linear_3dim.csv")

ggplot(gb_w_preds %>% dplyr::filter(is_tip), aes(as.factor(GB020), GB020_pred)) +
  geom_violin()

ggplot(gb_w_preds %>% dplyr::filter(is_tip), aes(as.factor(GB022), GB022_pred)) +
  geom_violin()
