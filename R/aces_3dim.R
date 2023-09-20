library(fibre)
library(phyf)
library(tidyverse)
library(ape)

gb_z <- read_csv("data/grambank_vae_latent_codes_means_only_v2_3dim.csv")
gb_tree <- read.tree("data/EDGE_pruned_tree.tree")

gb_tree <- drop.tip(gb_tree, which(!gb_tree$tip.label %in% gb_z$Language_ID))

gb_pf <- pf_as_pf(gb_tree) %>%
  left_join(gb_z, by = c("label" = "Language_ID"))

ace_fit <- fibre(latent_1 +
                   latent_2 +
                   latent_3 ~
                   bre_brownian(phlo),
                 data = gb_pf,
                 verbose = 2,
                 engine_options = list(control.family = list(hyper = list(hyper = list(prec = list(prior = "pc.prec", initial = 4, fixed = TRUE))))))

rates <- ace_fit$random$phlo %>%
  select(ID, mean) %>%
  separate(ID, c("var", "node"), ":") %>%
  pivot_wider(names_from = var, values_from = mean) %>%
  select(node,
         latent_1,
         latent_2,
         latent_3)

ace_preds <- predict(ace_fit)
names(ace_preds) <- paste0(names(ace_preds), "_pred")
ace_means <-  map(ace_preds, ".pred_mean") %>%
  as_tibble()

gb_w_preds <- gb_pf %>%
  bind_cols(ace_means) %>%
  mutate(time = pf_flow_sum(phlo))

write_rds(gb_w_preds, "data/gb_w_preds_3dim.rds")

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

write_rds(gb_edge_trajs, "data/gb_edge_trajs_3dim.rds")

ggplot(gb_edge_trajs,
       aes(x = latent_1_pred_start, y = latent_2_pred_start)) +
  geom_segment(aes(xend = latent_1_pred_end, yend = latent_2_pred_end)) +
  coord_equal() +
  theme_minimal() +
  geom_point(aes(latent_1_pred, latent_2_pred), data = gb_w_preds %>% select(-phlo) %>% as_tibble(),
             colour = "red", size = 0.1)

gb_rates <- gb_edge_trajs %>%
  mutate(vecs = pick(ends_with("_end")) - pick(ends_with("_start")),
         time_len = end_time - start_time) %>%
  mutate(vecs = vecs / time_len) %>%
  select(label = end, vecs) %>%
  unnest(vecs)

colnames(gb_rates) <- gsub("_pred_end", "", colnames(gb_rates))

write_rds(gb_pf, "data/gb_pf_3dim.rds")
write_csv(gb_rates, "data/init_rates_3dim.csv")


########### chop up edges by their raw length in latent space ###########
## we cut up the edges so that each has a number of segments proportional to their
## linear length and then interpolate the prediction as well
## This will form the starting values for our manifold evolution model

gb_edge_trajs <- gb_edge_trajs %>%
  mutate(diff = pick(ends_with("_end")) - pick(ends_with("_start"))) %>%
  rowwise() %>%
  mutate(dist = sqrt(sum(diff^2))) %>%
  drop_na(start, end) %>%
  rowwise() %>%
  mutate(props = list(seq(0, dist, by = 0.05)[-1] / dist)) %>%
  ungroup()

edges <- gb_edge_trajs %>%
  select(end, props) %>%
  unnest(props) %>%
  left_join(tibble(end = pf_edge_names(gb_w_preds$phlo),
                   len = pf_mean_edge_features(gb_w_preds))) %>%
  mutate(positions = len * props) %>%
  filter(positions != len)

write_csv(edges, "data/chopped_edges_big.csv")

####### convert to node numbers for use in torch #############
gb_edges <- pf_ends(gb_w_preds$phlo) %>%
  left_join(gb_w_preds %>%
              select(start = label) %>%
              mutate(start_node = 1:n())) %>%
  left_join(gb_w_preds %>%
              select(end = label) %>%
              mutate(end_node = 1:n()))

gb_aces <- gb_w_preds %>%
  select(ends_with("_pred")) %>%
  mutate(node = 1:n())

