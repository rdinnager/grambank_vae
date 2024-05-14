library(tidyverse)
library(rgl)
library(torch)
library(patchwork)
library(Polychrome)
library(wesanderson)

gb_w_preds <- read_rds("data/gb_w_preds_3dim.rds")
z_tree_df <- read_rds("data/gb_vae_aces_3dim_noise_schedule.rds")
gb_edge_trajs <- read_rds("data/gb_edge_trajs_3dim.rds")

z_tree_df <- z_tree_df %>%
  rowwise() %>%
  mutate(ace_latent_1 = tail(z_seqs$latent_1, 1),
         ace_latent_2 = tail(z_seqs$latent_2, 1),
         ace_latent_3 = tail(z_seqs$latent_3, 1))

ace_dat <- gb_w_preds %>%
  filter(!is_tip) %>%
  select(label, latent_1 = latent_1_pred,
         latent_2 = latent_2_pred,
         latent_3 = latent_3_pred) %>%
  mutate(method = "linear") %>%
  bind_rows(z_tree_df %>%
              filter(!is_tip) %>%
              select(label = edge,
                     latent_1 = ace_latent_1,
                     latent_2 = ace_latent_2,
                     latent_3 = ace_latent_3) %>%
              mutate(method = "manifold"))

tree_segs_df <- z_tree_df %>%
  select(label = edge, z_seqs) %>%
  unnest(z_seqs)

##### compare aces ##########
ace_comp_1 <- pivot_wider(ace_dat %>%
                         select(label, latent_1, method),
                         values_from = latent_1, names_from = method)
ace_comp_2 <- pivot_wider(ace_dat %>%
                         select(label, latent_2, method),
                         values_from = latent_2, names_from = method)
ace_comp_3 <- pivot_wider(ace_dat %>%
                         select(label, latent_3, method),
                         values_from = latent_3, names_from = method)

########### make panels ##############

mid_1 <- ggplot(ace_comp_1, aes(linear, manifold)) +
  geom_point(alpha = 0.5, size = 2) +
  geom_abline(intercept = 0, slope = 1, colour = "grey") +
  theme_minimal()
mid_2 <- ggplot(ace_comp_2, aes(linear, manifold)) +
  geom_point(alpha = 0.5, size = 2) +
  geom_abline(intercept = 0, slope = 1, colour = "grey") +
  theme_minimal()
mid_3 <- ggplot(ace_comp_3, aes(linear, manifold)) +
  geom_point(alpha = 0.5, size = 2) +
  geom_abline(intercept = 0, slope = 1, colour = "grey") +
  theme_minimal()

upper_1_2 <- ggplot(gb_edge_trajs,
                  aes(x = latent_1_pred_start, y = latent_2_pred_start)) +
  geom_segment(aes(xend = latent_1_pred_end, yend = latent_2_pred_end)) +
  coord_equal() +
  theme_minimal() +
  geom_point(aes(latent_1_pred, latent_2_pred), data = gb_w_preds %>% select(-phlo) %>% filter(is_tip) %>% as_tibble(),
             colour = "red", size = 0.1) +
  ylab("Latent Variable 2") +
  xlab("Latent Variable 1")

lower_1_2 <- ggplot(tree_segs_df, aes(latent_1, latent_2)) +
  geom_path(aes(group = label)) +
  coord_equal() +
  theme_minimal() +
  geom_point(aes(ace_latent_1, ace_latent_2),
             data = z_tree_df %>% filter(is_tip),
             colour = "red", size = 0.1) +
  ylab("Latent Variable 2") +
  xlab("Latent Variable 1")

upper_1_3 <- ggplot(gb_edge_trajs,
                  aes(x = latent_1_pred_start, y = latent_3_pred_start)) +
  geom_segment(aes(xend = latent_1_pred_end, yend = latent_3_pred_end)) +
  coord_equal() +
  theme_minimal() +
  geom_point(aes(latent_1_pred, latent_3_pred), data = gb_w_preds %>% select(-phlo) %>% filter(is_tip) %>% as_tibble(),
             colour = "red", size = 0.1) +
  ylab("Latent Variable 3") +
  xlab("Latent Variable 1")

lower_1_3 <- ggplot(tree_segs_df, aes(latent_1, latent_3)) +
  geom_path(aes(group = label)) +
  coord_equal() +
  theme_minimal() +
  geom_point(aes(ace_latent_1, ace_latent_3),
             data = z_tree_df %>% filter(is_tip),
             colour = "red", size = 0.1) +
  ylab("Latent Variable 3") +
  xlab("Latent Variable 1")

upper_2_3 <- ggplot(gb_edge_trajs,
                  aes(x = latent_2_pred_start, y = latent_3_pred_start)) +
  geom_segment(aes(xend = latent_2_pred_end, yend = latent_3_pred_end)) +
  coord_equal() +
  theme_minimal() +
  geom_point(aes(latent_2_pred, latent_3_pred), data = gb_w_preds %>% select(-phlo) %>% filter(is_tip) %>% as_tibble(),
             colour = "red", size = 0.1) +
  ylab("Latent Variable 3") +
  xlab("Latent Variable 2")

lower_2_3 <- ggplot(tree_segs_df, aes(latent_2, latent_3)) +
  geom_path(aes(group = label)) +
  coord_equal() +
  theme_minimal() +
  geom_point(aes(ace_latent_2, ace_latent_3),
             data = z_tree_df %>% filter(is_tip),
             colour = "red", size = 0.1) +
  ylab("Latent Variable 3") +
  xlab("Latent Variable 2")

p <- mid_1 + lower_1_2 + lower_1_3 +
  upper_1_2 + mid_2 + lower_2_3 +
  upper_1_3 + upper_2_3 + mid_3 +
  plot_layout(nrow = 3)

p

############ plot by family ################

set.seed(1234564535)

fams <- z_tree_df %>%
  filter(is_tip) %>%
  group_by(Family_name) %>%
  mutate(count = n()) %>%
  mutate(Family_name = ifelse(count < 10, NA, Family_name))

fam_pal <- createPalette(n_distinct(na.omit(fams$Family_name)),
                         wes_palettes$FantasticFox1[5])

names(fam_pal) <- unique(na.omit(fams$Family_name))

ggplot(tree_segs_df, aes(latent_1, latent_2)) +
  geom_path(aes(group = label)) +
  coord_equal() +
  theme_minimal() +
  geom_point(aes(ace_latent_1, ace_latent_2, color = Family_name),
             data = fams, size = 3) +
  scale_color_manual(values = fam_pal) +
  ylab("Latent Variable 2") +
  xlab("Latent Variable 1") +
  theme(legend.position = "right")

z_tree_df <- z_tree_df %>%
  select(-col) %>%
  left_join(tibble(Family_name = names(fam_pal),
                   col = fam_pal))

rgl::open3d()
for(i in 1:nrow(z_tree_df)) {
  rgl::lines3d(z_tree_df$z_seqs[[i]]$latent_1,
               z_tree_df$z_seqs[[i]]$latent_2,
               z_tree_df$z_seqs[[i]]$latent_3,
               col = "black",
               lwd = 1)
}
rgl::points3d(z_tree_df$true_latent_1,
              z_tree_df$true_latent_2,
              z_tree_df$true_latent_3,
              col = z_tree_df$col,
              size = 10)


####### rgl animation #########

tree_segs_rgl <- z_tree_df %>%
  select(label = edge, z_seqs) %>%
  left_join(gb_edge_trajs %>% select(label = end,
                                     start_time, end_time)) %>%
  rowwise() %>%
  mutate(z_seqs = list(bind_rows(z_seqs %>%
                                   mutate(time = seq(start_time, end_time, length.out = 100)),
                                          tibble(latent_1 = NA,
                                                 latent_2 = NA,
                                                 latent_3 = NA)
                       ))) %>%
  unnest(z_seqs)

rgl::lines3d(tree_segs_rgl$latent_1,
              tree_segs_rgl$latent_2,
              tree_segs_rgl$latent_3,
              lwd = 1)

rgl::clear3d()
Sys.sleep(5)
for(i in seq(from = 0, to = max(tree_segs_rgl$time, na.rm = TRUE),
             length.out = 1000)) {
  rgl::lines3d(tree_segs_rgl$latent_1[tree_segs_rgl$time <= i],
              tree_segs_rgl$latent_2[tree_segs_rgl$time <= i],
              tree_segs_rgl$latent_3[tree_segs_rgl$time <= i],
              lwd = 1)
}

