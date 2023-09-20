library(tidyverse)
library(torch)
library(patchwork)

grambank <- read_tsv("data/GB_cropped_for_missing.tsv")
preds <- read_csv("data/grambank_vae_latent_codes_means_only_v2_3dim.csv")
lang_dat <- read_csv("data/cldf/languages.csv")

options(torch.serialization_version = 2)
vae_3 <- torch_load("data/grambank_vae_v2_3dim.to")
vae_3 <- vae_3$cuda()

gb_tip_ord <- read_rds("data/gb_tips_order.rds")
gb_tip_num <- gb_tip_ord$y
names(gb_tip_num) <- gb_tip_ord$label
gb_ord <- read_rds("data/GB_order.rds")

pred_mat <- preds %>%
  select(starts_with("latent_")) %>%
  as.matrix() %>%
  torch_tensor(device = "cuda")

gb_preds <- vae_3$decoder(pred_mat)$cpu()

gb_preds <- gb_preds %>%
  as.matrix() %>%
  as.data.frame()

colnames(gb_preds) <- colnames(grambank[-1])

grambank_full <- grambank %>%
  pivot_longer(-Language_ID, names_to = "feature", values_to = "value") %>%
  left_join(gb_preds %>%
              mutate(Language_ID = preds$Language_ID) %>%
              pivot_longer(-Language_ID, names_to = "feature", values_to = "vae_prediction")) %>%
  mutate(feature_num = gb_ord[feature],
         tip_num = gb_tip_num[Language_ID])

na_langs <- grambank_full %>%
  group_by(Language_ID) %>%
  summarise(missing = any(is.na(tip_num))) %>%
  ungroup() %>%
  filter(missing) %>%
  mutate(tip_num = max(grambank_full$tip_num, na.rm = TRUE) + 1:n())

grambank_full <- grambank_full %>%
  left_join(na_langs %>% select(Language_ID, tip_num2 = tip_num)) %>%
  mutate(tip_num = ifelse(is.na(tip_num), tip_num2, tip_num))

true_plot <- ggplot(grambank_full, aes(tip_num, feature_num)) +
  geom_tile(aes(fill = value, colour = value)) +
  scale_fill_continuous(name = "", na.value = "pink") +
  scale_colour_continuous(name = "", na.value = "pink") +
  ylab("Grambank Feature") +
  xlab("") +
  ggtitle("'Ground Truth' Data") +
  theme_minimal() +
  theme(axis.text = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank(),
        legend.position = "left")

#true_plot

pred_plot <- ggplot(grambank_full, aes(tip_num, feature_num)) +
  geom_tile(aes(fill = vae_prediction, colour = vae_prediction)) +
  scale_fill_continuous(name = "", na.value = "pink") +
  scale_colour_continuous(name = "", na.value = "pink") +
  ylab("Grambank Feature") +
  xlab("Language") +
  ggtitle("VAE Reconstruction") +
  theme_minimal() +
  theme(axis.text = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank(),
        legend.position = "left")

#pred_plot

true_plot + pred_plot + plot_layout(nrow = 2,
                                    guides = "collect")

grambank_full <- grambank_full %>%
  mutate(square_diff = (value - vae_prediction) ^ 2,
         correct = as.logical(1 - (value - round(vae_prediction))))

error_plot <- ggplot(grambank_full, aes(tip_num, feature_num)) +
  geom_tile(aes(fill = square_diff, colour = square_diff)) +
  scale_fill_continuous(name = "", na.value = "pink") +
  scale_colour_continuous(name = "", na.value = "pink") +
  ylab("Grambank Feature") +
  xlab("Language") +
  ggtitle("Squared Error") +
  theme_minimal() +
  theme(axis.text = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank(),
        legend.position = "left")

error_plot

sum(grambank_full$correct) / nrow(grambank_full)

preds <- preds %>%
  left_join(lang_dat,
            by = c("Language_ID" = "Glottocode"))

library(ggpointdensity)

ggplot(preds, aes(latent_1, latent_2)) +
  geom_pointdensity() +
  scale_color_viridis_c() +
  ylab("VAE Latent Variable 2") +
  xlab("VAE Latent Variable 1") +
  coord_equal() +
  theme_minimal() +
  theme(legend.position = 'none')

ggplot(preds, aes(latent_3, latent_2)) +
  geom_pointdensity() +
  scale_color_viridis_c() +
  ylab("VAE Latent Variable 2") +
  xlab("VAE Latent Variable 3") +
  coord_equal() +
  theme_minimal() +
  theme(legend.position = 'none')
