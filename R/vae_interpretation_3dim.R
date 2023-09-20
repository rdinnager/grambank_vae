library(tidyverse)
library(energy)
library(torch)

gb_z <- read_csv("data/grambank_vae_latent_codes_v2_3dim.csv")
z_df <- gb_z %>%
  select(Language_ID, latent_dim, latent_mean) %>%
  pivot_wider(names_from = latent_dim, values_from = latent_mean)

grambank <- read_tsv("data/GB_cropped_for_missing.tsv")

grambank <- z_df %>%
  select(Language_ID) %>%
  left_join(grambank)


latent <- z_df$latent_1
GB <- grambank$GB020
calc_nonlinear_corr <- function(latent, GB, test = FALSE, reps = 500) {
  com <- complete.cases(cbind(latent, GB))
  l <- latent[com]
  g <- GB[com]
  if(test) {
    return(energy::dcor.test(l, g, R = reps))
  } else {
    return(energy::dcor(l, g))
  }

}

combos <- expand_grid(latent = colnames(z_df %>% select(-Language_ID)),
                      GB = colnames(grambank %>% select(-Language_ID)))

all_corrs <- map(list_transpose(as.list(combos)),
                 ~ calc_nonlinear_corr(z_df %>% pull({.x[[1]]}),
                                       grambank %>% pull({.x[[2]]})),
                 .progress = TRUE)

assoc_df <- combos %>%
  mutate(dcor = unlist(all_corrs))

test_corrs <- map(list_transpose(as.list(combos)),
                 ~ calc_nonlinear_corr(z_df %>% pull({.x[[1]]}),
                                       grambank %>% pull({.x[[2]]}),
                                       test = TRUE),
                 .progress = TRUE)

assoc_df <- assoc_df %>%
  mutate(p = map_dbl(test_corrs, "p.value"))

write_csv(assoc_df, "data/latent_GB_assoc_3dim.csv")
