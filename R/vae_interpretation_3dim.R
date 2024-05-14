library(tidyverse)
library(energy)
library(torch)
library(LambertW)
library(conflicted)

conflicts_prefer(purrr::map)
conflicts_prefer(dplyr::select)

gb_z <- read_csv("data/grambank_vae_latent_codes_v2_3dim.csv")
z_df <- gb_z %>%
  select(Language_ID, latent_dim, latent_mean) %>%
  pivot_wider(names_from = latent_dim, values_from = latent_mean)

grambank <- read_tsv("data/GB_cropped_for_missing.tsv")

grambank <- z_df %>%
  select(Language_ID) %>%
  left_join(grambank)

cauch_1 <- fitdistr(z_df$latent_1, "cauchy")
cauch_2 <- fitdistr(z_df$latent_2, "cauchy")
cauch_3 <- fitdistr(z_df$latent_3, "cauchy")

curve(dcauchy(x, scale = 0.67), from = -4, to = 4)
curve(dcauchy(x, scale = 0.7), from = -4, to = 4, add = TRUE, col = "blue")
points(z_df$latent_1, rep(0.01, length(z_df$latent_1)), pch = 19, col = alpha("black", 0.05))

norms <- apply(z_df %>% select(-Language_ID) %>% as.matrix(),
               1, function(x) sqrt(sum(x^2)))
hist(norms, breaks = 30)
hist(apply(matrix(rnorm(6000), ncol = 3),
               1, function(x) sqrt(sum(x^2))), breaks = 30)

lognorm <- fitdistr(norms, "lognormal")
curve(dlnorm(x, 0.54, 0.49), 0, 4.5)
points(norms, rep(0.06, length(norms)), pch = 19, col = alpha("black", 0.05))

weibull <- fitdistr(norms, "weibull")
curve(dweibull(x, 2.49, 2.16), 0, 4.5)
points(norms, rep(0.06, length(norms)), pch = 19, col = alpha("black", 0.05))

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

all_corrs_lin <- map(list_transpose(as.list(combos)),
                 ~ cor(z_df %>% pull({.x[[1]]}),
                       grambank %>% pull({.x[[2]]}),
                       use = "complete.obs"),
                 .progress = TRUE)

assoc_df <- assoc_df %>%
  mutate(lin_corr = unlist(all_corrs_lin),
         abs_lin_corr = abs(lin_corr))

assoc_df <- assoc_df %>%
  mutate(corr_diff = dcor - abs_lin_corr)

write_csv(assoc_df, "data/latent_GB_assoc_3dim.csv")

assoc_df_red <- assoc_df %>%
  group_by(latent) %>%
  arrange(desc(dcor), .by_group = TRUE) %>%
  slice_head(n = 20)

ggplot(assoc_df_red, aes(dcor, GB)) +
  geom_bar(aes(fill = dcor), stat = 'identity') +
  facet_wrap(vars(latent), nrow = 1) +
  xlab("Non-Linear Association (Distance Correlation)") +
  scale_fill_viridis_c(option = "inferno", name = "Association\nStrength") +
  theme_minimal()

plot(jitter(grambank$GB133), z_df$latent_2)

##### sample the manifold ###############

vae_3 <- torch_load("data/grambank_vae_v2_3dim.to")
vae_3 <- vae_3$cuda()

centroids <- z_df %>%
  select(-Language_ID) %>%
  as.matrix() %>%
  torch_tensor(device = "cuda")

vars <- gb_z %>%
  select(Language_ID, latent_dim, latent_var) %>%
  pivot_wider(names_from = latent_dim, values_from = latent_var) %>%
  select(-Language_ID) %>%
  as.matrix() %>%
  torch_tensor(device = "cuda")

mahalanobis_squared_fun <- function(z1, z2, v2) {

  n_1 <- z1$size(1)
  segs <- z1$size(3)
  n_2 <- z2$size(1)
  dim <- z1$size(2)

  mahalanobis_squared <- function(z1, z2, v2) {

    expanded_1 = z1$unsqueeze(2)$expand(c(n_1, n_2, dim, segs))
    expanded_2 = z2$unsqueeze(1)$unsqueeze(-1)$expand(c(n_1, n_2, dim, segs))
    expanded_3 = v2$unsqueeze(1)$unsqueeze(-1)$expand(c(n_1, n_2, dim, segs))

    diff <- expanded_1 - expanded_2
    torch_sum((diff / expanded_3) * diff, 3)
  }

  mahalanobis_squared_tr <- jit_trace(mahalanobis_squared,
                                      z1, z2, v2)

  mahalanobis_squared <- function(z1, z2, v2) {

    n_1 <- z1$size(1)
    segs <- z1$size(3)
    n_2 <- z2$size(1)
    dim <- z1$size(2)

    expanded_1 = z1$unsqueeze(2)$expand(c(n_1, n_2, dim, segs))
    expanded_2 = z2$unsqueeze(1)$unsqueeze(-1)$expand(c(n_1, n_2, dim, segs))
    expanded_3 = v2$unsqueeze(1)$unsqueeze(-1)$expand(c(n_1, n_2, dim, segs))

    diff <- expanded_1 - expanded_2
    torch_sum((diff / expanded_3) * diff, 3)
  }

  list(mahalanobis_squared, mahalanobis_squared_tr)

}

## sample from the prior
batch_size <- 10000
samp_zs <- torch_randn(batch_size, 3, device = "cuda")$unsqueeze(-1)

mahalanobis_squared <- mahalanobis_squared_fun(samp_zs, centroids, vars)

get_metric_tensor <- function(z, centroids, vars, lambda = 1e-2, rho) {
  mh <- torch_exp(-(mahalanobis_squared[[1]](z, centroids, vars) / (rho^2)))
  ## really relying on broadcasting correctly here! Hope I got it right!
  1 / (((1 / vars)$unsqueeze(1)$unsqueeze(-1) * mh$unsqueeze(3))$sum(dim = 2) + lambda)
}

rho <- read_rds("data/estimated_rho_3dim.rds")
rho <- torch_tensor(rho / 5, device = "cuda")

samp_G <- get_metric_tensor(samp_zs, centroids, vars, rho = rho)
samp_W <- torch_prod(samp_G$squeeze(-1), dim = 2)$sqrt()

norm_probs <- mvtnorm::dmvnorm(as.matrix(samp_zs$squeeze(-1)$cpu()))
M <- 0.000000005
W <- as.numeric(samp_W$cpu()) / M*norm_probs

plot(W)

a <- runif(length(W))

new_samp <- which(a <= W)
plot(as.matrix(samp_zs$squeeze(-1)$cpu())[new_samp,1:2])
points(z_df$latent_2, z_df$latent_3, col = "red")

length(unique(new_samp))

get_sample_zs <- function(M = 30000) {

  #sz <- matrix(rcauchy(batch_size * 3, scale = 0.75), ncol = 3)
  dists <- rweibull(batch_size, 2.49, 2.16)
  ns <- matrix(rnorm(batch_size * 3), ncol = 3)
  norms <- apply(ns, 1, function(x) sqrt(sum(x^2)))
  ns <- ns / norms
  sz <- ns * dists

  #samp_zs <- torch_randn(batch_size, 3, device = "cuda")$unsqueeze(-1)
  samp_zs <- torch_tensor(sz, device = "cuda")$unsqueeze(-1)

  samp_G <- 1 / get_metric_tensor(samp_zs, centroids, vars, rho = rho)
  samp_W <- torch_prod(samp_G$squeeze(-1), dim = 2)$sqrt()
  ref_probs <- dweibull(dists, 2.49, 2.16)
  W <- as.numeric(samp_W$cpu()) / M*cauch_probs
  #plot(W)
  a <- runif(length(W))
  new_samp <- which(a <= W)
  as.matrix(samp_zs$squeeze(-1)$cpu())[new_samp, ]

}

z_samples <- map(1:1000, ~ get_sample_zs(), .progress = TRUE)
z_samples <- do.call(rbind, z_samples)

plot(z_samples[ , 1:2])
points(z_df$latent_1, z_df$latent_2, col = "red")

plot(z_samples[ , 2:3], pch = 19, col = alpha("black", 0.1))
points(z_df$latent_2, z_df$latent_3, col = "red")
