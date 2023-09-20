library(torch)
library(tidyverse)
library(FNN)
library(ape)
library(phyf)
library(Matrix)
library(GPUmatrix) ## this is awesome, I only just discovered it!

gb_tree_cutup <- pf_as_pf(read.tree("data/gc_cutup_tree_custom.tre"))
gb_tree_sparse <- pf_as_sparse(gb_tree_cutup)
#mv <- kronecker(Diagonal(11), gb_tree_sparse)

## some tests
gb_tree_mat <- gpu.matrix(gb_tree_sparse, dtype = "float32")@gm

#param <- torch_randn(85988, 11, device = "cuda", requires_grad = TRUE)
#test <- call_torch_function("torch__sparse_mm", gb_tree_mat, param, quiet = TRUE)

#' Calculates Mahalanobis distance between matrices (assuming diagonal covariance in z2)
#' using torch
#'
#' @param z1 Matrix of values (rows are points, columns dimensions)
#' @param z2 Matrix of values (rows are points, columns dimensions)
#' @param v2 Variances associated with z2 points, should have the same
#' dimension as z2
#'
#' @return A vector of distances
#' @export
#'
#' @examples
mahalanobis_squared <- function(z1, z2, v2) {

  n_1 <- z1$size(1)
  n_2 <- z2$size(1)
  dim <- z1$size(2)

  expanded_1 = z1$unsqueeze(2)$expand(c(n_1, n_2, dim))
  expanded_2 = z2$unsqueeze(1)$expand(c(n_1, n_2, dim))

  diff <- expanded_1 - expanded_2
  torch_sum((diff / v2) * diff, 3)

}

mahalanobis1 <- function(z1, z2, v2) {
  d <- z1 - z2
  torch_mm(torch_mm(d$unsqueeze(1), torch_inverse(torch_diag_embed(v2))), d$unsqueeze(-1))
}

gb_z <- read_csv("data/grambank_vae_latent_codes_v2.csv")
centroids_df <- gb_z %>%
  select(Language_ID, latent_dim, latent_mean) %>%
  pivot_wider(names_from = latent_dim, values_from = latent_mean)

# nn <- knn.dist(centroids %>%
#                  select(-Language_ID) %>%
#                  as.matrix(),
#                k = 1)
#
# rho <- max(nn)
# rho <- torch_tensor(rho, device = "cuda")

centroids <- centroids_df %>%
  select(-Language_ID) %>%
  as.matrix() %>%
  torch_tensor(device = "cuda")

vars <- gb_z %>%
  select(Language_ID, latent_dim, latent_var) %>%
  pivot_wider(names_from = latent_dim, values_from = latent_var) %>%
  select(-Language_ID) %>%
  as.matrix() %>%
  torch_tensor(device = "cuda")
#vars <- 1/vars

centroid_dists <- mahalanobis_squared(centroids, centroids, vars)$sqrt()$cpu() %>%
  as.matrix()
diag(centroid_dists) <- 999999999999999
mins <- apply(centroid_dists, 1, min)
rho <- max(mins)
rho <- torch_tensor(rho, device = "cuda")

get_metric_tensor <- function(z, centroids, vars, lambda = 1e-2, rho) {
  mh <- torch_exp(-(mahalanobis_squared(z, centroids, vars) / (rho^2)))
  1 / ((vars$unsqueeze(1) * mh$unsqueeze(-1))$sum(dim = 2) + lambda)
}

get_manifold_dist <- function(vel, metric) {
  (vel*metric*vel)$sum(dim = 2)$sqrt()
}

# tt <- get_metric_tensor(test, centroids, vars, rho = rho)
# dd <- (param*tt*param)$sum(dim = 2)$sqrt()
# mean_d <- torch_mean(dd)

dat <- gb_tree_cutup %>%
  left_join(centroids_df, by = c("label" = "Language_ID"))

which_tips <- which(pf_is_tips(gb_tree_cutup))
dat <- dat[which_tips, ] %>%
  select(-is_tip, -phlo, -label) %>%
  as.matrix() %>%
  torch_tensor(device = "cuda")


mani_evo_mod <- nn_module("mani_evo",
                          initialize = function(n_rates, n_dim, tree_mat, centroids, vars, lambda, tip_weight, evenness_weight, which_tips,
                                                device = "cuda") {

                            self$tree_mat <- tree_mat$to(device = device)
                            self$centroids <- centroids$to(device = device)
                            self$vars <- vars$to(device = device)
                            self$lambda <- torch_tensor(lambda)$to(device = device)
                            self$tip_weight <- torch_tensor(tip_weight)$to(device = device)
                            self$evenness_weight <- torch_tensor(evenness_weight)$to(device = device)
                            self$which_tips <- torch_tensor(which_tips)$to(device = device)

                            centroid_dists <- mahalanobis_squared(centroids, centroids, vars)$sqrt()$cpu() %>%
                              as.matrix()
                            diag(centroid_dists) <- 999999999999999
                            mins <- apply(centroid_dists, 1, min)
                            rho <- max(mins)
                            self$rho <- torch_tensor(rho, device = device)

                            self$rates <- nn_parameter(torch_randn(n_rates, n_dim) * 0.01)

                          },
                          get_metric_tensor = function(z) {
                            mh <- torch_exp(-(mahalanobis_squared(z, self$centroids, self$vars) / (self$rho^2)))
                            1 / ((self$vars$unsqueeze(1) * mh$unsqueeze(-1))$sum(dim = 2) + self$lambda)
                          },
                          get_manifold_dist = function(vel, metric) {
                            (vel*metric*vel)$sum(dim = 2)$sqrt()
                          },
                          tip_loss = function(tip_data, tip_recon, which_tips) {
                            tip_dists <- self$get_manifold_dist(tip_data - tip_recon[which_tips, ],
                                                                self$get_metric_tensor(tip_recon[which_tips, ]))
                            torch_mean(tip_dists)
                          },
                          dist_loss = function(dists) {
                            torch_mean(dists)
                          },
                          evenness_loss = function(rates) {
                            torch_var(torch_norm(rates, dim = 2))
                          },
                          forward = function(tip_data) {

                            z <- call_torch_function("torch__sparse_mm", self$tree_mat, self$rates, quiet = TRUE)

                            #with_autocast(device_type = "cuda", {
                              dists <- self$get_manifold_dist(self$rates,
                                                              self$get_metric_tensor(z))

                              dist_loss <- self$dist_loss(dists)
                              evenness_loss <- self$evenness_loss(self$rates)
                            #})
                            #tip_loss <- self$tip_loss(tip_data, z)
                            list(tip_data, z, dist_loss, evenness_loss)

                          })

mod <- mani_evo_mod(nrow(gb_tree_sparse), 11,
                    gb_tree_mat, centroids, vars,
                    lambda = 1e-2, tip_weight = 4,
                    evenness_weight = 4,
                    which_tips)
mod <- mod$cuda()

n_epoch <- 10000
lr <- 0.01
save_every <- 10

optim1 <- optim_adam(mod$parameters, lr = lr)
scheduler <- lr_one_cycle(optim1, max_lr = lr,
                          epochs = n_epoch, steps_per_epoch = 1,
                          cycle_momentum = FALSE)
#scaler <- cuda_amp_grad_scaler()

for(epoch in 1:n_epoch) {

  optim1$zero_grad()

  res <- mod(dat)

#  with_autocast(device_type = "cuda", {
    tip_loss <- mod$tip_loss(res[[1]], res[[2]], mod$which_tips)
    loss <- mod$tip_weight * tip_loss + mod$evenness_weight * res[[4]] + res[[3]]
#  })

  cat("Epoch: ", epoch,
      "    loss: ", as.numeric(loss$cpu()),
      "    tip recon loss: ", as.numeric(tip_loss$cpu()),
      "    tree distance loss: ", as.numeric(res[[3]]$cpu()),
      "    segment evenness loss: ", as.numeric(res[[4]]$cpu()),
      "\n")

  if(epoch %% save_every == 0) {
    write_csv(res[[2]]$cpu() %>%
                as.matrix() %>%
                as.data.frame(),
              file.path("data", "progress", "run_1", paste0("z_progress_", str_pad(epoch,
                                                                          5,
                                                                          pad = "0"),
                                                   ".csv")))
  }

  loss$backward()
  optim1$step()
  # scale(loss)$backward()
  # scaler$step(optim1)
  # scaler$update()
}

torch_save(mod, "data/mani_evo_mod_run_1.to")

test <- mod(dat)
test_dat <- as.matrix(test[[2]]$cpu()) %>%
  as.data.frame()

plot(test_dat[, 3:4], xlim = c(0,0.25), ylim = c(0, 0.25))
gb_mani_aces <- gb_tree_cutup %>%
  bind_cols(test_dat) %>%
  mutate(edge = sapply(strsplit(label, "_"), function(x) x[1]))


pdat <- pf_ends(gb_mani_aces$phlo) %>%
  left_join(gb_mani_aces %>%
              select(start = label,
                     starts_with("V")) %>%
              rename_with(~ paste0(.x, "_start"),
                          starts_with("V"))) %>%
  left_join(gb_mani_aces %>%
              select(end = label,
                     starts_with("V")) %>%
              rename_with(~ paste0(.x, "_end"),
                          starts_with("V")))

ggplot(pdat, aes(x = V1_start, y = V2_start)) +
  geom_segment(aes(xend = V1_end, yend = V2_end)) +
  theme_minimal() +
  geom_point(aes(V1, V2), data = gb_mani_aces %>% select(-phlo) %>% as_tibble(),
             colour = "red", size = 0.1)

ggplot(pdat, aes(x = V1_start, y = V2_start)) +
  geom_segment(aes(xend = V1_end, yend = V2_end)) +
  coord_cartesian(xlim = c(-0.25, 0.25),
                  ylim = c(-0.25, 0.25)) +
  theme_minimal() +
  geom_point(aes(V1, V2), data = gb_mani_aces %>% select(-phlo) %>% as_tibble(),
             colour = "red", size = 0.1)

ggplot(pdat, aes(x = V1_start, y = V2_start)) +
  geom_segment(aes(xend = V1_end, yend = V2_end)) +
  coord_equal(xlim = c(-1, 1),
                  ylim = c(-1, 1)) +
  theme_minimal() +
  geom_point(aes(V1, V2), data = gb_mani_aces %>% select(-phlo) %>% as_tibble(),
             colour = "red", size = 0.1)

ggplot(pdat, aes(x = V1_start, y = V2_start)) +
  geom_segment(aes(xend = V1_end, yend = V2_end)) +
  coord_equal(xlim = c(1, 2),
                  ylim = c(-1, -2)) +
  theme_minimal() +
  geom_point(aes(V1, V2), data = gb_mani_aces %>% select(-phlo) %>% as_tibble(),
             colour = "red", size = 0.1)

ggplot(pdat, aes(x = V3_start, y = V4_start)) +
  geom_segment(aes(xend = V3_end, yend = V4_end)) +
  theme_minimal() +
  coord_equal()

ggplot(pdat, aes(x = V5_start, y = V6_start)) +
  geom_segment(aes(xend = V5_end, yend = V6_end)) +
  theme_minimal() +
  geom_point(aes(V5, V6), data = gb_mani_aces %>% select(-phlo) %>% as_tibble(),
             colour = "red", size = 0.1)

ggplot(pdat, aes(x = V7_start, y = V8_start)) +
  geom_segment(aes(xend = V7_end, yend = V8_end)) +
  theme_minimal() +
  coord_equal() +
  geom_point(aes(V7, V8), data = gb_mani_aces %>% select(-phlo) %>% as_tibble(),
             colour = "red", size = 0.1)

ggplot(pdat, aes(x = V7_start, y = V8_start)) +
  geom_segment(aes(xend = V7_end, yend = V8_end)) +
  theme_minimal() +
  coord_equal(xlim = c(-0.25, 0.25),
              ylim = c(-0.25, 0.25)) +
  geom_point(aes(V7, V8), data = gb_mani_aces %>% select(-phlo) %>% as_tibble(),
             colour = "red", size = 0.1)

#
#
# ########### testing ###############
# z1 <- torch_rand(3, 10)
# z2 <- torch_rand(5, 10)
# v2 <- torch_randint(1, 4, c(5, 10))
# dists <- matrix(nrow = 3, ncol = 5)
# for(i in 1:3) {
#   for(j in 1:5) {
#     dists[i, j] <- as.numeric(mahalanobis1(z1[i, ], z2[j, ], v2[j, ]))
#   }
# }
# dists
# mahalanobis(z1, z2, v2)
