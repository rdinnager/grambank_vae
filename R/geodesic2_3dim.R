library(torch)
library(tidyverse)
library(FNN)
library(ape)
library(phyf)
library(Matrix)
library(GPUmatrix) ## this is awesome, I only just discovered it!

gb_pf <- read_rds("data/gb_pf_3dim.rds")
init_rates <- read_csv("data/init_rates_3dim.csv")
gb_pf <- gb_pf %>%
  left_join(init_rates %>%
              rename_with(~ paste0("rate_", .x),
                          starts_with("latent_")),
            by = c("label" = "label"))
blens <- pf_mean_edge_features(gb_pf$phlo)
names(blens) <- pf_edge_names(gb_pf$phlo)
blens <- blens[gb_pf$label]

gb_tips <- gb_pf %>%
  filter(is_tip) %>%
  select(starts_with("latent_")) %>%
  as.matrix()

gb_tree_mat <- pf_as_sparse(gb_pf$phlo)
gb_tree_mat <- gb_tree_mat[ , gb_pf$label]

start_mat <- pf_as_sparse(gb_pf$phlo)
#start_mat[ , !attr(gb_pf$phlo, "internal")] <- 0
start_mat <- start_mat[ , gb_pf$label]
for(i in 1:nrow(start_mat)) {
  start_mat[i, colnames(start_mat) == rownames(start_mat)[i]] <- 0
}

gb_tree_mat_tips <- gb_pf %>%
  filter(is_tip) %>%
  pf_as_sparse()
gb_tree_mat_tips <- gb_tree_mat_tips[ , gb_pf$label]

gb_dataset <- dataset(name = "gb_ds",
                           initialize = function(end_tree_mat,
                                                 start_tree_mat,
                                                 tip_tree_mat,
                                                 tip_dat,
                                                 blens) {
                             self$end_tree_mat <- end_tree_mat
                             self$start_tree_mat <- start_tree_mat
                             self$tip_tree_mat <- gpu.matrix(tip_tree_mat, dtype = "float32", device = "cuda")@gm
                             self$tip_dat <- torch_tensor(tip_dat, device = "cuda")
                             self$blens <- torch_tensor(blens, device = "cuda")
                           },
                           .getbatch = function(i) {
                              end_tree_mat <- gpu.matrix(self$end_tree_mat[i, ], dtype = "float32", device = "cuda")@gm
                              start_tree_mat <- gpu.matrix(self$start_tree_mat[i, ], dtype = "float32", device = "cuda")@gm
                              list(end_tree_mat, start_tree_mat, self$tip_tree_mat, self$tip_dat, self$blens[i], torch_tensor(i, device = "cuda"))                           },
                           .length = function() {
                             nrow(self$end_tree_mat)
                           })

gb_ds <- gb_dataset(gb_tree_mat, start_mat, gb_tree_mat_tips, gb_tips, blens)
gb_dl <- dataloader(gb_ds, 600, shuffle = TRUE, drop_last = FALSE)

gb_init_rates <- gb_pf %>%
  select(starts_with("rate_")) %>%
  as.matrix()

test_tips <- gb_tree_mat_tips %*% gb_init_rates
test_mat1 <- gb_tree_mat %*% gb_init_rates
test_mat2 <- start_mat %*% gb_init_rates

df <- tibble(x_1 = test_mat2[ , 1], y_1 = test_mat2[ , 2],
             x_2 = test_mat1[ , 1], y_2 = test_mat1[ , 2])

ggplot(df) + geom_segment(aes(x = x_1, xend = x_2,
                              y = y_1, yend = y_2))

test <- dataloader_next(dataloader_make_iter(gb_dl))


gb_z <- read_csv("data/grambank_vae_latent_codes_v2_3dim.csv")
centroids_df <- gb_z %>%
  select(Language_ID, latent_dim, latent_mean) %>%
  pivot_wider(names_from = latent_dim, values_from = latent_mean)

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

n_segs <- 100

init_rates_test <- torch_tensor(gb_init_rates, device = "cuda")
z_ends_test <- call_torch_function("torch__sparse_mm", test[[1]], init_rates_test, quiet = TRUE)
z_starts_test <- call_torch_function("torch__sparse_mm", test[[2]], init_rates_test, quiet = TRUE)

a_test <- torch_randn(z_starts_test$size()[1], z_starts_test$size()[2], device = "cuda")
b_test <- torch_randn(z_starts_test$size()[1], z_starts_test$size()[2], device = "cuda")
segs_test <- torch_arange(0, 1, 1 / n_segs, device = "cuda")

get_segments <- function(z_starts, z_ends, segs, a, b) {
  lens <- z_ends - z_starts
  d <- lens$size()[2]
  n <- segs$size()
  segs <- segs$unsqueeze(1L)$`repeat`(c(lens$size()[1], 1))
  y <- (a$unsqueeze(-1)$`repeat`(c(1, 1, n)) * (segs^3)$unsqueeze(2) + b$unsqueeze(-1)$`repeat`(c(1, 1, n)) * (segs^2)$unsqueeze(2)) +
    (lens - a - b)$unsqueeze(-1) * segs$unsqueeze(2)$`repeat`(c(1, d, 1)) +
    z_starts$unsqueeze(-1)
  vec <- torch_diff(y)
  y_mid <- y[ , , 1:(y$size()[3] - 1)] + vec / 2
  list(vec, y_mid)
}


## some tests
#gb_tree_mat <- gpu.matrix(gb_tree_sparse, dtype = "float32", device = "cuda")@gm


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

# n_segs <- 100
#
# init_rates_test <- torch_tensor(gb_init_rates, device = "cuda")
# z_ends_test <- call_torch_function("torch__sparse_mm", test[[1]], init_rates_test, quiet = TRUE)
# z_starts_test <- call_torch_function("torch__sparse_mm", test[[2]], init_rates_test, quiet = TRUE)
# segs_test <- torch_arange(0, 1, 1 / n_segs)
#
# a_test <- torch_randn(z_starts_test$size()[1], device = "cuda")
# b_test <- torch_randn(z_starts_test$size()[1], device = "cuda")
# segs <- torch_arange(0, 1, 1 / n_segs, device = "cuda")

zs_test <- get_segments(z_starts_test, z_ends_test, segs_test, a_test, b_test)

z1 <- zs_test[[2]]
z2 <- centroids
v2 <- vars
mahalanobis_squared <- mahalanobis_squared_fun(z1, z2, v2)

mahalanobis_simple <- function(z1, z2, v2) {

  n_1 <- z1$size(1)
  n_2 <- z2$size(1)
  dim <- z1$size(2)

  expanded_1 = z1$unsqueeze(2)$expand(c(n_1, n_2, dim))
  expanded_2 = z2$unsqueeze(1)$expand(c(n_1, n_2, dim))

  diff <- expanded_1 - expanded_2
  torch_sum((diff / v2) * diff, 3)

}

centroid_dists <- mahalanobis_simple(centroids, centroids, vars)$sqrt()$cpu() %>%
  as.matrix()
diag(centroid_dists) <- 999999999999999
mins <- apply(centroid_dists, 1, min)
rho <- max(mins)
rho <- torch_tensor(rho, device = "cuda")

nn <- knn.dist(centroids_df %>%
                 select(-Language_ID) %>%
                 as.matrix(),
               k = 1)

rho <- mean(nn)
rho <- torch_tensor(rho, device = "cuda")

write_rds(as.numeric(rho$cpu()), "data/estimated_rho_3dim.rds")

get_metric_tensor <- function(z, centroids, vars, lambda = 1e-2, rho) {
  mh <- torch_exp(-(mahalanobis_squared[[1]](z, centroids, vars) / (rho^2)))
  ## really relying on broadcasting correctly here! Hope I got it right!
  1 / (((1 / vars)$unsqueeze(1)$unsqueeze(-1) * mh$unsqueeze(3))$sum(dim = 2) + lambda)
}

get_manifold_dist <- function(vel, metric) {
  (vel*metric*vel)$sum(dim = 2)$sqrt()
}

met_test <- get_metric_tensor(z1, centroids, vars, rho = rho)
dist_test <- get_manifold_dist(zs_test[[1]], met_test)


evenness_loss <- function(dists) {
  torch_mean(torch_var(dists, 2))
}
evenness_loss(dist_test)

dist_loss <- function(dists) {
  torch_mean(dists)
}
dist_loss(dist_test)

tip_loss <- function(tip_data, tip_recon) {
  tip_dists <- torch_square(tip_data - tip_recon)
  torch_mean(tip_dists)
}

mani_evo_mod <- nn_module("mani_evo",
                          initialize = function(n_rates, n_dim, centroids, vars, lambda, tip_weight, evenness_weight,
                                                n_segs = 100, init_rates = NULL,
                                                device = "cuda") {

                            #self$tree_mat <- tree_mat$to(device = device)
                            self$n_rates <- n_rates
                            self$n_dim <- n_dim
                            self$n_segs <- n_segs
                            self$segs <- torch_arange(0, 1, 1 / n_segs, device = device)
                            self$centroids <- centroids$to(device = device)
                            self$vars <- vars$to(device = device)
                            self$lambda <- torch_tensor(lambda)$to(device = device)
                            self$tip_weight <- torch_tensor(tip_weight)$to(device = device)
                            self$evenness_weight <- torch_tensor(evenness_weight)$to(device = device)
                            #self$which_tips <- torch_tensor(which_tips)$to(device = device)

                            centroid_dists <- mahalanobis_simple(centroids, centroids, vars)$sqrt()$cpu() %>%
                              as.matrix()
                            diag(centroid_dists) <- 999999999999999
                            mins <- apply(centroid_dists, 1, min)
                            maxs <- apply(centroid_dists, 1, max)
                            rho <- max(mins)
                            target_rho <- rho / 3
                            start_rho <- rho * 3
                            self$target_rho <- torch_tensor(target_rho, device = device)
                            self$start_rho <- torch_tensor(start_rho, device = device)
                            self$rho <- torch_tensor(start_rho, device = device)

                            self$a <- nn_parameter(torch_randn(n_rates, n_dim) * 0.001)
                            self$b <- nn_parameter(torch_randn(n_rates, n_dim) * 0.001)
                            if(!is.null(init_rates)) {
                              self$rates <- nn_parameter(torch_tensor(init_rates))
                            } else {
                              self$rates <- nn_parameter(torch_randn(n_rates, n_dim) * 0.01)
                            }

                          },

                          forward = function(x) {

                            z_ends <- call_torch_function("torch__sparse_mm", x[[1]], self$rates, quiet = TRUE)
                            z_starts <- call_torch_function("torch__sparse_mm", x[[2]], self$rates, quiet = TRUE)
                            z_tips <- call_torch_function("torch__sparse_mm", x[[3]], self$rates, quiet = TRUE)

                            #segs <- torch_arange(0, 1, 1 / self$n_segs)
                            zs <- get_segments(z_starts, z_ends, self$segs, self$a[x[[6]]], self$b[x[[6]]])

                            met <- get_metric_tensor(zs[[2]], centroids, vars, rho = self$rho)
                            dists <- get_manifold_dist(zs[[1]], met) / (x[[5]]$unsqueeze(-1) / self$n_segs)

                            dist_loss <- dist_loss(dists)
                            #evenness_loss <- evenness_loss(dists) * self$evenness_weight
                            evenness_loss <- 0
                            tip_loss <- tip_loss(x[[4]], z_tips) * self$tip_weight

                            #tip_loss <- self$tip_loss(tip_data, z)
                            list(dist_loss, evenness_loss, tip_loss, zs, dists)

                          })


mod <- mani_evo_mod(n_rates = nrow(gb_init_rates), n_dim = 3,
                    centroids, vars,
                    lambda = 1e-3, tip_weight = 10,
                    evenness_weight = 1,
                    n_segs = 100, init_rate = gb_init_rates,
                    device = "cuda")
mod <- mod$cuda()

write_rds(as.numeric(mod$target_rho$cpu()), "data/estimated_rho_3dim_rho_schedule.rds")

#tt <- mod(test)

## tests
#z <- call_torch_function("torch__sparse_mm", mod$tree_mat, mod$rates, quiet = TRUE)
#z <- torch_cat(list(z, z, z))
#dists <- list()
# for(i in 1:22) {
#   met <- get_metric_tensor(z[(((i - 1) * 1000) + 1):(((i) * 1000)), ], centroids, vars, rho = rho)
#   dists <- get_manifold_dist(mod$rates[(((i - 1) * 1000) + 1):(((i) * 1000)), ],
#                              met)
#   dist <- torch_mean(dists)
#   dist$backward()
#   print(i)
# }

cosine_schedule <- function(t, start=0, end=1, tau=1, clip_min=1e-9) {
  # A gamma function based on cosine function.
  v_start <- cos(start * pi / 2) ^ (2 * tau)
  v_end <- cos(end * pi / 2) ^ (2 * tau)
  output <- cos((t * (end - start) + start) * pi / 2) ^ (2 * tau)
  output <- (v_end - output) / (v_end - v_start)
  output[output < clip_min] <- clip_min
  output[output > 1.0] <- 1.0
  output
}

n_epoch <- 10000
lr <- 0.02
save_every <- 50

optim1 <- optim_adam(mod$parameters, lr = lr)
scheduler <- lr_one_cycle(optim1, max_lr = lr,
                          epochs = n_epoch, steps_per_epoch = 1,
                          cycle_momentum = FALSE)
#scaler <- cuda_amp_grad_scaler()

optim1$zero_grad()

clamp_rates <- function(rates, blens, min_rate = 0.001) {
  norms <- torch_abs(rates / torch_tensor(blens, device = "cuda")$unsqueeze(-1))
  norms_clamped <- torch_clamp(norms, min = min_rate)
  rates <- rates / norms * norms_clamped
  rates
}

test_dat <- list()

for(epoch in 1:n_epoch) {

  optim1$zero_grad()
  total_loss <- 0
  total_recon_loss <- 0
  total_dist_loss <- 0
  #total_evenness_loss <- 0

  ## make sure no rate gets too close to zero
  ## (which leads to nan gradients)
  #mod$rates <- clamp_rates(mod$rates, blens, min_rate = 0.0001)

  i <- 0
  coro::loop(for (b in gb_dl) {

      i <- i + 1

      res <- mod(b)

      if(epoch %% save_every == 0) {
        with_no_grad({
          test_dat[[i]] <- purrr::map(purrr::list_flatten(res[-2]), ~.x$cpu()$detach())
        })
      }

      loss <- (res[[1]] + res[[3]]) * 100 #/ length(gb_dl) ## + res[[2]]
      loss$backward()

      ## deal with undeflowing nan issue
      ## hope it doesn't have weird unintended consequences
      mod$a$grad <- mod$a$grad$nan_to_num()
      mod$b$grad <- mod$b$grad$nan_to_num()
      mod$rates$grad <- mod$rates$grad$nan_to_num()
      #nn_utils_clip_grad_norm_(mod$parameters, 1)

      total_loss <- total_loss + loss
      total_recon_loss <- total_recon_loss + (res[[3]] * 10) #/ length(gb_dl))
      total_dist_loss <- total_dist_loss + (res[[1]] * 10)# / length(gb_dl))
   #   total_evenness_loss <- total_evenness_loss + (res[[2]] / length(gb_dl))

  })

  cat("Epoch: ", epoch,
      "    loss: ", as.numeric(total_loss$cpu()),
      "    tip recon loss: ", as.numeric(total_recon_loss$cpu()),
      "    tree distance loss: ", as.numeric(total_dist_loss$cpu()),
      "    rho: ", as.numeric(mod$rho$cpu()),
#      "    segment evenness loss: ", as.numeric(total_evenness_loss$cpu()),
      "\n")

  if(epoch %% save_every == 0) {

    test_zs <- torch_cat(map(test_dat, 4)) %>%
      as.array()
    plot(cbind(as.vector(test_zs[ , 1, ]), as.vector(test_zs[, 2, ])), type = "n")
    points(as.matrix(centroids$cpu())[, 1:2], cex = 0.2, col = "green")
    for(j in seq_len(dim(test_zs)[1])) {
      points(cbind(as.vector(test_zs[j, 1, ]), as.vector(test_zs[j, 2, ])), type = "l", col = "red")
    }
    points(as.matrix(centroids$cpu())[, 1:2], cex = 0.2, col = "green")


  #   write_csv(res[[2]]$cpu() %>%
  #               as.matrix() %>%
  #               as.data.frame(),
  #             file.path("data", "progress", "run_2", paste0("z_progress_", str_pad(epoch,
  #                                                                         5,
  #                                                                         pad = "0"),
  #                                                  ".csv")))
  }


  optim1$step()

  ## update rho
  prop_epoch <- epoch / n_epoch
  noise <- cosine_schedule(prop_epoch, 0.1, 1, 6)
  new_rho <- mod$target_rho + (mod$start_rho - mod$target_rho) * noise
  mod$rho <- new_rho

  # scale(loss)$backward()
  # scaler$step(optim1)
  # scaler$update()
}

options(torch.serialization_version = 2)
torch_save(mod, "data/mani_evo_mod_run_new_full_param_3dim_rho_scedule.to")

gb_test_dl <- dataloader(gb_ds, 100, shuffle = FALSE, drop_last = FALSE)
test_dat <- list()
i <- 0
coro::loop(for (b in gb_test_dl) {
  i <- i + 1
  with_no_grad({
    test_dat[[i]] <- purrr::map(purrr::list_flatten(mod(b)[-2]), ~.x$cpu()$detach())
  })
  print(i)
})

test_zs <- torch_cat(map(test_dat, 4)) %>%
  as.array()

plot(cbind(as.vector(test_zs[ , 1, ]), as.vector(test_zs[, 2, ])), type = "n")
points(as.matrix(centroids$cpu())[, 1:2], cex = 0.2, col = "green")
for(i in seq_len(dim(test_zs)[1])) {
  points(cbind(as.vector(test_zs[i, 1, ]), as.vector(test_zs[i, 2, ])), type = "l", col = "red")
  print(i)
}
points(as.matrix(centroids$cpu())[, 1:2], cex = 0.2, col = "green")

plot(cbind(as.vector(test_zs[ , 2, ]), as.vector(test_zs[, 3, ])), type = "n")
points(as.matrix(centroids$cpu())[, 2:3], cex = 0.2, col = "green")
for(i in seq_len(dim(test_zs)[1])) {
  points(cbind(as.vector(test_zs[i, 2, ]), as.vector(test_zs[i, 3, ])), type = "l", col = "red")
  print(i)
}
points(as.matrix(centroids$cpu())[, 2:3], cex = 0.2, col = "green")


plot(cbind(as.vector(test_zs[ , 1, ]), as.vector(test_zs[, 3, ])), type = "n")
points(as.matrix(centroids$cpu())[, c(1, 3)], cex = 0.2, col = "green")
for(i in seq_len(dim(test_zs)[1])) {
  points(cbind(as.vector(test_zs[i, 1, ]), as.vector(test_zs[i, 3, ])), type = "l", col = "red")
  print(i)
}
points(as.matrix(centroids$cpu())[, c(1, 3)], cex = 0.2, col = "green")


######## get the data into a tibble and save it ############

edge_ids <- rownames(gb_tree_mat)
z_list_l1 <- array_branch(test_zs, 1) %>%
  purrr::map(~ as.data.frame(t(.x)) %>%
               rename(latent_1 = V1,
                      latent_2 = V2,
                      latent_3 = V3))

z_tree_df <- tibble(edge = edge_ids) %>%
  mutate(z_seqs = z_list_l1) %>%
  left_join(gb_pf %>% select(edge = label, is_tip,
                             true_latent_1 = latent_1,
                             true_latent_2 = latent_2,
                             true_latent_3 = latent_3))

lang_dat <- read_csv("data/cldf/languages.csv")
z_tree_df <- z_tree_df %>%
  left_join(lang_dat, by = c("edge" = "Glottocode"))

write_rds(z_tree_df, "data/gb_vae_aces_3dim_noise_schedule.rds")

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
              col = "red",
              size = 5)
rgl::texts3d(z_tree_df$true_latent_1[z_tree_df$is_tip],
              z_tree_df$true_latent_2[z_tree_df$is_tip],
              z_tree_df$true_latent_3[z_tree_df$is_tip],
             z_tree_df$edge[z_tree_df$is_tip],
             adj = c(0, 0))








points(cbind(as.vector(test_zs[6, 1, ]), as.vector(test_zs[6, 2, ])), type = "l", col = "red")
ss <- sample.int(2804, 1)
points(cbind(as.vector(test_zs[ss, 1, ]), as.vector(test_zs[ss, 2, ])), type = "l", col = "red")

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
