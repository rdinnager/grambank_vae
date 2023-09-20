library(tidyverse)
library(phyf)
library(rgl)
library(ape)
library(phytools)
library(colourvalues)
library(fibre)

set.seed(123456789)

######### simulate evolution on a 'swiss roll' manifold
## start with two standard Brownian motion variables
## random tree, random sim of two traits

tree <- rtree(400)  %>%
  force.ultrametric()
tree$edge.length[tree$edge.length <= 0] <- 0.01
tree <- tree %>%
  pf_as_pf()
# cut_info <- pf_epoch_info(tree, seq(0,
#                                     max(node.depth.edgelength(pf_as_phylo(tree))),
#                                     length.out = 1000))
# tree_cut <- pf_edge_segmentize(tree, cut_info$edge, cut_info$position)

# tree_cut_pf <- pf_as_phylo(tree_cut, collapse_singletons = FALSE) %>%
#   pf_as_pf()
traits <- pf_as_sparse(tree) %*% matrix(rnorm(2 * nrow(tree), sd = 0.5), ncol = 2)

tt <- (1 + ((traits[ , 2] - min(traits[ , 2])) / max(traits[ , 2])) * 2) * (3 * pi/2)
tt_0 <- (1 + ((0 - min(traits[ , 2])) / max(traits[ , 2])) * 2) * (3 * pi/2)

x <- tt * cos(tt)
y <- traits[ , 1] * 15
z <- tt * sin(tt)

x_0 <- tt_0 * cos(tt_0)
y_0 <- 0
z_0 <- tt_0 * sin(tt_0)

rgl::points3d(x, y, z)

# test <- KODAMA::swissroll(200)
# rgl::points3d(test)

tree_swissroll <- tree %>%
  mutate(x = x, y = y, z = z, tt = tt) %>%
  mutate(time = pf_flow_sum(phlo))

pdat <- pf_ends(tree_swissroll$phlo) %>%
  mutate(isna = is.na(end),
         end = ifelse(isna, start, end),
         start = ifelse(isna, NA, start)) %>%
  left_join(tree_swissroll %>%
              select(start = label,
                     x_start = x, y_start = y, z_start = z,
                     tt_start = tt,
                     time_start = time)) %>%
  left_join(tree_swissroll %>%
              select(end = label,
                     x_end = x, y_end = y, z_end = z,
                     tt_end = tt,
                     time_end = time)) %>%
  mutate(x_start = ifelse(isna, x_0, x_start),
         y_start = ifelse(isna, y_0, y_start),
         z_start = ifelse(isna, z_0, z_start),
         tt_start = ifelse(isna, tt_0, tt_start),
         time_start = ifelse(isna, 0, time_start))

pdat <- pdat %>%
  rowwise() %>%
  mutate(tt_seq = list(seq(from = tt_start, to = tt_end, length.out = ceiling(abs(tt_end - tt_start) / 0.01)))) %>%
  mutate(new_x_seq = list(tt_seq * cos(tt_seq)),
         new_y_seq = list(seq(from = y_start, to = y_end, length.out = length(tt_seq))),
         new_z_seq = list(tt_seq * sin(tt_seq)),
         new_time_seq = list(seq(from = time_start, to = time_end, length.out = length(tt_seq))))

x_line <- unlist(purrr::map(pdat$new_x_seq, ~ c(.x, NA)))
y_line <- unlist(purrr::map(pdat$new_y_seq, ~ c(.x, NA)))
z_line <- unlist(purrr::map(pdat$new_z_seq, ~ c(.x, NA)))
time_line <- unlist(purrr::map(pdat$new_time_seq, ~ c(.x, NA)))

#points3d(x_line, y_line, z_line)
lines3d(x_line, y_line, z_line, lwd = 5)
points3d(tree_swissroll %>% filter(is_tip) %>% pull(x),
         tree_swissroll %>% filter(is_tip) %>% pull(y),
         tree_swissroll %>% filter(is_tip) %>% pull(z),
         col = "red", size = 10)


clear3d()
inds <- time_line <= 0.01
inds[is.na(inds)] <- TRUE
id_old <- lines3d(x_line[inds], y_line[inds], z_line[inds], lwd = 3)
points3d(x_0, y_0, z_0, col = "yellow", size = 15)
Sys.sleep(3)
for(i in seq(0.01, max(time_line, na.rm = TRUE), length.out = 500)) {

  inds <- time_line <= i
  inds[is.na(inds)] <- TRUE
  id <- lines3d(x_line[inds], y_line[inds], z_line[inds], lwd = 3)
  pop3d(id = id_old)
  id_old <- id

  points3d(tree_swissroll %>% filter(is_tip, time <= i) %>% pull(x),
           tree_swissroll %>% filter(is_tip, time <= i) %>% pull(y),
           tree_swissroll %>% filter(is_tip, time <= i) %>% pull(z),
           col = "red", size = 10)
  points3d(tree_swissroll %>% filter(!is_tip, time <= i) %>% pull(x),
           tree_swissroll %>% filter(!is_tip, time <= i) %>% pull(y),
           tree_swissroll %>% filter(!is_tip, time <= i) %>% pull(z),
           col = "blue", size = 6)


}
points3d(x_0, y_0, z_0, col = "yellow", size = 15)

################ put it into a dataset and analyse ###############
tip_swissroll <- tree_swissroll %>%
  filter(is_tip)

cols <- colour_values(tip_swissroll$tt)
points3d(tip_swissroll$x,
         tip_swissroll$y,
         tip_swissroll$z,
         col = cols, size = 10
         )
axes3d()
title3d(xlab = "x", ylab = "y", zlab = "z")

## pca
swissroll_pca <- prcomp(tip_swissroll %>%
                          select(x, y, z))
pca_df <- swissroll_pca$x %>%
  as.data.frame() %>%
  mutate(tt = tip_swissroll$tt)

ggplot(pca_df, aes(PC1, PC2)) +
  geom_point(aes(colour = tt), size = 3) +
  scale_colour_viridis_c(name = "Distance\nalong spiral") +
  theme_minimal()

plot(swissroll_pca)

ggplot(pca_df, aes(PC2, PC3)) +
  geom_point(aes(colour = tt), size = 3) +
  scale_colour_viridis_c(name = "Distance\nalong spiral") +
  theme_minimal()

##### run fibre

swissroll <- tree_swissroll %>%
  select(label, is_tip, phlo) %>%
  left_join(tip_swissroll %>% select(-phlo, -is_tip))
mean_sds <- swissroll %>%
  summarise(x_m = mean(x, na.rm = TRUE),
            y_m = mean(y, na.rm = TRUE),
            z_m = mean(z, na.rm = TRUE),
            x_s = sd(x, na.rm = TRUE),
            y_s = sd(y, na.rm = TRUE),
            z_s = sd(z, na.rm = TRUE))
swissroll <- swissroll %>%
  mutate(x = (x - mean_sds$x_m) / mean_sds$x_s,
         y = (y - mean_sds$y_m) / mean_sds$y_s,
         z = (z - mean_sds$z_m) / mean_sds$z_s)

res <- fibre(x + y + z ~ bre_brownian(phlo),
             data = swissroll,
             engine_options = list(control.family =
                                     list(hyper =
                                            list(hyper =
                                                   list(prec =
                                                          list(prior = "pc.prec",
                                                               initial = 4,
                                                               fixed = TRUE))))))

swissroll_aces <- predict(res)

swissroll_aces_df <- swissroll %>%
  bind_cols(purrr::map(swissroll_aces,
                       ".pred_mean") %>%
              as_tibble() %>%
              rename(x_pred = x, y_pred = y, z_pred = z)) %>%
  mutate(x_pred = (x_pred * mean_sds$x_s) + mean_sds$x_m,
         y_pred = (y_pred * mean_sds$y_s) + mean_sds$y_m,
         z_pred = (z_pred * mean_sds$z_s) + mean_sds$z_m)

pdat2 <- pf_ends(swissroll_aces_df$phlo) %>%
  mutate(isna = is.na(end),
         end = ifelse(isna, start, end),
         start = ifelse(isna, NA, start)) %>%
  left_join(swissroll_aces_df %>%
              select(start = label,
                     x_start = x_pred, y_start = y_pred,
                     z_start = z_pred)) %>%
  left_join(swissroll_aces_df %>%
              select(end = label,
                     x_end = x_pred, y_end = y_pred,
                     z_end = z_pred)) %>%
  mutate(x_start = ifelse(isna, 0, x_start),
         y_start = ifelse(isna, 0, y_start),
         z_start = ifelse(isna, 0, z_start))

segs_x <- pdat2 %>%
  select(x_start, x_end) %>%
  mutate(na = NA) %>%
  as.matrix() %>%
  t() %>%
  as.vector()

segs_y <- pdat2 %>%
  select(y_start, y_end) %>%
  mutate(na = NA) %>%
  as.matrix() %>%
  t() %>%
  as.vector()

segs_z <- pdat2 %>%
  select(z_start, z_end) %>%
  mutate(na = NA) %>%
  as.matrix() %>%
  t() %>%
  as.vector()

lines3d(segs_x, segs_y, segs_z)

cols <- colour_values(swissroll_aces_df$tt)
points3d(swissroll_aces_df$x_pred,
         swissroll_aces_df$y_pred,
         swissroll_aces_df$z_pred,
         col = cols, size = 10
         )
axes3d()
title3d(xlab = "x", ylab = "y", zlab = "z")

lines3d(x_line, y_line, z_line)
cols <- colour_values(tree_swissroll %>%
                        mutate(tt = ifelse(is_tip, tt, NA)) %>%
                        pull(tt))
points3d(tree_swissroll %>% pull(x),
         tree_swissroll %>% pull(y),
         tree_swissroll %>% pull(z),
         col = cols, size = 10)
axes3d()
title3d(xlab = "x", ylab = "y", zlab = "z")

##### run the vae

library(torch)
library(tidyverse)
library(dagnn)
library(zeallot)

swissroll_dataset <- dataset(name = "swissroll_ds",
                           initialize = function(rr) {
                             self$swissroll <- torch_tensor(rr)
                           },
                           .getbatch = function(i) {
                             self$swissroll[i, ]
                           },
                           .length = function() {
                             self$swissroll$size()[[1]]
                           })

train_tt <- runif(2000, min(swissroll$tt, na.rm = TRUE),
                  max(swissroll$tt, na.rm = TRUE))
train_y <- runif(2000, min(swissroll$y, na.rm = TRUE),
                  max(swissroll$y, na.rm = TRUE))
train_x <- ((train_tt * cos(train_tt)) - mean_sds$x_m) / mean_sds$x_s
train_z <- ((train_tt * sin(train_tt)) - mean_sds$z_m) / mean_sds$z_s

points3d(train_x, train_y, train_z, col = colour_values(train_tt))
points3d(swissroll %>%
           filter(is_tip) %>%
           select(x, y, z) %>%
           as.matrix(),
         col = "red", size = 10)

train_ds <- swissroll_dataset(cbind(train_x, train_y, train_z))
train_dl <- dataloader(train_ds, 500, shuffle = TRUE)

## note that this module includes the ability to provide data to condition on, making
## it a 'conditional VAE', however, for this study I do not use any conditioning information
## In this case we just provide a tensor of zeroes for the conditioning data, effectively turning
## it into an unconditional VAE. The conditioning ability is for future extensions.
vae_mod <- nn_module("CVAE",
                 initialize = function(input_dim, c_dim, latent_dim, breadth = 128, loggamma_init = 0) {
                   self$latent_dim <- latent_dim
                   self$input_dim <- input_dim
                   self$encoder <- nndag(i_1 = ~ input_dim,
                                         c = ~ c_dim,
                                         c_1 = c ~ c_dim,
                                         e_1 = i_1 + c_1 ~ breadth,
                                         e_2 = e_1 + c_1 ~ breadth,
                                         e_3 = e_2 + c_1 ~ breadth,
                                         means = e_3 ~ latent_dim,
                                         logvars = e_3 ~ latent_dim,
                                         .act = list(nn_relu,
                                                     logvars = nn_identity,
                                                     means = nn_identity))

                   self$decoder <- nndag(i_1 = ~ latent_dim,
                                         c = ~ c_dim,
                                         c_1 = c ~ c_dim,
                                         d_1 = i_1 + c_1 ~ breadth,
                                         d_2 = d_1 + c_1 ~ breadth,
                                         d_3 = d_2 + c_1 ~ breadth,
                                         out = d_3 ~ input_dim,
                                         .act = list(nn_relu,
                                                     out = nn_identity))

                   self$loggamma <- nn_parameter(torch_tensor(loggamma_init))

                 },
                 reparameterize = function(mean, logvar) {
                   std <- torch_exp(torch_tensor(0.5, device = "cuda") * logvar)
                   eps <- torch_randn_like(std)
                   eps * std + mean
                 },
                 loss_function = function(reconstruction, input, mean, log_var, loggamma) {
                   #kl <- torch_sum(torch_exp(log_var) + torch_square(mean) - log_var, dim = 2L) - latent_dim
                   log_sd <- torch_log(torch_sqrt(torch_exp(log_var)))
                   batch_size <- mean$size()[1]
                   kl <- torch_sum(torch_square(mean) + torch_square(torch_exp(log_sd)) - 2 * log_sd - 1) / 2.0 / batch_size
                   #recon1 <- torch_sum(torch_square(input - reconstruction), dim = 2L) / torch_exp(loggamma)
                   recon1 <- torch_sum(torch_square((input - reconstruction) / torch_exp(loggamma)) / 2.0) / batch_size
                   #recon2 <- self$input_dim * loggamma + torch_log(torch_tensor(2 * pi, device = "cuda")) * self$input_dim
                   recon2 <- self$input_dim * (loggamma + 0.5*torch_log(torch_tensor(2 * pi, device = "cuda")))
                   #loss <- torch_mean(recon1 + recon2 + kl)
                   loss <- recon1 + recon2 + kl
                   list(loss, torch_mean(recon1*torch_exp(loggamma)), torch_mean(kl))
                 },
                 forward = function(x, c) {
                   c(means, log_vars) %<-% self$encoder(c, x)
                   z <- self$reparameterize(means, log_vars)
                   list(self$decoder(c, z), x, means, log_vars)
                 })

input_dim <- 3
c_dim <- 1
latent_dim <- 2
breadth <- 512

vae <- vae_mod(input_dim, c_dim, latent_dim, breadth,
               loggamma_init = -20)
vae <- vae$cuda()

num_epochs <- 50000

lr <- 0.005
optimizer <- optim_adam(vae$parameters, lr = lr)
scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                          epochs = num_epochs, steps_per_epoch = length(train_dl),
                          cycle_momentum = FALSE)


# mseloss <- 0.000001
# gamma_x <- sqrt(mseloss)
# loggamma <- log(gamma_x)

for (epoch in 1:num_epochs) {

    batchnum <- 0
    coro::loop(for (b in train_dl) {

        batchnum <- batchnum + 1
        input <- b$to(device = "cuda")
        optimizer$zero_grad()
        c_null <- torch_zeros(input$size()[[1]], 1, device = "cuda") ## null conditioning data
        c(reconstruction, input, mean, log_var) %<-% vae(input, c_null)
        c(loss, reconstruction_loss, kl_loss) %<-% vae$loss_function(reconstruction, input, mean, log_var, vae$loggamma)

        # mseloss <- min(mseloss, mseloss * .99 + as.numeric(reconstruction_loss$cpu() / vae$input_dim) * .01)
        # gamma_x <- sqrt(mseloss)
        # loggamma <- log(gamma_x)

        if(batchnum %% 2 == 0) {

            cat("Epoch: ", epoch,
                "    batch: ", batchnum,
                "    loss: ", as.numeric(loss$cpu()),
                "    recon loss: ", as.numeric(reconstruction_loss$cpu()),
                "    KL loss: ", as.numeric(kl_loss$cpu()),
                "    loggamma: ", as.numeric(vae$loggamma$cpu()),
                #"    loggamma: ", loggamma,
                "    active dims: ", as.numeric((torch_exp(log_var)$mean(dim = 1L) < 0.5)$sum()$cpu()),
                "\n")

        }
        loss$backward()
        optimizer$step()
        scheduler$step()
    })
}

options(torch.serialization_version = 2)
torch_save(vae, "data/trained_vae_swissroll.to")


post_dl <- dataloader(train_ds, 200, shuffle = FALSE)
post_it <- as_iterator(post_dl)

all_mean_var <- list()
it <- 0
loop(for(i in post_dl) {
  it <- it + 1
  with_no_grad({
    c_null <- torch_zeros(i$size()[[1]], 1, device = "cuda")
    all_mean_var[[it]] <- vae$encoder(c_null, i$cuda())
  })
  print(it)
})

all_means <- purrr::map(all_mean_var, 1) %>%
  purrr::map(~ as.matrix(.x$cpu())) %>%
  do.call(rbind, .)

all_vars <- purrr::map(all_mean_var, 2) %>%
  purrr::map(~ as.matrix(torch_exp(.x)$cpu())) %>%
  do.call(rbind, .)

vars <- apply(all_vars, 2, mean)

vars_df <- tibble(`Mean Variance` = vars) %>%
  mutate(Status = ifelse(`Mean Variance` < 0.5, "Keep", "Toss"))

ggplot(vars_df, aes(`Mean Variance`)) +
  geom_histogram(aes(fill = Status)) +
  ylab("Count") +
  theme_minimal() +
  theme(legend.position = "none")

# dat <- swissroll %>%
#   filter(is_tip) %>%
#   select(x, y, z, tt) %>%
#   bind_cols(as.data.frame(all_means)) %>%
#   as_tibble()

ggplot(all_means %>% as.data.frame() %>%
         mutate(tt = train_tt),
       aes(V1, V2)) +
  geom_point(aes(colour = tt)) +
  scale_color_viridis_c() +
  theme_minimal()
