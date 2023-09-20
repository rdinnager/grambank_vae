####### This script fits a Two-Stage Variational Autoencoder (VAE) to the Grambank data ##########

####### Load Libraries ###################
library(torch)
library(tidyverse)
library(dagnn) ## This is in-development package: https://github.com/rdinnager/dagnn
library(zeallot)
library(conflicted)

####### Load Data ########################
grambank <- read_tsv("data/GB_cropped_for_missing.tsv")
## reshape to a matrix with GB parameters in columns, languages in rows
## and replace ? with NA
# grambank <- grambank %>%
#   select(Language_ID, Parameter_ID, Value) %>%
#   pivot_wider(names_from = Parameter_ID, values_from = Value) %>%
#   mutate(across(-Language_ID, .fns = ~ ifelse(.x == "?", NA_integer_, as.integer(.x))))

lang_ids <- grambank$Language_ID

## For now we will just use binary codes for simplicity. We can add back in other codes
## using one hot coding later perhaps?
binary <- map_lgl(grambank,
                 ~ all(unique(na.omit(.x)) %in% c(0L, 1L)))
grambank <- grambank[ , binary]

## Make a 'missingness' mask that will be added as conditioning data
observed <- grambank %>%
  mutate(across(everything(), ~ as.numeric(!is.na(.x))))

## now replace missing values with 0
grambank <- grambank %>%
  mutate(across(everything(), ~ ifelse(is.na(.x), 0, .x)))

## Now convert to matrices and make a torch dataloader for it
grambank_mat <- apply(as.matrix(grambank), 2, as.integer)
observed_mat <- apply(as.matrix(observed), 2, as.integer)

grambank_dataset <- dataset(name = "grambank_ds",
                            initialize = function(gb, ob) {
                              self$gb <- torch_tensor(gb, dtype = torch_float())
                              self$ob <- torch_tensor(ob, dtype = torch_float())
                            },
                            .getbatch = function(i) {
                              list(gb = self$gb[i, ],
                                   ob = self$ob[i, ])
                            },
                            .length = function() {
                              self$gb$size()[[1]]
                            })

train_ds <- grambank_dataset(grambank_mat, observed_mat)
train_dl <- dataloader(train_ds, 1240, shuffle = TRUE)

####### Setup VAE model as nn_module ###########
vae_mod <- nn_module("CVAE",
                 initialize = function(input_dim, latent_dim, breadth) {
                   self$latent_dim <- latent_dim
                   self$input_dim <- input_dim
                   self$m_encoder <- nndag(m_1 = ~ input_dim,
                                           x_1 = ~ input_dim,
                                           me_1 = m_1 + x_1 ~ breadth,
                                           me_2 = me_1 ~ breadth,
                                           me_3 = me_2 ~ breadth,
                                           m_means = me_3 ~ latent_dim,
                                           m_logvars = me_3 ~ latent_dim,
                                           .act = list(nn_relu,
                                                       m_logvars = nn_identity,
                                                       m_means = nn_identity))
                   self$m_decoder <- nndag(mz_1 = ~ latent_dim,
                                           z_1 = ~ latent_dim,
                                           md_1 = mz_1 + z_1 ~ breadth,
                                           md_2 = md_1 ~ breadth,
                                           md_3 = md_2 ~ breadth,
                                           out = md_3 ~ input_dim,
                                           .act = list(nn_relu,
                                                       out = nn_sigmoid))
                   self$encoder <- nndag(x_1 = ~ input_dim,
                                         e_1 = x_1 ~ breadth,
                                         e_2 = e_1 ~ breadth,
                                         e_3 = e_2 ~ breadth,
                                         means = e_3 ~ latent_dim,
                                         logvars = e_3 ~ latent_dim,
                                         .act = list(nn_relu,
                                                     logvars = nn_identity,
                                                     means = nn_identity))

                   self$decoder <- nndag(z_1 = ~ latent_dim,
                                         d_1 = z_1 ~ breadth,
                                         d_2 = d_1 ~ breadth,
                                         d_3 = d_2 ~ breadth,
                                         out = d_3 ~ input_dim,
                                         .act = list(nn_relu,
                                                     out = nn_sigmoid))

                 },
                 reparameterize = function(mean, logvar) {
                   std <- torch_exp(torch_tensor(0.5, device = "cuda") * logvar)
                   eps <- torch_randn_like(std)
                   eps * std + mean
                 },
                 loss_function = function(reconstruction, input, recon_miss, missing, mean, log_var, m_mean, m_logvar, loggamma) {
                   kl <- torch_sum(torch_exp(log_var) + torch_square(mean) - log_var, dim = 2L) - self$latent_dim
                   m_kl <- torch_sum(torch_exp(m_logvar) + torch_square(m_mean) - m_logvar, dim = 2L) - self$latent_dim
                   recon1 <- torch_sum(nnf_binary_cross_entropy(reconstruction, input, reduction = "none") * missing, dim = 2L) / torch_exp(loggamma)
                   m_recon1 <- torch_sum(nnf_binary_cross_entropy(recon_miss, missing, reduction = "none"), dim = 2L) / torch_exp(loggamma)
                   #recon2 <- self$input_dim * self$logtau
                   loss <- torch_mean(recon1 + kl + m_recon1 + m_kl)
                   list(loss, torch_mean(recon1) * torch_exp(loggamma), torch_mean(kl), torch_mean(m_recon1) * torch_exp(loggamma), torch_mean(m_kl))
                 },
                 forward = function(x, m) {
                   x2 <- ((x * 2) - 1) * m
                   c(means, log_vars) %<-% self$encoder(x2)
                   c(m_means, m_logvars) %<-% self$m_encoder(m, x2)
                   z <- self$reparameterize(means, log_vars)
                   mz <- self$reparameterize(m_means, m_logvars)
                   m_recon <- self$m_decoder(mz, z)
                   recon <- self$decoder(z)
                   #total_recon <- recon * m_recon
                   list(recon, m_recon, x, m, means, log_vars, m_means, m_logvars)
                 })

######## Run VAE stage 1 training ##################
## params
input_dim <- ncol(grambank_mat)
#c_dim <- ncol(observed_mat)
latent_dim <- 64L
breadth <- 1024L

vae <- vae_mod(input_dim, latent_dim, breadth)
vae <- vae$cuda()

num_epochs <- 15000

lr <- 0.002
optimizer <- optim_adam(vae$parameters, lr = lr)
scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                          epochs = num_epochs, steps_per_epoch = length(train_dl),
                          cycle_momentum = FALSE)

gamma_x <- 0.1
mseloss <- 0.1#gamma_x^2
loggamma <- log(gamma_x)

for (epoch in 1:num_epochs) {

    batchnum <- 0
    coro::loop(for (b in train_dl) {

        batchnum <- batchnum + 1
        optimizer$zero_grad()
        c(reconstruction, recon_miss, input, missing, mean, log_var, m_mean, m_logvar) %<-% vae(b$gb$cuda(), b$ob$cuda())
        c(loss, reconstruction_loss, kl_loss, recon_loss_missing, kl_missing) %<-%
          vae$loss_function(reconstruction, input,
                            recon_miss, missing,
                            mean, log_var,
                            m_mean, m_logvar,
                            torch_tensor(loggamma, device = "cuda")
                            )

        mseloss <- min(mseloss, mseloss * .99 + as.numeric(reconstruction_loss$cpu() / vae$input_dim) * .01)
        gamma_x <- sqrt(mseloss)
        loggamma <- log(gamma_x)

        if(batchnum %% 2 == 0) {

            cat("Epoch: ", epoch,
                "    batch: ", batchnum,
                "    loss: ", as.numeric(loss$cpu()),
                "    recon loss: ", as.numeric(reconstruction_loss$cpu()),
                "    recon missing loss: ", as.numeric(recon_loss_missing$cpu()),
                "    KL loss: ", as.numeric(kl_loss$cpu()),
                "    KL missing loss: ", as.numeric(kl_missing$cpu()),
                "    loggamma: ", as.numeric(loggamma),
                #"    loggamma: ", loggamma,
                "    active dims: ", as.numeric((torch_exp(log_var)$mean(dim = 1L) < 0.5)$sum()$cpu()),
                "\n")

        }
        loss$backward()

        # if((epoch / num_epochs) > 0.6) {
        #   nn_utils_clip_grad_norm_(vae$parameters, 1)
        # }

        optimizer$step()
        scheduler$step()
    })
}

torch_save(vae, "data/grambank_vae_v2.to")

post_dl <- dataloader(train_ds, 1200, shuffle = FALSE)
post_it <- as_iterator(post_dl)

all_mean_var <- list()
it <- 0
loop(for(i in post_dl) {
  it <- it + 1
  with_no_grad({
    all_mean_var[[it]] <- vae$encoder(((i$gb$cuda() * 2) - 1) )
  })
  print(it)
})

all_means <- map(all_mean_var, 1) %>%
  map(~ as.matrix(.x$cpu())) %>%
  do.call(rbind, .)

all_vars <- map(all_mean_var, 2) %>%
  map(~ as.matrix(torch_exp(.x)$cpu())) %>%
  do.call(rbind, .)

vars <- apply(all_vars, 2, mean)

vars_df <- tibble(`Mean Variance` = vars) %>%
  mutate(Status = ifelse(`Mean Variance` < 0.5, "Active", "Ignored"))

ggplot(vars_df, aes(`Mean Variance`)) +
  geom_histogram(aes(fill = Status)) +
  ylab("Count") +
  theme_minimal() +
  theme(legend.position = c(0.5, 0.5))

active_dims <- which(vars < 0.5)
plot(all_means[ , active_dims][ , 1:2])

######### Save latent codes ################

active_dims <- which(vars < 0.5)
latent_means <- all_means[ , active_dims]
latent_vars <- all_vars[ , active_dims]

colnames(latent_means) <- colnames(latent_vars) <- paste0("latent_", 1:ncol(latent_means))

latent_df <- as.data.frame(latent_means) %>%
  mutate(Language_ID = lang_ids) %>%
  pivot_longer(-Language_ID, names_to = "latent_dim", values_to = "latent_mean") %>%
  left_join(as.data.frame(latent_vars) %>%
              mutate(Language_ID = lang_ids) %>%
              pivot_longer(-Language_ID, names_to = "latent_dim", values_to = "latent_var")) %>%
  mutate(latent_lower = latent_mean - 1.96 * sqrt(latent_var),
         latent_upper = latent_mean + 1.96 * sqrt(latent_var))

write_csv(latent_df, "data/grambank_vae_latent_codes_v2.csv")
latent_df_means <- as.data.frame(latent_means) %>%
  mutate(Language_ID = lang_ids) %>%
  select(Language_ID, everything())

write_csv(latent_df_means, "data/grambank_vae_latent_codes_means_only_v2.csv")

######### Some tests #############
test_code <- torch_randn(1000, 64)
test_preds <- as.matrix(vae$decoder(test_code$cuda())$cpu())
image(t(test_preds))
image(t(grambank_mat))
image(cbind(t(grambank_mat), t(test_preds)))

## interpolation
active_dims <- which(vars < 0.5)
latents <- all_means[ , active_dims]
min_1_lang <- which.min(latents[ , 1])
max_1_lang <- which.max(latents[ , 1])
lang_names <- lang_ids[c(min_1_lang, max_1_lang)]
min_1_vec <- rep(0, ncol(all_means))
min_1_vec[active_dims] <- latents[min_1_lang, ]
max_1_vec <- rep(0, ncol(all_means))
max_1_vec[active_dims] <- latents[max_1_lang, ]
lang_vec <- max_1_vec - min_1_vec
lang_interp <- sapply(seq(0, 1, length.out = 100),
                      function(x) min_1_vec + x * lang_vec)
lang_interp_tens <- torch_tensor(t(lang_interp))$cuda()

gb_codes <- as.matrix(vae$decoder(lang_interp_tens)$cpu())
image(t(gb_codes))

## spherical
theta <- acos( sum(min_1_vec*max_1_vec) / ( sqrt(sum(min_1_vec * min_1_vec)) * sqrt(sum(max_1_vec * max_1_vec)) ) )
lang_interp_sp <- sapply(seq(0, 1, length.out = 100),
                         function(x) ((sin((1-x) * theta)) / sin(theta)) * min_1_vec +
                           (sin(x * theta) / sin(theta)) * max_1_vec)

lang_interp_sp_tens <- torch_tensor(t(lang_interp_sp))$cuda()

gb_codes_sp <- as.matrix(vae$decoder(lang_interp_sp_tens)$cpu())
image(t(gb_codes_sp))

## compare with interpolation in original feature space
min_1_feat <- grambank_mat[min_1_lang, ]
max_1_feat <- grambank_mat[max_1_lang, ]
lang_vec_feat <- max_1_feat - min_1_feat

lang_interp_feat <- sapply(seq(0, 1, length.out = 100),
                           function(x) min_1_feat + x * lang_vec_feat)

image(lang_interp_feat)


######## Estimate reconstruction error with only active dimensions ########
zeros <- matrix(0, nrow = nrow(latent_means), ncol = ncol(latent_means))


######## Setup VAE stage 2 ##############
## It looks like we don't need this as the 14 active dimensions of
## the first stage VAE look very Gaussian and well behaved?
## I might try it later just to see what happens..
