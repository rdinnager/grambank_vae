library(tidyverse)
library(VoxR)
library(rgl)
library(torch)
library(ggtree)
library(gifski)
library(phangorn)
library(phyf)
library(patchwork)
library(Polychrome)
library(wesanderson)

z_tree_df <- read_rds("data/gb_vae_aces_3dim.rds")
gb_edge_trajs <- read_rds("data/gb_edge_trajs_3dim.rds")
gb_pf <- read_rds("data/gb_pf_3dim.rds")
options(torch.serialization_version = 2)
vae_3 <- torch_load("data/grambank_vae_v2_3dim.to")
vae_3 <- vae_3$cuda()

z_tree_df <- z_tree_df %>%
  rowwise() %>%
  mutate(ace_latent_1 = tail(z_seqs$latent_1, 1),
         ace_latent_2 = tail(z_seqs$latent_2, 1),
         ace_latent_3 = tail(z_seqs$latent_3, 1))

tree_segs <- z_tree_df %>%
  select(label = edge, z_seqs) %>%
  left_join(gb_edge_trajs %>% select(label = end,
                                     start_time, end_time)) %>%
  rowwise() %>%
  mutate(z_seqs = list(z_seqs %>%
                         mutate(time = seq(start_time, end_time, length.out = 100))
                       )) %>%
  unnest(z_seqs)


z_dataset <- dataset(name = "z_ds",
                     initialize = function(z) {
                       self$z <- torch_tensor(z)
                     },
                     .getbatch = function(i) {
                       self$z[i, ]
                     },
                     .length = function() {
                       self$z$size()[[1]]
                     })

z_ds <- z_dataset(tree_segs %>%
                        select(latent_1, latent_2, latent_3) %>%
                        as.matrix())
z_dl <- dataloader(z_ds, 1240, shuffle = FALSE)

x_anc <- list()
i <- 0
coro::loop(for (b in z_dl) {
  i <- i + 1
  x_anc[[i]] <- vae_3$decoder(b$cuda())$cpu()
  print(i)
})

x_ancs <- torch_cat(x_anc) %>%
  as.matrix() %>%
  as.data.frame()


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

colnames(x_ancs) <- colnames(grambank)


tree_segs <- tree_segs %>%
  bind_cols(x_ancs)

gb_clust <- hclust(dist(t(as.matrix(grambank)),  "manhattan"), "ward.D2")
gb_ord <- colnames(grambank)[gb_clust$order]

gb_tree <- pf_as_phylo(gb_pf)
tree_dat <- fortify(pf_as_phylo(gb_pf))

tree_segs <- tree_segs %>%
  left_join(tree_dat %>%
              select(label, x, y))
gb_fact <- colnames(tree_segs %>%
                      select(starts_with("GB"))) %>%
  as.factor()
nam <- as.character(gb_fact)
#gb_fact <- reorder(gb_fact, parse_number(nam))
gb_fact <- factor(gb_fact, levels = gb_ord)
gb_fact <- as.numeric(gb_fact)
names(gb_fact) <- nam

write_rds(gb_fact, "data/GB_order.rds")

gb_tip_y <- tree_dat %>%
  filter(isTip) %>%
  select(label, y)

write_rds(gb_tip_y, "data/gb_tips_order.rds")

tree_end <- max(tree_dat$x)
max_y <- max(tree_dat$y)
plot_timeslice <- function(timeslice) {
  tree_p <- ggtree(tree_dat) + coord_flip() + scale_x_reverse(expand = c(0, 0)) +
    annotate("rect", xmin = timeslice, xmax = tree_end, ymin = 0, ymax = max_y,
             alpha = 0.75, fill = "white") +
    scale_y_continuous(expand = c(0, 0)) +
    theme_tree() +
    theme(plot.margin = unit(c(0, 0, 0, 0), "cm"))
  tree_p
  sliced_dat <- tree_segs %>%
    ungroup() %>%
    filter(time <= timeslice) %>%
    group_by(label) %>%
    slice_max(time, n = 1) %>%
    select(label, y, starts_with("GB")) %>%
    pivot_longer(c(-label, -y), names_to = "feature", values_to = "value") %>%
    mutate(x = gb_fact[feature])
  gb_plot <- ggplot(sliced_dat, aes(y = x, x = y)) +
    #geom_point(aes(colour = value)) +
    geom_tile(aes(fill = value, colour = value), width = 0.3) +
    #scale_x_reverse() +
    scale_x_continuous(expand = c(0, 0), limits = c(1, max_y)) +
    scale_y_continuous(expand = c(0, 0)) +
    theme_tree() +
    theme(plot.margin = unit(c(0, 0, 0, 0), "cm"),
          legend.position = "none")
  #gb_plot

  p <- tree_p + gb_plot + plot_layout(nrow = 2, heights = c(0.3, 0.7))
  p
}

plot_timeslice(100)

times <- c(tree_end - (exp(-seq(0,5,length.out = 200)) * tree_end), tree_end)

save_here <- "presentation/animations/ancestral_recon_thin"
ii <- 0
for(i in times) {
  ii <- ii + 1
  p_name <- file.path(save_here, paste0("timeslice_", str_pad(ii, 3, pad = "0"),
                                        "_", round(i, 2),
                                        ".png"))
  p <- plot_timeslice(i)
  ragg::agg_png(p_name, width = 1200, height = 800, bitsize = 16, scaling = 2)
  plot(p)
  dev.off()
  print(ii)
}

gif_1 <- gifski(list.files(save_here, full.names = TRUE),
                "presentation/animations/anc_anim_thin_1.gif",
                width = 1200, height = 800)

utils::browseURL(gif_1)


tree_end <- max(tree_dat$x)
max_y <- max(tree_dat$y)
plot_timeslice_block <- function(timeslice, discrete = FALSE) {
  tree_p <- ggtree(tree_dat) + coord_flip() + scale_x_reverse(expand = c(0, 0)) +
    annotate("rect", xmin = timeslice, xmax = tree_end, ymin = 0, ymax = max_y,
             alpha = 0.75, fill = "white") +
    scale_y_continuous(expand = c(0, 0)) +
    theme_tree() +
    theme(plot.margin = unit(c(0, 0, 0, 0), "cm"))
  #tree_p
  sliced_dat <- tree_segs %>%
    ungroup() %>%
    filter(time <= timeslice) %>%
    group_by(label) %>%
    slice_max(time, n = 1) %>%
    select(label, y, starts_with("GB")) %>%
    pivot_longer(c(-label, -y), names_to = "feature", values_to = "value") %>%
    mutate(x = gb_fact[feature])

  if(discrete) {
    sliced_dat <- sliced_dat %>%
      ungroup() %>%
      mutate(value = ifelse(value < 0.5, 0, 1))
  }

  node_names <- unique(sliced_dat$label)
  n_tips <- Ntip(gb_tree)
  nodes_pres <- match(node_names, c(gb_tree$tip.label, gb_tree$node.label))
  names(nodes_pres) <- node_names
  desc <- purrr::map(na.omit(nodes_pres[nodes_pres > n_tips]), ~ Descendants(gb_tree, .x, "all"))

  desc_df <- purrr::imap(desc,
                        ~ tibble(desc_nodes = .x) %>%
                          mutate(anc = .y)) %>%
    list_rbind() %>%
    unnest(desc_nodes) %>%
    bind_rows(tibble(anc = c(gb_tree$tip.label, gb_tree$node.label)[nodes_pres[nodes_pres <= n_tips]],
                     desc_nodes = nodes_pres[nodes_pres <= n_tips])) %>%
    group_by(anc) %>%
    mutate(is_anc = any(desc_nodes %in% nodes_pres[nodes_pres > n_tips])) %>%
    filter(!is_anc)

  node_names <- unique(desc_df$anc)
  n_tips <- Ntip(gb_tree)
  nodes_pres <- match(node_names, c(gb_tree$tip.label, gb_tree$node.label))
  names(nodes_pres) <- node_names
  desc <- purrr::map(na.omit(nodes_pres[nodes_pres > n_tips]), ~ Descendants(gb_tree, .x, "tips"))

  desc_df <- purrr::imap(desc,
                        ~ tibble(desc_nodes = .x) %>%
                          mutate(anc = .y)) %>%
    list_rbind() %>%
    unnest(desc_nodes) %>%
    bind_rows(tibble(anc = c(gb_tree$tip.label, gb_tree$node.label)[nodes_pres[nodes_pres <= n_tips]],
                     desc_nodes = nodes_pres[nodes_pres <= n_tips])) %>%
    mutate(desc = gb_tree$tip.label[desc_nodes]) %>%
    left_join(sliced_dat %>% select(label, feature, value, x),
              by = c("anc" = "label")) %>%
    left_join(tree_dat %>%
                filter(isTip) %>%
                select(label, y),
              by = c("desc" = "label"))

  gb_plot <- ggplot(desc_df, aes(y = x, x = y)) +
    #geom_point(aes(colour = value)) +
    geom_tile(aes(fill = value, colour = value)) +
    #scale_x_reverse() +
    scale_x_continuous(expand = c(0, 0), limits = c(1, max_y)) +
    scale_y_continuous(expand = c(0, 0)) +
    theme_tree() +
    theme(plot.margin = unit(c(0, 0, 0, 0), "cm"),
          legend.position = "none")
  #gb_plot

  p <- tree_p + gb_plot + plot_layout(nrow = 2, heights = c(0.3, 0.7))
  p
}

plot_timeslice_block(50, discrete = TRUE)

times <- c(tree_end - (exp(-seq(0,5,length.out = 200)) * tree_end), tree_end)

save_here <- "presentation/animations/ancestral_recon_fat"
ii <- 0
for(i in times) {
  ii <- ii + 1
  p_name <- file.path(save_here, paste0("timeslice_", str_pad(ii, 3, pad = "0"),
                                        "_", round(i, 2),
                                        ".png"))
  suppressWarnings(p <- plot_timeslice_block(i))
  ragg::agg_png(p_name, width = 1200, height = 800, bitsize = 16, scaling = 2)
  plot(p)
  dev.off()
  print(ii)
}

gif_1 <- gifski(list.files(save_here, full.names = TRUE),
                "presentation/animations/anc_anim_fat_1.gif",
                width = 1200, height = 800)

utils::browseURL(gif_1)


times <- c(tree_end - (exp(-seq(0,5,length.out = 200)) * tree_end), tree_end)

save_here <- "presentation/animations/ancestral_recon_fat_binary"
ii <- 0
for(i in times) {
  ii <- ii + 1
  p_name <- file.path(save_here, paste0("timeslice_", str_pad(ii, 3, pad = "0"),
                                        "_", round(i, 2),
                                        ".png"))
  suppressWarnings(p <- plot_timeslice_block(i, discrete = TRUE))
  ragg::agg_png(p_name, width = 1200, height = 800, bitsize = 16, scaling = 2)
  plot(p)
  dev.off()
  print(ii)
}

gif_1 <- gifski(list.files(save_here, full.names = TRUE),
                "presentation/animations/anc_anim_fat_binary_1.gif",
                width = 1200, height = 800)

utils::browseURL(gif_1)



############# linear model version #################

gb_edge_trajs_lin <- read_rds("data/gb_edge_trajs_linear_3dim.rds")
tree_segs_lin <- gb_edge_trajs_lin %>%
  select(label = end, starts_with("GB"), start_time, end_time) %>%
  pivot_longer(starts_with("GB"), names_to = c("feature", NA, "end"),
               names_sep = "_", values_to = "val") %>%
  pivot_wider(names_from = end, values_from = val) %>%
  rowwise() %>%
  mutate(v_seq = list(seq(from = start, to = end, length.out = 100)),
         t_seq = list(seq(from = start_time, to = end_time, length.out = 100))) %>%
  select(-start, -end, -start_time, -end_time) %>%
  unnest(c(v_seq, t_seq))

tree_segs_lin <- tree_segs_lin %>%
  left_join(tree_dat %>%
              select(label, x, y))


tree_end <- max(tree_dat$x)
max_y <- max(tree_dat$y)
plot_timeslice_linear <- function(timeslice, discrete = FALSE) {
  tree_p <- ggtree(tree_dat) + coord_flip() + scale_x_reverse(expand = c(0, 0)) +
    annotate("rect", xmin = timeslice, xmax = tree_end, ymin = 0, ymax = max_y,
             alpha = 0.75, fill = "white") +
    scale_y_continuous(expand = c(0, 0)) +
    theme_tree() +
    theme(plot.margin = unit(c(0, 0, 0, 0), "cm"))
  #tree_p
  sliced_dat2 <- tree_segs_lin %>%
    ungroup() %>%
    filter(t_seq <= timeslice) %>%
    group_by(label) %>%
    slice_max(t_seq, n = 1) %>%
    mutate(x = gb_fact[feature]) %>%
    rename(value = v_seq, time = t_seq)

  if(discrete) {
    sliced_dat2 <- sliced_dat2 %>%
      ungroup() %>%
      mutate(value = ifelse(value < 0.5, 0, 1))
  }

  node_names <- unique(sliced_dat2$label)
  n_tips <- Ntip(gb_tree)
  nodes_pres <- match(node_names, c(gb_tree$tip.label, gb_tree$node.label))
  names(nodes_pres) <- node_names
  desc <- purrr::map(na.omit(nodes_pres[nodes_pres > n_tips]), ~ Descendants(gb_tree, .x, "all"))

  desc_df <- purrr::imap(desc,
                        ~ tibble(desc_nodes = .x) %>%
                          mutate(anc = .y)) %>%
    list_rbind() %>%
    unnest(desc_nodes) %>%
    bind_rows(tibble(anc = c(gb_tree$tip.label, gb_tree$node.label)[nodes_pres[nodes_pres <= n_tips]],
                     desc_nodes = nodes_pres[nodes_pres <= n_tips])) %>%
    group_by(anc) %>%
    mutate(is_anc = any(desc_nodes %in% nodes_pres[nodes_pres > n_tips])) %>%
    filter(!is_anc)

  node_names <- unique(desc_df$anc)
  n_tips <- Ntip(gb_tree)
  nodes_pres <- match(node_names, c(gb_tree$tip.label, gb_tree$node.label))
  names(nodes_pres) <- node_names
  desc <- purrr::map(na.omit(nodes_pres[nodes_pres > n_tips]), ~ Descendants(gb_tree, .x, "tips"))

  desc_df <- purrr::imap(desc,
                        ~ tibble(desc_nodes = .x) %>%
                          mutate(anc = .y)) %>%
    list_rbind() %>%
    unnest(desc_nodes) %>%
    bind_rows(tibble(anc = c(gb_tree$tip.label, gb_tree$node.label)[nodes_pres[nodes_pres <= n_tips]],
                     desc_nodes = nodes_pres[nodes_pres <= n_tips])) %>%
    mutate(desc = gb_tree$tip.label[desc_nodes]) %>%
    left_join(sliced_dat2 %>% select(label, feature, value, x),
              by = c("anc" = "label")) %>%
    left_join(tree_dat %>%
                filter(isTip) %>%
                select(label, y),
              by = c("desc" = "label"))

  gb_plot <- ggplot(desc_df, aes(y = x, x = y)) +
    #geom_point(aes(colour = value)) +
    geom_tile(aes(fill = value, colour = value)) +
    #scale_x_reverse() +
    scale_x_continuous(expand = c(0, 0), limits = c(1, max_y)) +
    scale_y_continuous(expand = c(0, 0)) +
    theme_tree() +
    theme(plot.margin = unit(c(0, 0, 0, 0), "cm"),
          legend.position = "none")
  #gb_plot

  p <- tree_p + gb_plot + plot_layout(nrow = 2, heights = c(0.3, 0.7))
  p
}

plot_timeslice_linear(50)

times <- c(tree_end - (exp(-seq(0,5,length.out = 200)) * tree_end), tree_end)

save_here <- "presentation/animations/ancestral_recon_fat_linear"
ii <- 0
for(i in times) {
  ii <- ii + 1
  p_name <- file.path(save_here, paste0("timeslice_", str_pad(ii, 3, pad = "0"),
                                        "_", round(i, 2),
                                        ".png"))
  suppressWarnings(p <- plot_timeslice_linear(i))
  ragg::agg_png(p_name, width = 1200, height = 800, bitsize = 16, scaling = 2)
  plot(p)
  dev.off()
  print(ii)
}

gif_1 <- gifski(list.files(save_here, full.names = TRUE),
                "presentation/animations/anc_anim_fat_linear_1.gif",
                width = 1200, height = 800)

utils::browseURL(gif_1)


## binary

times <- c(tree_end - (exp(-seq(0,5,length.out = 200)) * tree_end), tree_end)

save_here <- "presentation/animations/ancestral_recon_fat_linear_binary"
ii <- 0
for(i in times) {
  ii <- ii + 1
  p_name <- file.path(save_here, paste0("timeslice_", str_pad(ii, 3, pad = "0"),
                                        "_", round(i, 2),
                                        ".png"))
  suppressWarnings(p <- plot_timeslice_linear(i, discrete = TRUE))
  ragg::agg_png(p_name, width = 1200, height = 800, bitsize = 16, scaling = 2)
  plot(p)
  dev.off()
  print(ii)
}

gif_1 <- gifski(list.files(save_here, full.names = TRUE),
                "presentation/animations/anc_anim_fat_linear_binary_1.gif",
                width = 1200, height = 800)

utils::browseURL(gif_1)



####### rates ###############

write_rds(tree_segs, "data/manifold_tree_segs.rds")
write_rds(tree_segs_lin, "data/linear_tree_segs.rds")

lin_rates <- read_csv("data/init_rates_linear_3dim.csv")

all_rates <- tree_segs %>%
  group_by(label) %>%
  mutate(time_diff = time - lag(time, order_by = time),
         across(starts_with("GB"), ~ .x - lag(.x, order_by = time))) %>%
  summarise(across(starts_with("GB"), ~ sum(.x / time_diff, na.rm = TRUE)))

all_rates_sum <- all_rates %>%
  rowwise() %>%
  mutate(sum = sum(abs(c_across(starts_with("GB")))))

all_rates_sum <- all_rates_sum %>%
  left_join(lin_rates %>%
              rowwise() %>%
              mutate(sum = sum(abs(c_across(starts_with("GB"))))) %>%
              select(label, sum_lin = sum))

all_rates_sum <- all_rates_sum %>%
  select(label, sum, sum_lin)

all_rates_sum <- all_rates_sum %>%
  mutate(sum = sum / 100)

gb_tree_rates <- gb_tree
rate_df <- gb_pf %>%
  left_join(all_rates_sum)
rate_df <- tibble(label = c(gb_tree$tip.label, gb_tree$node.label)[gb_tree$edge[ , 2]]) %>%
  left_join(rate_df)
gb_tree_rates$edge.length <- rate_df$sum

plot(gb_tree_rates)

gb_tree_rates_df <- fortify(gb_tree_rates) %>%
  left_join(lang_dat, by = c("label" = "Glottocode"))

set.seed(1234564535)

fams <- z_tree_df %>%
  filter(is_tip) %>%
  group_by(Family_name) %>%
  mutate(count = n()) %>%
  mutate(Family_name = ifelse(count < 10, NA, Family_name))

fam_pal <- createPalette(n_distinct(na.omit(fams$Family_name)),
                         wes_palettes$FantasticFox1[5])

names(fam_pal) <- unique(na.omit(fams$Family_name))

p1 <- ggtree(gb_tree_rates_df) +
  geom_tippoint(aes(colour = Family_name)) +
  scale_color_manual(values = fam_pal) +
  ggtitle("Manifold Evolutionary Rates") +
  #scale_x_continuous(trans = "log1p", breaks = c(0, 25, 50, 100, 250, 500, 1000)) +
  theme_tree2()

# ggtree(gb_tree_rates_df, layout = "circular") +
#   geom_tippoint(aes(colour = Family_name)) +
#   scale_color_manual(values = fam_pal) +
#   scale_x_continuous(trans = "log1p", breaks = c(0, 25, 50, 100, 250, 500, 1000))
#   #theme_tree2()



gb_tree_rates_lin <- gb_tree
gb_tree_rates_lin$edge.length <- rate_df$sum_lin

plot(gb_tree_rates_lin)

gb_tree_rates_df_lin <- fortify(gb_tree_rates_lin) %>%
  left_join(lang_dat, by = c("label" = "Glottocode"))

p2 <- ggtree(gb_tree_rates_df_lin) +
  geom_tippoint(aes(colour = Family_name)) +
  scale_color_manual(values = fam_pal) +
  ggtitle("Linear Evolutionary Rates") +
  #scale_x_continuous(trans = "log1p", breaks = c(0, 25, 50, 100, 250, 500, 1000)) +
  theme_tree2()

p1 + p2 + plot_layout(nrow = 2, guides = "collect")

all_rates_sum <- all_rates_sum %>%
  ungroup() %>%
  mutate(sum_st = sum / mean(sum),
         sum_lin_st = sum_lin / mean(sum_lin))

ggplot(all_rates_sum, aes(sum_lin_st, sum_st)) +
  geom_point() +
  scale_x_continuous(trans = "log1p") +
  scale_y_continuous(trans = "log1p") +
  geom_abline(slope = 1, intercept = 0) +
  ylab("Manifold Evolutionary Rate (mean standardised)") +
  xlab("Linear Evolutionary Rate (mean standardised)") +
  coord_equal() +
  theme_minimal()
