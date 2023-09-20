library(torch)
library(tidyverse)
library(FNN)
library(ape)
library(phyf)

gb_tree <- read.tree("data/EDGE_pruned_tree.tree")

tree_cuts <- read_csv("data/chopped_edges.csv")
tree_cuts <- tree_cuts %>%
  rename(edge = end, position = positions)

# max_len <- max(node.depth.edgelength(gb_tree))
# cuts <- seq(0, max_len, length.out = 250)
gb_pf <- pf_as_pf(gb_tree)

#tree_cuts <- pf_epoch_info(gb_pf$phlo, cuts)
gb_tree_cutup <- pf_edge_segmentize(gb_pf$phlo, tree_cuts$edge, tree_cuts$position)

tips <- pf_labels(gb_tree_cutup)[pf_is_tips(gb_tree_cutup)]
new_tips <- sapply(strsplit(tips, "_"), function(x) x[1])
names(new_tips) <- tips

gb_cutup_tree <- pf_as_phylo(gb_tree_cutup, collapse_singletons = FALSE)

gb_cutup_tree$tip.label[gb_cutup_tree$tip.label %in% tips] <- new_tips[gb_cutup_tree$tip.label[gb_cutup_tree$tip.label %in% tips]]
gb_cutup_tree$node.label[gb_cutup_tree$node.label %in% tips] <- new_tips[gb_cutup_tree$node.label[gb_cutup_tree$node.label %in% tips]]

write.tree(gb_cutup_tree, "data/gc_cutup_tree_custom.tre")
