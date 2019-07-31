library(dplyr)
weights <- read.csv("/home/shawarma/data/Thyroid/SVS_Test_Set_B/balanced_weights.csv")
case_labels <- unique(weights[,(2:3)])

#weights2 = weights[,-(0:8)]
mean_weights <- aggregate(weights[, -(0:8)], list(weights$Case), mean)
mean_weights <- dplyr::rename(mean_weights, Case = Group.1)
merged <- (merge(case_labels, mean_weights, by = 'Case'))
merged <- merged[order(merged$Category),]
# Remove FTC cases
merged <- merged[!(merged$Category=="FTC"),]
weights2 <- merged[,-(1:2)]

k_clusters <- kmeans(weights2, 5)
merged$KMeansCluster <- as.factor(k_clusters$cluster)

category_labels <- merged[,2]
library(colorspace) # get nice colors
category_col <- rev(rainbow_hcl(5))[as.numeric(category_labels)]

library(parallelDist)
weight_matrix = as.matrix(as.data.frame(weights2))
d_weights <- parDist(weight_matrix, method="euclidean") # method="man" # is a bit better
hc_weights <- hclust(d_weights, method = "complete")
cut_weights <- cutree(hc_weights, k=5)
categories <- rev(levels(merged[,2]))

library(dendextend)
dend <- as.dendrogram(hc_weights)
# order it the closest we can to the order of the observations:
dend <- rotate(dend, 1:48)

# Color the branches based on the clusters:
dend <- color_branches(dend, k=5) #, groupLabels=iris_species)

# Manually match the labels, as much as possible, to the real classification of the flowers:
labels_colors(dend) <-
  rainbow_hcl(5)[sort_levels_values(
    as.numeric(merged[,2])[order.dendrogram(dend)]
  )]

# We shall add the category type to the labels:
labels(dend) <- paste(as.character(merged[,2])[order.dendrogram(dend)],
                      "(",labels(dend),")-",merged[,1539], 
                      sep = "")
# We hang the dendrogram a bit:
dend <- hang.dendrogram(dend,hang_height=0.1)
# reduce the size of the labels:
# dend <- assign_values_to_leaves_nodePar(dend, 0.5, "lab.cex")
dend <- set(dend, "labels_cex", 0.8)
# And plot:
#pdf("dendrogram.pdf", width=400, height=150)

par(mar = c(3,3,3,7))
plot(dend, #cut(dend, h=50)$lower[[2]]
     main = "Thyroid model: Clustering of cases by P-1 features (mean)",  nodePar = list(cex = .007))
#legend("topright", legend = categories, fill = rainbow_hcl(5))
#dev.off()
seeds_df_cl <- mutate(weights2, cluster=cut_weights)
count(seeds_df_cl, cluster)

