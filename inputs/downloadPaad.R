#' An R script that pulls example TCGA data from GDC using the bioconductor package in R
#' 
#' Date Created: June 5, 2023
#' Last Updated: June 5, 2023
#' Author: Alex Turco

setwd("/scratch/jhowe4/data")
BiocManager::install("TCGAbiolinks")

library(TCGAbiolinks)

query <- GDCquery(
  project = "TCGA-PAAD", 
  data.category = "Biospecimen",
  data.type = "Slide Image",
  experimental.strategy = "Tissue Slide"
)

print("Downloading...")

# Download whole slide images from TCGA for Pancreatic adenocarcinoma
GDCdownload(query, method="api")
