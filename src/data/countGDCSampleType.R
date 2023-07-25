#' A script that counts the number of primary tumour vs. solid tissue normal for each type of TCGA project.
#' 
#' Date Created: July 24, 2023
#' Last Updated: July 24, 2023

setwd("../../data/external/gdc_manifest")
BiocManager::install("TCGAbiolinks")
library(TCGAbiolinks)
library(purrr)

# Function to extract the first element of a filename corresponding to a TCGA project case.
getID <- function(str) {
  elements <- strsplit(str, ".", fixed=TRUE)
  return(elements[[1]][1])
}

# List all the manifest files in the current directory.
manifest_files = list.files()

# Find all TCGA project names in the manifest and store the barcodes corresponding to each.
for (i in 1:length(manifest_files)) {
  df <- read.table(manifest_files[i], sep="\t")
  files <- df$V2
  barcodes <- map(files, getID)

  # Find normal solid tissue 
  normal <- TCGAquery_SampleTypes(
    barcode=barcodes,
    typesample=c("NT")
  )

  # Find normal solid tissue 
  tumour <- TCGAquery_SampleTypes(
    barcode=barcodes,
    typesample=c("TP")
  )

  # Get the name of the TCGA project
  projectName <- strsplit(manifest_files[i], "_")[[1]][3]
  projectName <- strsplit(projectName, ".", fixed=TRUE)[[1]][1]
  
  # Print out stats
  print(paste("Project:", toupper(projectName), "# Tumour:", length(tumour), "# Normal:", length(normal)))
}