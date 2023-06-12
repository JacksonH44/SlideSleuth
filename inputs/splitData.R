#' An R script that splits TCGA data based on how the tumour slide image
#' is classified (either solid tissue normal or primary tumour)
#' 
#' Date Created: June 12, 2023
#' Last Updated: June 12, 2023
#' Author: Jackson Howe
 
setwd("/scratch/jhowe4/inputs/GDC/paad_example")
BiocManager::install("TCGAbiolinks")

library(TCGAbiolinks)

#' Find all barcodes from TCGA data
#' 
#' @description 
#' 'findBarcodes' returns a vector of all barcodes in a TCGA dataset
#' 
#' @details
#' This function takes in a directory where TCGA images are stored and
#' extracts their TCGA barcode from the file name. It returns a vector of 
#' all the barcodes
#' 
#' @param dir the path to the directory holding all images
#' 
#' @returns the vector of barcodes from the images

findBarcodes <- function(dir) {
  # Create a vector of all files in the directory
  files <- list.files(path=dir)

  # Create a vector of barcodes
  barcodes <- character(length(files))

  for (i in 1:length(files)) {
    inputString <- files[i]

    # Split the string using delimiter "."
    tokens <- strsplit(inputString, ".", fixed = TRUE)[[1]]

    # Extract the first token, the barcode
    first_token <- tokens[1]
    barcodes[i] <- first_token
  }

  return(barcodes)
}

# Generate barcodes
codes <- findBarcodes(getwd())

# Find normal solid tissue 
normal <- TCGAquery_SampleTypes(
  barcode = codes,
  typesample = c("NT")
)

# Find primary tumour
tumour <- TCGAquery_SampleTypes(
  barcode = codes,
  typesample = c("TP")
)

