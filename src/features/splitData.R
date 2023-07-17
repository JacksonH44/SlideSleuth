#' An R script that splits TCGA data based on how the tumour slide image
#' is classified (either solid tissue normal or primary tumour)
#' 
#' Date Created: June 12, 2023
#' Last Updated: June 15, 2023
#' Author: Jackson Howe
 
prevDir <- getwd()
setwd("/scratch/jhowe4/inputs/GDC/paad_example")
BiocManager::install("TCGAbiolinks")

outPath <- "/scratch/jhowe4/outputs/GDC/paad_example_5x/labels.csv"

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
  files <- list.files(path = dir)

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

#' Create a dataframe of GDC data
#' 
#' @description 
#' Generate a data frame consisting of all images and their labels
#' 
#' @details 
#' Looping through the normal barcodes, then the tumour barcodes, find
#' the file with which they're associated with, then store the file name
#' and the classification (0 - normal, 1 - primary tumour) in a dataframe
#' 
#' @param normal vector of normal case barcodes
#' @param tumour vector of primary tumour case barcodes
#' @param dir path to the directory holding all images
#' 
#' @returns a dataframe of the image names and their classification

generateDataFrame <- function(normal, tumour, dir) {
  # Create a vector of all files in the directory
  files <- list.files(path = dir)

  # Create a data frame with file name and tumour presence (1 or 0) 
  # as the columns
  df <- data.frame()

  # Find all files corresponding to normal cases
  for (i in 1:length(normal)) {
    # find the index of the file that matches the barcode
    fileIdx <- match(1, grepl(normal[i], files))

    # Add a new entry to the data frame for a normal case
    entry <- c(files[fileIdx], 0)
    df <- rbind(df, entry)
  }

  # Find all files corresponding to tumour cases
  for (i in 1:length(tumour)) {
    # find the index of the file that matches the barcode
    fileIdx <- match(1, grepl(tumour[i], files))

    # Add a new entry to the data frame for a tumour case
    entry <- c(files[fileIdx], 1)
    df <- rbind(df, entry)
  }

  return(df)
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

# Generate a data frame
df <- generateDataFrame(normal, tumour, getwd())
colnames(df) <- c("file", "class")

# Write the dataframe to a csv
write.csv(df, outPath, row.names = FALSE)