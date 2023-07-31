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

# Function to extract the case ID from the longer TCGA case ID
getBarcode <- function(str) {
  elements <- strsplit(str, "-", fixed=TRUE)[[1]]
  return(paste(elements[1:3], collapse="-"))
}

# List all the manifest files in the current directory.
manifest_files = list.files()
projectName <- ""

# Find all TCGA project names in the manifest and store the barcodes corresponding to each.
for (i in 1:length(manifest_files)) {
  # Get the name of the TCGA project
  projectName <- strsplit(manifest_files[i], "_")[[1]][3]
  projectName <- strsplit(projectName, ".", fixed=TRUE)[[1]][1]

  # We want to select only lung adenocarcinoma as it has a good amount of normal tissue as well as primary tumour tissue
  if (toupper(projectName) == "LUAD") {
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

    # Undersample from the primary tumour case to match the number of samples of normal tissue
    tumour <- sample(tumour, size=length(normal))
  
    # Print out stats
    print(paste("Project:", toupper(projectName), "# Tumour:", length(tumour), "# Normal:", length(normal)))

    setwd("/scratch/jhowe4/inputs/GDC/luad_example")
    # Query the combined sample
    combinedSample <- c(tumour, normal)
    combinedSample <- map(combinedSample, getBarcode)
    normalSample <- map(normal, getBarcode)
    query <- GDCquery(
      project="TCGA-LUAD",
      barcode=normalSample,
      data.category="Biospecimen",
      data.type="Slide Image",
      experimental.strategy="Tissue Slide"
    )
    GDCdownload(query, method="api")
  }
}