setwd("/scratch/jhowe4/data")
BiocManager::install("TCGAbiolinks")

library(TCGAbiolinks)

query <- GDCquery(
  project = "TCGA-LUAD", 
  data.category = "Biospecimen",
  data.type = "Slide Image"
)

print("Downloading...")
# Download a list of barcodes with platform IlluminaHiSeq_RNASeqV2
GDCdownload(query, method="api")
