# Computed ICC based on this link: https://www.datanovia.com/en/lessons/intraclass-correlation-coefficient-in-r/

# On Compute Canada, need to specify a mirror, but for other platforms you
# may not have to

# install.packages("irr", repos="https://cloud.r-project.org")
# install.packages("readxl", repos="https://cloud.r-project.org")
install.packages("stringr", repos="https://cloud.r-project.org")


library("irr")
library("readxl")
library("stringr")

#' Filter a matrix based on the ICC
#' 
#' @description
#' `filterRow` returns a boolean value representing whether the ICC is greater than 0.75
#' 
#' @details 
#' This function takes an nxm matrix, with n subjects and m raters. It will 
#' calculate ICC of the matrix. As set in (Yang & Tsao, 2022), An ICC of 
#' greater than 0.75 is considered good, and thus the filter will return 
#' TRUE, otherwise, it will return false.
#' 
#' @param df the nxm matrix to be scored
#' 
#' @returns the filter decision of the matrix, either true (substantial 
#' correlation), or false (unsubstantial correlation)
#' 
#' @examples 
#' 
#' df <- data.frame("class" = c("invasive", "probable_invasive", "probable_noninvasive", "noninvasive"),
#'                  "Tsao" = c(100, 0, 0, 0),
#'                  "Yang" = c(50, 20, 30, 0),
#'                  "MRC" = c(80, 0, 0 ,20),
#'                  "Najd" = c(90, 0, 0, 10))
#' 
#' filterRow(df[,2:5])

filterRow <- function(df) {
  icc_value <- as.numeric(icc(df, 
                              model = "twoway", 
                              type = "agreement", 
                              unit = "single", 
                              conf.level = 0.95)$value)
  ret_val = FALSE
  if (icc_value >= 0.75) {
    ret_val <- TRUE
  } 
  return(ret_val)
}

# Read in all four scorers' excel files
tsaoData <- read_excel("../inputs/CK7 study_database_rescoring_final_TSAOv2.xlsx", 
                        sheet = "CK7",
                        col_names = c(
                          "case",
                          "invasive", 
                          "probable_invasive", 
                          "probable_noninvasive", 
                          "noninvasive",
                          "simple_architecture",
                          "complex_architecture",
                          "single_cell_invasion",
                          "review"),
                        col_types = c(
                          "text", 
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "text"),
                        na = '0',
                        skip = 1)

eyData <- read_excel("../inputs/CK7 study_database_rescoring_final_EY.xlsx", 
                        sheet = "CK7",
                        col_names = c(
                          "case",
                          "invasive", 
                          "probable_invasive", 
                          "probable_noninvasive", 
                          "noninvasive",
                          "simple_architecture",
                          "complex_architecture",
                          "single_cell_invasion",
                          "review",
                          "notes"),
                        col_types = c(
                          "text", 
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "text",
                          "text"),
                        na = '0',
                        skip = 1)

mrcData <- read_excel("../inputs/CK7 study_database_rescoring_final_MRCv2.xlsx", 
                        sheet = "CK7",
                        col_names = c(
                          "case",
                          "invasive", 
                          "probable_invasive", 
                          "probable_noninvasive", 
                          "noninvasive",
                          "simple_architecture",
                          "complex_architecture",
                          "single_cell_invasion",
                          "review"),
                        col_types = c(
                          "text", 
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "text"),
                        na = '0',
                        skip = 1)

najdData <- read_excel("../inputs/CK7 study_database_rescoring_final-Najd.xlsx", 
                        sheet = "CK7",
                        col_names = c(
                          "case",
                          "invasive", 
                          "probable_invasive", 
                          "probable_noninvasive", 
                          "noninvasive",
                          "simple_architecture",
                          "complex_architecture",
                          "single_cell_invasion"),
                        col_types = c(
                          "text", 
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric",
                          "numeric"),
                        na = '0',
                        skip = 1)

# Create an indicator vector representing whether the case is used as a gold standard
# 0: not used
# 1: used as a positive case
# 2: used as a negative case
ind <- integer(108)

# Set the current case number to 1, current row number to 1, if we're inside of a 
# continued case to False, and the current vector of results, each entry in the 
# vector representing a result of a subcase
case <- 1
row <- 1
continuedCase <- FALSE
resVec <- c()

while (row <= nrows(tsaoData)) {
  # Dr. Tsao's file wrote that case 46 had faint staining and didn't score it, so we 
  # remove this case (row 67)
  # Dr. Yang's file wrote that case 21 had 2 lesions (omit), so we remove this case
  # (row 32)
  if (row != 67 && row != 32 && row == 1) {
    # If the current case is part of a mult-slide case
    if (is.na(as.numeric(tsaoData[row,]$'case'))) {
      # remove the last character in the case, thus removing the part identifier 
      # i.e. ('108A' -> '108')
      subCaseNum <- str_sub(tsaoData[row,]$'case', end = -2)

      # If we're in the same case as we were last loop iteration
      if (as.numeric(subCaseNum) == case) {
        continuedCase <- TRUE
      } else {
        # We're in a different case, one with multiple parts
        continuedCase <- FALSE
        case <- case + 1
      }
    } else {
      # We are in a single slide case, thus a new case
      case <- case + 1
      continuedCase <- FALSE
    }

    # Create a data frame that only includes the four scoring columns
    df <- data.frame(
      "class" = c("invasive", "probable_invasive", "probable_noninvasive", "noninvasive"),
      "Tsao" = c(
        tsaoData[row,]$'invasive',
        tsaoData[row,]$'probable_invasive',
        tsaoData[row,]$'probable_noninvasive',
        tsaoData[row,]$'noninvasive'),
      "EY" = c(
        eyData[row,]$'invasive',
        eyData[row,]$'probable_invasive',
        eyData[row,]$'probable_noninvasive',
        eyData[row,]$'noninvasive'),
      "MRC" = c(
        mrcData[row,]$'invasive',
        mrcData[row,]$'probable_invasive',
        mrcData[row,]$'probable_noninvasive',
        mrcData[row,]$'noninvasive'),
      "Najd" = c(
        najdData[row,]$'invasive',
        najdData[row,]$'probable_invasive',
        najdData[row,]$'probable_noninvasive',
        najdData[row,]$'noninvasive')
    )
    df[is.na(df)] <- 0

    # compute icc and filter based off newly created data frame
    filterRes <- filterRow(df[,2:5])

    # If we're in a continued case, we should check if we're at the end. If we are, we
    # can base our conclusion off a previously stored vector
    if !(continuedCase) {
      # When we hit a new case, we update the value of the indicator vector in the 
      # previous case. All values in resVec (however long it is) must be true for the
      # case to be used as a gold standard for the dataset
      flag <- FALSE
      for (i in 1:length(resVec)) {
        # If we find a false element, then we can't use that case, and assign it a 0
        if !(resVec[i]) {
          ind[case - 1] <- 0
        }
      }

      # if we didn't find a false element, we can use this case as a gold standard case
      if (!flag) {
        ind[case - 1] <- 1
      }

      # Now that we have updated the previous result, reset the result vector to nothing
      resVec <- c()
    }

    # Add filter result to the results vector
    resVec <- cbind(resVec, c(filterRes))
  }

  # Update loop variable
  row <- row + 1
}

# Update the indication for the last loop
flag <- FALSE
for (i in 1:length(resVec)) {
  # If we find a false element, then we can't use that case, and assign it a 0
  if !(resVec[i]) {
    ind[case] <- 0
  }
}

# if we didn't find a false element, we can use this case as a gold standard case
if (!flag) {
  ind[case] <- 1
}