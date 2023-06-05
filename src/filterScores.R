# Computed ICC based on this link: https://www.datanovia.com/en/lessons/intraclass-correlation-coefficient-in-r/

# On Compute Canada, need to specify a mirror, but for other platforms you
# may not have to
install.packages("irr", repos="https://cloud.r-project.org")

library("irr")

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