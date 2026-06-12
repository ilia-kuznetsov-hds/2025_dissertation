script_path <- NULL

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  script_path <- sub("^--file=", "", file_arg[[1]])
} else {
  script_path <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
}

if (!is.null(script_path)) {
  setwd(dirname(normalizePath(script_path)))
}

shiny::runApp(".", launch.browser = TRUE)