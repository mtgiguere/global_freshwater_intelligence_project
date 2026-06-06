load_panel <- function(path = NULL) {
  if (is.null(path)) {
    path <- here::here("..", "data", "processed", "master_panel_r.csv")
  }
  panel <- read.csv(path, stringsAsFactors = FALSE)
  panel$year <- as.integer(panel$year)
  panel
}

panel_path <- function() {
  here::here("..", "data", "processed", "master_panel_r.csv")
}
