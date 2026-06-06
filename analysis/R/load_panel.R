load_panel <- function() {
  path <- here::here("..", "data", "processed", "master_panel_r.csv")
  panel <- read.csv(path, stringsAsFactors = FALSE)
  panel$year <- as.integer(panel$year)
  panel
}
