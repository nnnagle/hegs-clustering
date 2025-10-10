# app.R
library(shiny)
library(shinyjs)
library(rhandsontable)
library(tidyverse)
library(arrow)
library(Matrix)
library(DBI)
library(RSQLite)

source("keyword_samplers.R")
source("../shiny/utils.R")
source("../shiny/base_data.R")
source("../shiny/mod_rater.R")
source("../shiny/mod_sampling.R")
source("../shiny/mod_pairs_table.R")
source("../shiny/mod_preview.R")

pairs_file <- "../data/interim/kw_pairs.csv"

ui <- fluidPage(
  useShinyjs(),
  titlePanel("Keyword Pair Labeling Interface"),
  tags$head(tags$style(HTML("
    mark { background-color: #008000 !important; color: #fff; }
  "))),
  sidebarLayout(
    sidebarPanel(
      width = 3,
      mod_rater_ui("rater"),
      tags$hr(),
      mod_sampling_ui("sampling"),
      div(style = "border-top: 2px solid #e5e7eb; margin-top: 18px; padding-top: 14px;"),
      mod_preview_ui("preview")
    ),
    mainPanel(
      width = 9,
      mod_pairs_table_ui("pairs")
    )
  )
)

server <- function(input, output, session) {
  # Load base data once
  base <- load_base_data()
  
  # Modules
  rater <- mod_rater_server("rater", base, pairs_file)
  sampling <- mod_sampling_server("sampling", base, rater)
  mod_pairs_table_server("pairs", base, rater, pairs_file, sampled_df_reactive = sampling$sampled_df)
  mod_preview_server("preview", base)
}

shinyApp(ui, server)
