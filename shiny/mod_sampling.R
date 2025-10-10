# shiny/mod_sampling.R
mod_sampling_ui <- function(id) {
  ns <- NS(id)
  tagList(
    h3("Sampling Options"),
    selectInput(
      ns("sampler"), "Choose a sampler:",
      choices = c("Random"="random", "Distance Midpoint"="distance_mid", "Co-occurrence"="cooccur", "Hard Negative"="hard_negative")
    ),
    numericInput(ns("n_pairs"), "Number of pairs to sample:", value = 10, min = 1, step = 1),
    conditionalPanel(
      sprintf("input['%s'] == 'distance_mid'", ns("sampler")),
      numericInput(ns("target"), "Target similarity:", value = 0.5, min = 0, max = 1, step = 0.05),
      numericInput(ns("tol"),    "Tolerance:", value = 0.05, min = 0, max = 1, step = 0.01)
    ),
    conditionalPanel(
      sprintf("input['%s'] == 'cooccur'", ns("sampler")),
      numericInput(ns("max_sim"),    "Max similarity:", value = 0.9, min = 0, max = 1, step = 0.05),
      numericInput(ns("min_shared"), "Minimum shared docs:", value = 1, min = 0, step = 1)
    ),
    conditionalPanel(
      sprintf("input['%s'] == 'hard_negative'", ns("sampler")),
      numericInput(ns("min_sim"),         "Minimum similarity:", value = 0.7, min = 0, max = 1, step = 0.05),
      numericInput(ns("max_shared_docs"), "Max shared docs:", value = 0, min = 0, step = 1)
    ),
    shinyjs::disabled(actionButton(ns("sample"), "Sample Pairs"))
  )
}

mod_sampling_server <- function(id, base, rater) {
  moduleServer(id, function(input, output, session) {
    sampled_df <- reactiveVal(NULL)
    
    # Gate the button
    observe({
      if (isTRUE(base$data_ready()) && isTRUE(rater$rater_loaded())) {
        shinyjs::enable("sample")
      } else {
        shinyjs::disable("sample")
      }
    })
    
    # Defensive: reassert rater text when clicking Sample (avoids UI resets)
    observeEvent(input$sample, {
      rater$reset_text()
    }, ignoreInit = TRUE, priority = 100)
    
    observeEvent(input$sample, {
      req(isTRUE(base$data_ready()), isTRUE(rater$rater_loaded()))
      kw_sub <- base$kw_sub(); kids <- base$kids()
      if (length(kids) < 2) { showNotification("Not enough active keywords to sample.", type="error"); return() }
      
      #idx <- match(kids, base$kids_universe)
      sim_active <- base$kw_sim_full[kids, kids, drop = FALSE]
      C_active   <- base$C_full[kids,      kids,  drop = FALSE]
      
      want <- input$n_pairs
      upk  <- rater$used_pair_keys()  # NULL -> skip used-checks
      
      idx_pairs <-
        if (input$sampler == "random") {
          sample_pair_random_n(n_kw = length(kids), n = want, kids = kids, used_pair_keys = upk)
        } else if (input$sampler == "distance_mid") {
          sample_pair_distance_mid_n(sim = sim_active, n = want, kids = kids,
                                     target = input$target, tol = input$tol, used_pair_keys = upk)
        } else if (input$sampler == "cooccur") {
          sample_pair_cooccur_sparse_n(n = want, C = C_active, sim = sim_active, kids = kids,
                                       min_shared = input$min_shared, max_sim = input$max_sim, used_pair_keys = upk)
        } else if (input$sampler == "hard_negative") {
          sample_pair_hard_negative_sparse_n(n = want, C = C_active, sim = sim_active, kids = kids,
                                             min_sim = input$min_sim, max_shared = input$max_shared_docs, used_pair_keys = upk)
        } else {
          matrix(NA_integer_, 0, 2)
        }
      
      if (!is.matrix(idx_pairs) || nrow(idx_pairs) == 0) {
        showNotification("No eligible pairs found. Try different parameters.", type="warning")
        sampled_df(NULL); return()
      }
      
      phrase <- kw_sub$phrase
      df <- tibble(
        `keyword 1`  = phrase[idx_pairs[, 1]],
        `omit 1`     = FALSE,
        `method 1`   = FALSE,
        `thematic 1` = FALSE,
        `keyword 2`  = phrase[idx_pairs[, 2]],
        `omit 2`     = FALSE,
        `method 2`   = FALSE,
        `thematic 2` = FALSE,
        value        = NA_real_
      )
      sampled_df(df)
    })
    
    list(sampled_df = sampled_df)
  })
}
