# shiny/mod_rater.R
mod_rater_ui <- function(id) {
  ns <- NS(id)
  tagList(
    textInput(ns("rater"), "Enter your initials here", value = "", placeholder = "e.g., AB"),
    actionButton(ns("enter"), "Enter rater / refresh")
  )
}

mod_rater_server <- function(id, base, pairs_file) {
  moduleServer(id, function(input, output, session) {
    entered_rater <- reactiveVal("")
    rater_loaded  <- reactiveVal(FALSE)
    used_pair_keys <- reactiveVal(NULL)
    kp_all  <- reactiveVal(NULL)
    kp_by_r <- reactiveVal(NULL)
    
    current_rater_input <- reactive(norm_init(input$rater))
    
    # If text changes, require reload
    observeEvent(input$rater, {
      rater_loaded(FALSE)
    }, ignoreInit = TRUE)
    
    # Enter/refresh
    observeEvent(input$enter, {
      rt <- current_rater_input()
      if (!nzchar(rt)) {
        used_pair_keys(NULL); kp_by_r(NULL); rater_loaded(FALSE)
        showNotification("Enter your initials first.", type = "error"); return()
      }
      
      # Commit normalized text to the box
      entered_rater(rt)
      updateTextInput(session, "rater", value = rt)
      
      if (!file.exists(pairs_file)) {
        kp_all(tibble(kw1=character(), kw2=character(), value=numeric(), rater=character()))
        kp_by_r(NULL); used_pair_keys(NULL); rater_loaded(TRUE)
        showNotification("No kw_pairs.csv yet. Sampling will ignore used-pair checks.", type = "message")
        return()
      }
      
      ok <- TRUE
      kp <- tryCatch(readr::read_csv(pairs_file, show_col_types = FALSE), error = function(e) { ok <<- FALSE; NULL })
      if (!ok) {
        rater_loaded(FALSE); showNotification("Failed to read kw_pairs.csv.", type = "error"); return()
      }
      
      if (!("kw1" %in% names(kp) && "kw2" %in% names(kp))) {
        if (ncol(kp) >= 3) names(kp)[1:3] <- c("kw1","kw2","value")
      }
      if (!("rater" %in% names(kp))) kp$rater <- NA_character_
      kp_all(kp)
      
      kp_r <- kp %>%
        mutate(rater = norm_init(rater)) %>%
        filter(rater == rt, !is.na(kw1), !is.na(kw2))
      
      kp_by_r(kp_r)
      used_pair_keys(compute_used_keys(base$phr_to_kid(), kp_r))
      rater_loaded(TRUE)
      showNotification(paste0("Loaded ", nrow(kp_r), " pairs for rater ", rt, "."), type = "message")
    }, ignoreInit = TRUE)
    
    list(
      entered_rater  = entered_rater,
      rater_loaded   = rater_loaded,
      used_pair_keys = used_pair_keys,
      kp_all         = kp_all,
      kp_by_rater    = kp_by_r,
      reset_text     = function() updateTextInput(session, "rater", value = entered_rater())
    )
  })
}
