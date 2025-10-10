# mod_preview.R — Abstract preview module (keyword → modal with highlighted abstracts)
# -----------------------------------------------------------------------------------
# Purpose:
#   Provides UI + server logic to preview an example abstract for a selected keyword.
#   Opens a modal and highlights literal occurrences of the chosen phrase.
#
# Consumes (from `base`, the list returned by load_base_data()):
#   - base$kw_sub()      : reactive data.frame with at least `phrase` (chr) and
#                          `award_ids` (list-column of character vectors).
#   - base$hegs_df       : data.frame with `award_id` (chr) and `abstract_clean` (chr).
#
# UI elements:
#   - selectizeInput(id = "kw_for_preview")  : keyword picker (choices track base$kw_sub()).
#   - actionButton(id = "preview_abs")       : opens the modal.
#   - Modal shows: title "Award <id> — <keyword>" and HTML with <mark> highlights.
#
# Outputs / side effects:
#   - Calls showModal() to render the preview.
#   - Calls showNotification() on edge cases (no abstracts, keyword missing, etc.).
#
# Behavior:
#   - Keeps the selectize choices in sync with base$kw_sub() (reactive).
#   - Only award_ids present in base$hegs_df$award_id are shown.
#   - Wraps case-insensitive literal matches of the phrase with <mark>.
#
# Assumptions:
#   - Column `abstract_clean` exists in base$hegs_df. If your data uses `abstract`,
#     swap in `abstract` inside the server code.
#   - `highlight_phrase()` is available (from shiny/utils.R).
#
# Dependencies:
#   - shiny, stringr, dplyr (already loaded upstream).
#
# Example wiring (in app.R):
#   mod_preview_ui("preview")
#   mod_preview_server("preview", base)
# -----------------------------------------------------------------------------------
mod_preview_ui <- function(id) {
  ns <- NS(id)
  tagList(
    h3("Abstract Preview"),
    selectizeInput(ns("kw_for_preview"), "Select a keyword:", choices = NULL, selected = NULL,
                   options = list(placeholder = "Type to search…")),
    actionButton(ns("preview_abs"), "Show example abstract")
  )
}

mod_preview_server <- function(id, base) {
  moduleServer(id, function(input, output, session) {
    ns <- session$ns
    modal_state <- reactiveValues(ids = character(0), idx = 1, kw = NULL)
    
    # Keep choices in sync with active set
    observe({
      kw_sub <- base$kw_sub()
      updateSelectizeInput(session, "kw_for_preview",
                           choices = kw_sub$phrase,
                           selected = NULL, server = TRUE)
    })
    
    output$modal_title <- renderText({
      req(length(modal_state$ids) >= 1, modal_state$kw, modal_state$idx)
      cur_id <- modal_state$ids[modal_state$idx]
      paste0("Award ", cur_id, " — ", modal_state$kw)
    })
    
    output$abs_html <- renderUI({
      req(length(modal_state$ids) >= 1, modal_state$idx)
      hegs_df <- base$hegs_df
      cur_id <- modal_state$ids[modal_state$idx]
      abs_row <- hegs_df %>%
        dplyr::filter(.data$award_id == cur_id) %>%
        dplyr::transmute(award_id,
                         abstract = stringr::str_replace_all(abstract_clean, "-", " "))
      if (!nrow(abs_row) || is.na(abs_row$abstract[1]) || !nzchar(abs_row$abstract[1])) {
        return(HTML("<em>Abstract not found.</em>"))
      }
      HTML(highlight_phrase(abs_row$abstract[1], modal_state$kw))
    })
    
    observeEvent(input$preview_abs, {
      kw_sub <- base$kw_sub()
      req(!is.null(kw_sub), !is.null(input$kw_for_preview), nrow(kw_sub) > 0)
      row_idx <- which(kw_sub$phrase == input$kw_for_preview)
      if (length(row_idx) != 1) { showNotification("Keyword not found in active set.", type="error"); return() }
      
      hegs_df <- base$hegs_df
      ids <- kw_sub$award_ids[[row_idx]]
      ids <- ids[!is.na(ids) & nzchar(ids)]
      ids <- intersect(ids, hegs_df$award_id)
      if (!length(ids)) { showNotification("No abstracts found for this keyword.", type="warning"); return() }
      
      modal_state$ids <- ids
      modal_state$idx <- 1
      modal_state$kw  <- input$kw_for_preview
      
      showModal(modalDialog(
        title = textOutput(ns("modal_title")),
        easyClose = TRUE, size = "l",
        footer = tagList(
          actionButton(ns("prev_abs"), "← Prev"),
          uiOutput(ns("pos_indicator")),
          actionButton(ns("next_abs"), "Next →"),
          modalButton("Close")
        ),
        div(style = "max-height: 60vh; overflow-y: auto;",
            htmlOutput(ns("abs_html")))
      ))
    })
    
    output$pos_indicator <- renderUI({
      req(length(modal_state$ids) >= 1)
      span(style = "margin: 0 10px;", paste0(modal_state$idx, " of ", length(modal_state$ids)))
    })
    
    observeEvent(input$next_abs, {
      req(length(modal_state$ids) >= 1)
      n <- length(modal_state$ids)
      modal_state$idx <- (modal_state$idx %% n) + 1
    })
    observeEvent(input$prev_abs, {
      req(length(modal_state$ids) >= 1)
      n <- length(modal_state$ids)
      modal_state$idx <- ((modal_state$idx - 2) %% n) + 1
    })
  })
}
