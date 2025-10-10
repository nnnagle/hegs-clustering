# shiny/mod_pairs_table.R
mod_pairs_table_ui <- function(id) {
  ns <- NS(id)
  tagList(
    h3("Keyword Pairs to Label"),
    rHandsontableOutput(ns("pair_table")),
    br(),
    fluidRow(
      column(4, actionButton(ns("submit"), "Submit")),
      column(4, actionButton(ns("close"),  "Close"))
    )
  )
}

mod_pairs_table_server <- function(id, base, rater, pairs_file, sampled_df_reactive) {
  moduleServer(id, function(input, output, session) {
    output$pair_table <- renderRHandsontable({
      df <- sampled_df_reactive()
      if (!is.null(df)) rhandsontable(df, useTypes = TRUE, stretchH = "all")
    })
    
    observeEvent(input$close, { stopApp() })
    
    observeEvent(input$submit, {
      shinyjs::disable("submit"); on.exit(shinyjs::enable("submit"), add=TRUE)
      df_edit <- hot_to_r(input$pair_table)
      if (is.null(df_edit) || nrow(df_edit) == 0) return()
      
      rt <- rater$entered_rater()
      if (!nzchar(rt)) { showNotification("Enter your initials before submitting.", type="error"); return() }
      
      # 1) Update keyword metadata (by phrase) in-memory
      keyword_candidates <- base$keyword_candidates()
      for (i in seq_len(nrow(df_edit))) {
        kw1 <- df_edit$`keyword 1`[i]
        kw2 <- df_edit$`keyword 2`[i]
        idx1 <- which(keyword_candidates$phrase == kw1)
        idx2 <- which(keyword_candidates$phrase == kw2)
        if (length(idx1) == 1) {
          if (!is.na(df_edit$`omit 1`[i]))     keyword_candidates$omit[idx1]     <- as.logical(df_edit$`omit 1`[i])
          if (!is.na(df_edit$`method 1`[i]))   keyword_candidates$method[idx1]   <- as.logical(df_edit$`method 1`[i])
          if (!is.na(df_edit$`thematic 1`[i])) keyword_candidates$thematic[idx1] <- as.logical(df_edit$`thematic 1`[i])
        }
        if (length(idx2) == 1) {
          if (!is.na(df_edit$`omit 2`[i]))     keyword_candidates$omit[idx2]     <- as.logical(df_edit$`omit 2`[i])
          if (!is.na(df_edit$`method 2`[i]))   keyword_candidates$method[idx2]   <- as.logical(df_edit$`method 2`[i])
          if (!is.na(df_edit$`thematic 2`[i])) keyword_candidates$thematic[idx2] <- as.logical(df_edit$`thematic 2`[i])
        }
      }
      base$keyword_candidates(keyword_candidates) # write back
      
      # 2) Append labeled rows to kw_pairs.csv (only finite values), with rater column
      new_rows <- df_edit %>%
        dplyr::filter(is.finite(value)) %>%
        dplyr::transmute(kw1 = `keyword 1`, kw2 = `keyword 2`, value, rater = rt)
      
      if (nrow(new_rows)) {
        new_rows <- new_rows %>% dplyr::select(kw1, kw2, value, rater)
        if (!file.exists(pairs_file)) {
          readr::write_csv(new_rows, pairs_file, quote = "all")
          kp <- new_rows
        } else {
          kp <- tryCatch(readr::read_csv(pairs_file, show_col_types = FALSE), error = function(e) NULL)
          if (is.null(kp)) {
            readr::write_csv(new_rows, pairs_file, append = TRUE, quote = "all")
            kp <- new_rows
          } else {
            if (!("kw1" %in% names(kp) && "kw2" %in% names(kp))) {
              if (ncol(kp) >= 3) names(kp)[1:3] <- c("kw1","kw2","value")
            }
            if (!("rater" %in% names(kp))) kp$rater <- NA_character_
            kp <- kp %>% dplyr::select(kw1, kw2, value, rater)
            kp <- dplyr::bind_rows(kp, new_rows)
            readr::write_csv(kp, pairs_file, quote = "all")
          }
        }
        
        # Hot-update used_pair_keys for this rater (no need to re-enter)
        nr_r <- new_rows %>% mutate(rater = norm_init(rater)) %>% filter(rater == rt)
        if (nrow(nr_r)) {
          added <- compute_used_keys(base$phr_to_kid(), nr_r)
          cur <- rater$used_pair_keys(); if (is.null(cur)) cur <- character(0)
          rater$used_pair_keys(unique(c(cur, added)))
        }
      }
      
      # 3) Persist updated flags back to SQLite (batched)
      phr2kid <- base$phr_to_kid()
      kids_touched <- unique(c(df_edit$`keyword 1`, df_edit$`keyword 2`))
      kids_touched <- unname(phr2kid[kids_touched])
      kids_touched <- kids_touched[!is.na(kids_touched)]
      if (length(kids_touched)) {
        upd_df <- base$keyword_candidates() %>%
          filter(kid %in% kids_touched) %>%
          transmute(kid,
                    omit     = as.integer(omit),
                    method   = as.integer(method),
                    thematic = as.integer(thematic))
        tryCatch({
          DBI::dbWithTransaction(base$kc_con, {
            DBI::dbExecute(base$kc_con, "DROP TABLE IF EXISTS tmp_updates")
            DBI::dbWriteTable(base$kc_con, "tmp_updates", upd_df, temporary = TRUE)
            DBI::dbExecute(base$kc_con, "
              UPDATE keyword_candidates
                 SET omit     = (SELECT omit     FROM tmp_updates u WHERE u.kid = keyword_candidates.kid),
                     method   = (SELECT method   FROM tmp_updates u WHERE u.kid = keyword_candidates.kid),
                     thematic = (SELECT thematic FROM tmp_updates u WHERE u.kid = keyword_candidates.kid)
               WHERE kid IN (SELECT kid FROM tmp_updates)
            ")
          })
        }, error = function(e) showNotification(paste("SQLite update failed:", e$message), type="error"))
      }
      
      # 4) Refresh active subset + phrase choices (no recompute of big matrices)
      keyword_candidates <- base$keyword_candidates()
      kid_sub_new <- keyword_candidates %>% filter(!omit) %>% pull(kid)
      if (!identical(kid_sub_new, base$kids())) {
        kw_sub_new <- keyword_candidates %>%
          filter(kid %in% kid_sub_new) %>%
          arrange(match(kid, kid_sub_new))
        base$kids(kid_sub_new)
        base$kw_sub(kw_sub_new)
        # phr_to_kid may change if phrases updated; refresh to be safe
        base$phr_to_kid(setNames(keyword_candidates$kid, keyword_candidates$phrase))
      }
      
      showNotification("Saved.", type="message")
    })
  })
}
