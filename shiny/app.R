library(shiny)
library(shinythemes)

rsconnect::writeManifest()

# Path to your CSV file
csv_path <- "random_questions_by_category.csv"  # Change as needed

ui <- fluidPage(
  titlePanel("Question Answer App"),
  tabsetPanel(
    tabPanel(
      "Question Comparator",
      uiOutput("model_selector"),
      uiOutput("category_selector"),
      actionButton("randomize", "Show Random Question"),
      br(), br(),
      uiOutput("qa_ui")
    ),
    tabPanel(
      "Other Feature",
      h3("Coming Soon!"),
      p("This tab can contain any other analysis or visualization you want.")
    )
  )
)

server <- function(input, output, session) {
  # Load data once at startup
  all_data <- read.csv(csv_path, stringsAsFactors = FALSE)
  if (!is.data.frame(all_data)) {
    stop("CSV did not load as a data.frame. Check the file path and format.")
  }
  
  num <- function(x) suppressWarnings(as.numeric(x))
  
  # MODEL SELECTOR
  output$model_selector <- renderUI({
    models <- unique(na.omit(all_data$model_name))
    selectInput(
      "selected_model",
      "Select Model:",
      choices = models,
      selected = if (length(models)) models[1] else NULL
    )
  })
  
  # CATEGORY SELECTOR (depends on model)
  output$category_selector <- renderUI({
    req(input$selected_model)
    cats <- unique(na.omit(all_data$category[all_data$model_name == input$selected_model]))
    if (length(cats) == 0) cats <- unique(na.omit(all_data$category))
    selectInput(
      "selected_category",
      "Select Psychiatric Category:",
      choices = cats,
      selected = if (length(cats)) cats[1] else NULL
    )
  })
  
  # FILTER
  filtered_data <- reactive({
    req(input$selected_model, input$selected_category)
    subset(all_data, model_name == input$selected_model & category == input$selected_category)
  })
  
  selected_index <- reactiveVal(1)
  observeEvent(filtered_data(), {
    dat <- filtered_data()
    if (nrow(dat) > 0) selected_index(sample(nrow(dat), 1)) else selected_index(NA_integer_)
  })
  
  observeEvent(input$randomize, {
    dat <- filtered_data()
    if (nrow(dat) > 0) selected_index(sample(nrow(dat), 1)) else selected_index(NA_integer_)
  })
  
  output$qa_ui <- renderUI({
    i <- selected_index()
    dat <- filtered_data()
    if (is.na(i) || nrow(dat) == 0) return(h4("No questions matching this model & category."))
    dat <- dat[i, , drop = FALSE]
    
    mean_vanilla_relevancy <- mean(c(
      num(dat$answer_relevancy_vanilla_run_1),
      num(dat$answer_relevancy_vanilla_run_2),
      num(dat$answer_relevancy_vanilla_run_3)
    ), na.rm = TRUE)
    
    mean_rag_relevancy <- mean(c(
      num(dat$answer_relevancy_rag_run_1),
      num(dat$answer_relevancy_rag_run_2),
      num(dat$answer_relevancy_rag_run_3)
    ), na.rm = TRUE)
    
    tagList(
      fluidRow(
        column(
          width = 12,
          h4(strong("Question:")),
          h5(dat$question)
        ),
        column(
          width = 12,
          h4(strong("Reference Answer:")),
          h5(dat$ground_truth)
        ),
        column(
          width = 6,
          wellPanel(
            h4("Generated Vanilla Answer"),
            dat$vanilla_answer
          )
        ),
        column(
          width = 6,
          wellPanel(
            h4("Generated RAG Answer"),
            dat$rag_answer,
            br(),
            tags$hr(),
            h5(strong("Retrieved Context:")),
            dat$rag_context
          )
        )
      ),
      hr(),
      fluidRow(
        column(
          width = 6,
          wellPanel(
            h4("Metrics: Vanilla"),
            p(strong("Semantic Similarity:"), round(num(dat$vanilla_semantic_similarity), 2)),
            p(strong("Rubric Score:"), round(num(dat$vanilla_rubric_score), 2)),
            p(strong("Answer Relevancy (mean):"), round(mean_vanilla_relevancy, 2)),
            p("Run 1:", round(num(dat$answer_relevancy_vanilla_run_1), 2)),
            p("Run 2:", round(num(dat$answer_relevancy_vanilla_run_2), 2)),
            p("Run 3:", round(num(dat$answer_relevancy_vanilla_run_3), 2))
          )
        ),
        column(
          width = 6,
          wellPanel(
            h4("Metrics: RAG"),
            p(strong("Semantic Similarity:"), round(num(dat$rag_semantic_similarity), 2)),
            p(strong("Rubric Score:"), round(num(dat$rag_rubric_score), 2)),
            p(strong("Answer Relevancy (mean):"), round(mean_rag_relevancy, 2)),
            p("Run 1:", round(num(dat$answer_relevancy_rag_run_1), 2)),
            p("Run 2:", round(num(dat$answer_relevancy_rag_run_2), 2)),
            p("Run 3:", round(num(dat$answer_relevancy_rag_run_3), 2))
          )
        )
      )
    )
  })
}

shinyApp(ui, server)
