library(shiny)
library(jsonlite)
library(shinythemes)

# Path to your JSON file
json_path <- "Meta Llama 4 Maverick 17B-128E-Instruct-FP8.json"  # Change as needed

ui <- fluidPage(
  titlePanel("Question Answer App"),
  tabsetPanel(
    tabPanel(
      "Question Comparator",
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
  all_data <- fromJSON(json_path)
  if (is.list(all_data) && !is.data.frame(all_data)) all_data <- as.data.frame(all_data)
  
  # DYNAMIC SELECTOR UI
  output$category_selector <- renderUI({
    selectInput("selected_category", "Select Psychiatric Category:",
                choices = unique(all_data$psychiatric_category),
                selected = unique(all_data$psychiatric_category)[1]
    )
  })
  
  # FILTERED DATA BASED ON SELECTED CATEGORY
  filtered_data <- reactive({
    req(input$selected_category)
    subset(all_data, psychiatric_category == input$selected_category)
  })
  
  # INITIALLY SELECT RANDOM ROW FROM FILTERED DATA
  selected_index <- reactiveVal(1)
  observeEvent(filtered_data(), {
    if(nrow(filtered_data()) > 0)
      selected_index(sample(nrow(filtered_data()), 1))
    else
      selected_index(NA)
  })
  
  observeEvent(input$randomize, {
    if(nrow(filtered_data()) > 0)
      selected_index(sample(nrow(filtered_data()), 1))
    else
      selected_index(NA)
  })
  
  output$qa_ui <- renderUI({
    i <- selected_index()
    dat <- filtered_data()
    if (is.na(i) || nrow(dat) == 0) return(h4("No questions in this category."))
    dat <- dat[i, ]
    
    # Calculate mean relevancy scores
    mean_vanilla_relevancy <- mean(c(
      dat$`answer_relevancy for Vanilla run 1`,
      dat$`answer_relevancy for Vanilla run 2`,
      dat$`answer_relevancy for Vanilla run 3`
    ), na.rm = TRUE)
    
    mean_rag_relevancy <- mean(c(
      dat$`answer_relevancy for RAG run 1`,
      dat$`answer_relevancy for RAG run 2`,
      dat$`answer_relevancy for RAG run 3`
    ), na.rm = TRUE)
    
    tagList(
      fluidRow(
        column(
          width = 12,
          h4(strong("Question:")),
          h5(dat$`Modified Question`)
        ),
        column(
          width = 12,
          h4(strong("Reference Answer:")),
          h5(dat$Reasonings)
        ),
        column(
          width = 6,
          wellPanel(
            h4("Generated Vanilla Answer"),
            dat$`Generated Vanilla Answer`
          )
        ),
        column(
          width = 6,
          wellPanel(
            h4("Generated RAG Answer"),
            dat$`Generated RAG Answer`
          )
        )
      ),
      hr(),
      fluidRow(
        column(
          width = 6,
          wellPanel(
            h4("Metrics: Vanilla"),
            p(strong("Semantic Similarity:"), round(dat$`Answer Semantic Similarity for vanilla`, 2)),
            p(strong("Rubric Score:"), round(dat$`Vanilla Rubric Score`, 2)),
            p(strong("Answer Relevancy (mean):"), round(mean_vanilla_relevancy, 2)),
            p("Run 1:", round(dat$`answer_relevancy for Vanilla run 1`, 2)),
            p("Run 2:", round(dat$`answer_relevancy for Vanilla run 2`, 2)),
            p("Run 3:", round(dat$`answer_relevancy for Vanilla run 3`, 2))
          )
        ),
        column(
          width = 6,
          wellPanel(
            h4("Metrics: RAG"),
            p(strong("Semantic Similarity:"), round(dat$`Answer Semantic Similarity for rag`, 2)),
            p(strong("Rubric Score:"), round(dat$`RAG Rubric Score`, 2)),
            p(strong("Answer Relevancy (mean):"), round(mean_rag_relevancy, 2)),
            p("Run 1:", round(dat$`answer_relevancy for RAG run 1`, 2)),
            p("Run 2:", round(dat$`answer_relevancy for RAG run 2`, 2)),
            p("Run 3:", round(dat$`answer_relevancy for RAG run 3`, 2))
          )
        )
      )
    )
  })
}

      


shinyApp(ui, server)
