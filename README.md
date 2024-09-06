# Task_backend

## Overview

This FastAPI backend provides endpoints for uploading and processing PDF files, querying an SQLite database, and handling survey data. It utilizes Google Generative AI for text processing and SQLite for storing survey data.

## Features

- **Upload PDF Files**: Upload PDF files, extract text, and create a vector store for querying.
- **Ask Questions**: Query the processed text from PDFs using a conversational AI model.
- **Database Querying**: Convert natural language questions into SQL queries and fetch results from an SQLite database.

## Installation

1. **Clone the repository**:
2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

4. Install dependencies:
   
    ```bash
    pip install -r requirements.txt

## Configurations:

1. Google Generative AI API Key: Replace the placeholder in app.py with your actual API key
2. Database Setup: The SQLite database is automatically set up when the FastAPI app starts.

## Run the application:

The application will be accessible at http://localhost:8000.

## API Endpoints:

1. Upload PDF Files: POST /upload_pdf/
2. Ask Questions: POST /ask_pdf/
3. Query Database: POST /query_db/









    
      

    
   

  
