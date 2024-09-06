from fastapi import FastAPI, UploadFile, File, HTTPException
import sqlite3
import io
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust according to your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Generative AI API key
GOOGLE_API_KEY = 'AIzaSyAmyK2-L52gXmIdHbaY5ZwQxaouJaLalBM'
genai.configure(api_key=GOOGLE_API_KEY)

logging.basicConfig(level=logging.INFO)

def create_survey_db():
    conn = sqlite3.connect('survey.db')
    cursor = conn.cursor()

    logging.info("Creating tables...")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS User (
        user_id INTEGER,
        survey_id INTEGER,
        UNIQUE(user_id, survey_id)  -- Enforce uniqueness on user_id and survey_id
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Survey (
        survey_id INTEGER,
        question_id INTEGER,
        question TEXT,
        UNIQUE(survey_id, question_id)  -- Enforce uniqueness on survey_id and question_id
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Responses (
        user_id INTEGER,
        survey_id INTEGER,
        question_id INTEGER,
        response TEXT,
        UNIQUE(user_id, survey_id, question_id)  -- Enforce uniqueness on user_id, survey_id, and question_id
    );
    """)

    # Insert initial data
    cursor.execute("INSERT OR IGNORE INTO User VALUES (1, 101)")
    cursor.execute("INSERT OR IGNORE INTO User VALUES (2, 101)")
    cursor.execute("INSERT OR IGNORE INTO User VALUES (3, 102)")
    cursor.execute("INSERT OR IGNORE INTO User VALUES (4, 102)")
    cursor.execute("INSERT OR IGNORE INTO User VALUES (5, 103)")

    cursor.execute("INSERT OR IGNORE INTO Survey VALUES (101, 1, 'How satisfied are you with our service?')")
    cursor.execute("INSERT OR IGNORE INTO Survey VALUES (101, 2, 'Would you recommend our service to others?')")
    cursor.execute("INSERT OR IGNORE INTO Survey VALUES (102, 3, 'How easy was it to use the product?')")
    cursor.execute("INSERT OR IGNORE INTO Survey VALUES (102, 4, 'How likely are you to use the product again?')")

    cursor.execute("INSERT OR IGNORE INTO Responses VALUES (1, 101, 1, 'Very satisfied')")
    cursor.execute("INSERT OR IGNORE INTO Responses VALUES (1, 101, 2, 'Yes')")
    cursor.execute("INSERT OR IGNORE INTO Responses VALUES (2, 101, 1, 'Satisfied')")
    cursor.execute("INSERT OR IGNORE INTO Responses VALUES (2, 101, 2, 'Maybe')")
    cursor.execute("INSERT OR IGNORE INTO Responses VALUES (3, 102, 3, 'Easy')")
    cursor.execute("INSERT OR IGNORE INTO Responses VALUES (3, 102, 4, 'Very likely')")
    cursor.execute("INSERT OR IGNORE INTO Responses VALUES (4, 102, 3, 'Difficult')")
    cursor.execute("INSERT OR IGNORE INTO Responses VALUES (4, 102, 4, 'Unlikely')")
    cursor.execute("INSERT OR IGNORE INTO Responses VALUES (5, 103, 1, 'Neutral')")

    conn.commit()
    conn.close()

    logging.info("Database setup complete.")


# Initialize database when the app starts
@app.on_event("startup")
async def startup_event():
    create_survey_db()


def check_db():
    conn = sqlite3.connect('survey.db')
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print("Tables in the database:", tables)

    conn.close()


check_db()

# Function to extract text from uploaded PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf_name, pdf_bytes in pdf_docs.items():
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to process text chunks and create vector store
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Endpoint to upload and process PDF files
@app.post("/upload_pdf/")
async def upload_pdf(files: list[UploadFile] = File(...)):
    pdf_docs = {file.filename: await file.read() for file in files}
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    return {"message": "PDF processing done. You can now ask questions."}

# Function to get conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user queries
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

# Endpoint to ask questions from the processed PDF
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask_pdf/")
async def ask_pdf(request: QuestionRequest):
    try:
        question = request.question
        response = user_input(question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Function to query the SQLite database
def read_sql_query(sql):
    conn = sqlite3.connect("survey.db")
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()
    return rows

# Endpoint to query the SQLite database
class QuestionRequest(BaseModel):
    question: str

@app.post("/query_db/")
async def query_db(request: QuestionRequest):
    prompt = [
        """
        You are an expert in converting English questions to SQL queries!
        You understand the english test very well, what exact parameter is asked and can effifciently convert into SQL query
        The SQL database has three tables with the following columns:

        - `User` table: 
            - `user_id` INTEGER
            - `survey_id` INTEGER

        - `Survey` table: 
            - `survey_id` INTEGER
            - `question_id` INTEGER
            - `question` TEXT

        - `Responses` table: 
            - `user_id` INTEGER
            - `survey_id` INTEGER
            - `question_id` INTEGER
            - `response` TEXT

        \nFor example:
        - How many users have participated in survey 101?
          SQL: SELECT COUNT(*) FROM User WHERE survey_id=101;

        - List all the questions for survey 102
          SQL: SELECT question FROM Survey WHERE survey_id=102;

        - What are the responses given by user 1 for survey 101?
          SQL: SELECT response FROM Responses WHERE user_id=1 AND survey_id=101;

        Ensure that the SQL code does not include additional words or the word 'sql'.
        """
    ]
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt[0], request.question])
    sql_query = response.text.strip()

    # Debugging: Print generated SQL query
    print(f"Generated SQL Query: {sql_query}")

    if not sql_query:
        raise HTTPException(status_code=400, detail="Generated SQL query is empty or invalid.")

    try:
        result = read_sql_query(sql_query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing SQL query: {str(e)}")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)