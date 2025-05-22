import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,set_tracing_disabled,set_default_openai_client
from agents.run import RunConfig
import pdfplumber  # To read PDF files
from sentence_transformers import SentenceTransformer  # To create text embeddings
from pinecone import Pinecone, ServerlessSpec  # To use the vector database
import time  # To add a delay
import asyncio  # To run the agent
from dotenv import load_dotenv



load_dotenv()


gemini_api_key = os.getenv("GEMINI_API_KEY")

set_tracing_disabled(disabled=True)

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

set_default_openai_client(external_client)


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Get your Pinecone API key from Colab secrets
PINECONE_API_KEY = 'pcsk_SgNxq_HgttwRS6tafMaeDREj6noCfhMmFzxdXJ3C5BPVowVtaG79XLLh7rtg3nk6nNiuo'

# Validate and clean the API key
def validate_pinecone_api_key(api_key):
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set or contains only whitespace.")
    
    # Trim whitespace and validate format
    api_key = api_key.strip()
    
    if not api_key.startswith('pcsk_'):
        raise ValueError("Invalid Pinecone API key format. Must start with 'pcsk_'.")
    
    # Ensure the key is complete
    if len(api_key) < 50:
        raise ValueError("Pinecone API key seems incomplete.")
    
    return api_key

try:
    # Validate and clean the API key
    PINECONE_API_KEY = validate_pinecone_api_key(PINECONE_API_KEY)
    logging.info(f"Validated Pinecone API Key: {PINECONE_API_KEY[:5]}...{PINECONE_API_KEY[-5:]}")

    # Connect to Pinecone
    from pinecone import Pinecone, ServerlessSpec

    # Initialize Pinecone with robust error handling
    def initialize_pinecone(api_key):
        try:
            logging.info("Attempting to initialize Pinecone...")
            pc = Pinecone(api_key=api_key)
            
            # Verify connection by listing indexes
            logging.info("Attempting to list indexes...")
            indexes = pc.list_indexes()
            logging.info(f"Existing indexes: {indexes}")
            
            return pc
        except Exception as e:
            logging.error(f"Pinecone Initialization Error: {e}")
            raise

    # Initialize Pinecone
    pc = initialize_pinecone(PINECONE_API_KEY)

    # Define the index name
    index_name = 'hasnain-pdf-index'


    # Check if the index exists, create if not
    try:
        # Check if index exists
        existing_indexes = pc.list_indexes()
        index_exists = any(index['name'] == index_name for index in existing_indexes.get('indexes', []))
        
        if not index_exists:
            logging.info(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=384,  # Dimension of sentence transformer model
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            logging.info(f"Index {index_name} created successfully")
        else:
            logging.info(f"Index {index_name} already exists")

        # Connect to the index
        index = pc.Index(index_name)
        logging.info("Successfully connected to Pinecone index")

    except Exception as e:
        logging.error(f"Error creating/connecting to Pinecone index: {e}")
        raise

except Exception as e:
    logging.error(f"Critical Error: {e}")
    raise

# Load the model to create embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def read_pdf(file_path):
    """
    Reads text from a PDF file.
    Returns the text or an error message.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n\n'
            return text
    except:
        return "Error: Could not read the PDF file."

async def main(file_path):
    """
    Reads the PDF, stores its content in Pinecone, and answers a question.
    """
    # Step 1: Read the PDF
    print("Reading the PDF file...")
    content = read_pdf(file_path)
    if content.startswith("Error"):
        print(content)
        return

    # Step 2: Split text into paragraphs
    paragraphs = [p for p in content.split('\n\n') if p.strip()]
    if not paragraphs:
        print("Error: No text found in the PDF.")
        return

    # Step 3: Create embeddings for each paragraph
    print("Creating embeddings...")
    embeddings = embedding_model.encode(paragraphs)

    # Step 4: Prepare data for Pinecone
    data = [
        (str(i), emb.tolist(), {'text': para})
        for i, (emb, para) in enumerate(zip(embeddings, paragraphs))
    ]

    # Step 5: Clear old data in the index
    print("Clearing old data in Pinecone...")
    try:
        index.delete(delete_all=True, namespace='default')
    except Exception as e:
        if "Namespace not found" in str(e):
            print("No existing data to clear.")
        else:
            raise e

    # Step 6: Store new data in Pinecone
    print("Storing data in Pinecone...")
    index.upsert(vectors=data, namespace='default')

    # Step 7: Define the question
    question = "What Job Hasnain Ali Can Do?"

    # Step 8: Create an embedding for the question
    print("Searching for relevant information...")
    question_embedding = embedding_model.encode([question])[0].tolist()

    # Step 9: Search Pinecone for similar paragraphs
    results = index.query(
        vector=question_embedding,
        top_k=5,  # Get top 5 matches
        include_metadata=True,
        namespace='default'  # Using 'default' namespace explicitly
    )

    # Step 10: Get the matching paragraphs
    matching_paragraphs = [
        res['metadata']['text']
        for res in results.get('matches', [])
        if res['metadata'].get('text')
    ]

    # Step 11: Combine the matches
    search_results = '\n\n'.join(matching_paragraphs)

    # Step 12: Prepare instructions for the AI agent
    if matching_paragraphs:
        instructions = (
            f"You are a helpful researcher. Here is some information from a PDF about '{question}':\n"
            f"{search_results}\n"
            "Use this to answer the question clearly."
        )
    else:
        instructions = (
            f"You are a helpful researcher. No information was found in the PDF about '{question}'. "
            "Please tell the user no details were found."
        )

    # Step 13: Set up the AI agent
    agent = Agent(
        name="Assistant",
        instructions=instructions,
        model=model  # Make sure 'model' is defined
    )

    # Step 14: Ask the agent to answer the question
    print("Getting the answer...")
    result = await Runner.run(agent, question)
    print("\nAnswer:")
    print(result.final_output)

# Run the program
if __name__ == "__main__":
    # Set the path to your PDF file
    file_path = 'data.pdf'  # Change this to your PDF's path
    asyncio.run(main(file_path))