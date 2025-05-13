from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, set_default_openai_client
from agents.run import RunConfig
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
import chainlit as cl ,os ,time ,logging,pdfplumber

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Load environment variables
load_dotenv()

# Get API keys
gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Validate API keys
if not gemini_api_key or not pinecone_api_key:
    raise ValueError("API keys for Gemini and Pinecone must be set in the .env file.")

# Set up Gemini model
set_tracing_disabled(disabled=True)
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

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = 'zia-pdf-index'
namespace = 'default'

# Create or connect to Pinecone index
if index_name not in [index['name'] for index in pc.list_indexes().get('indexes', [])]:
    logging.info(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    time.sleep(10)  # Wait for index initialization
index = pc.Index(index_name)
logging.info("Successfully connected to Pinecone index")

# Load embedding model
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
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return "Error: Could not read the PDF file."

async def process_pdf(file_path):
    """
    Processes a PDF file, generates embeddings, and stores them in Pinecone.
    """
    content = read_pdf(file_path)
    if content.startswith("Error"):
        return content

    paragraphs = [p for p in content.split('\n\n') if p.strip()]
    if not paragraphs:
        return "Error: No text found in the PDF."

    embeddings = embedding_model.encode(paragraphs)
    data = [
        (str(i), emb.tolist(), {'text': para})
        for i, (emb, para) in enumerate(zip(embeddings, paragraphs))
    ]

    # Clear old data
    index.delete(delete_all=True, namespace=namespace)
    # Store new data
    index.upsert(vectors=data, namespace=namespace)
    return "PDF processed and data stored in Pinecone."

@cl.on_chat_start
async def start():
    """
    Handles the start of a chat session, prompting for and processing a PDF upload.
    """
    await cl.Message(content="Welcome! Please upload a PDF file to begin.").send()

    files = await cl.AskFileMessage(
        content="Upload your PDF file",
        accept=["application/pdf"],
        max_size_mb=10
    ).send()

    if files:
        file = files[0]
        file_path = file.path
        await cl.Message(content=f"Processing PDF: {file.name}").send()

        result = await process_pdf(file_path)
        await cl.Message(content=result).send()
        if "Error" not in result:
            await cl.Message(content="PDF processed successfully. You can now ask questions.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    """
    Handles user queries by searching Pinecone and generating a response.
    """
    question = message.content
    question_embedding = embedding_model.encode([question])[0].tolist()

    # Search Pinecone
    results = index.query(
        vector=question_embedding,
        top_k=5,
        include_metadata=True,
        namespace=namespace
    )

    # Extract matching paragraphs
    matching_paragraphs = [
        res['metadata']['text']
        for res in results.get('matches', [])
        if res['metadata'].get('text')
    ]
    search_results = '\n\n'.join(matching_paragraphs)

    # Prepare agent instructions
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

    # Set up and run the agent
    agent = Agent(
        name="Assistant",
        instructions=instructions,
        model=model
    )
    result = Runner.run_streamed(agent, question, run_config=config)
    # Create an initial empty message
    message = cl.Message(content="")
    await message.send()
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            # Update the existing message by appending the delta
            message.content += event.data.delta
            await message.update()  # Assumes cl.Message has an update method

if __name__ == "__main__":
    cl.run()
