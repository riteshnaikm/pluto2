import asyncio
import logging
from hypercorn.config import Config
from hypercorn.asyncio import serve
from app import asgi_app, initialize_pinecone, build_bm25_index, setup_llm_chain
import os

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hr_assistant.log')
    ]
)

# Check HR_docs folder
hr_docs_path = "HR_docs/"
if not os.path.exists(hr_docs_path):
    logging.error(f"❌ HR_docs folder not found at: {os.path.abspath(hr_docs_path)}")
    os.makedirs(hr_docs_path)
    logging.info("✅ Created HR_docs folder")
else:
    pdf_files = [f for f in os.listdir(hr_docs_path) if f.endswith('.pdf')]
    logging.info(f"📚 Found {len(pdf_files)} PDF files in HR_docs folder:")
    for pdf in pdf_files:
        logging.info(f"   - {pdf}")

async def main():
    try:
        # Initialize Pinecone
        logging.info("🔧 Initializing Pinecone...")
        initialize_pinecone()
        
        # Build BM25 index
        logging.info("🔍 Building BM25 index...")
        build_bm25_index(hr_docs_path)
        
        # Set up LLM and QA chain
        logging.info("🤖 Setting up LLM and QA chain...")
        setup_llm_chain()
        
        # Start server in production mode
        logging.info("🚀 Starting HR Assistant Suite (Production Mode)...")
        config = Config()
        
        # Bind to all interfaces for production
        config.bind = ["0.0.0.0:5000"]
        
        # Disable reloader in production
        config.use_reloader = False
        
        # Production settings
        config.workers = 4  # Adjust based on your server's CPU cores
        config.accesslog = '-'  # Log to stdout
        config.errorlog = '-'   # Log to stderr
        
        await serve(asgi_app, config)
        
    except Exception as e:
        logging.error(f"❌ Startup error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

