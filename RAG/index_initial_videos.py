import sys
from rag_engine_vid import RAGEngine

def main():
    urls = [
        "https://youtu.be/pSVk-5WemQ0",
        "https://youtu.be/2TJxpyO3ei4",
        "https://youtu.be/Bx9BBG3jQCQ"
    ]
    
    print("Initializing RAG Engine for pre-population...")
    try:
        engine = RAGEngine(db_path="./chroma_db", embedding_model_name="all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error initializing RAG Engine: {e}")
        sys.exit(1)
        
    print("\nStarting indexing of initial videos...")
    for idx, url in enumerate(urls, 1):
        print(f"\n[{idx}/{len(urls)}] Processing URL: {url}")
        try:
            result = engine.index_video(url)
            print(f"Success! Title: '{result['title']}' | Author: {result['author']} | Chunks Saved: {result['chunks_count']}")
        except Exception as e:
            print(f"Failed to index {url}: {e}")
            
    print("\nPre-population check finished.")

if __name__ == "__main__":
    main()
