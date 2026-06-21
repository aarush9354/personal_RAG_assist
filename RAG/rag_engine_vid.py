import os
import re
import requests
from typing import List, Dict, Any, Tuple, Optional
from youtube_transcript_api import YouTubeTranscriptApi
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker

class SimpleEmbeddings(Embeddings):
    def __init__(self, model: SentenceTransformer):
        self.model = model
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
        
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

class RAGEngine:
    def __init__(self, db_path: str = "./chroma_db", embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the YouTube RAG Engine.
        - db_path: Path to store ChromaDB persistent database.
        - embedding_model_name: The local HuggingFace sentence-transformer model to use.
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        
        # Load local embedding model
        print(f"Loading local embedding model '{embedding_model_name}'...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Wrap for LangChain SemanticChunker
        self.langchain_embeddings = SimpleEmbeddings(self.embedding_model)
        self.text_splitter = SemanticChunker(self.langchain_embeddings)
        
        # Initialize ChromaDB persistent client
        print(f"Initializing ChromaDB at '{db_path}'...")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Get or create the collection for YouTube transcripts
        # We use cosine similarity (ChromaDB's distance space is cosine)
        self.collection = self.chroma_client.get_or_create_collection(
            name="youtube_videos",
            metadata={"hnsw:space": "cosine"}
        )
        print("RAG Engine successfully initialized.")

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """
        Extracts the 11-character YouTube video ID from various YouTube URL formats.
        """
        pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, url)
        return match.group(1) if match else None

    @staticmethod
    def fetch_video_metadata(url: str) -> Dict[str, str]:
        """
        Fetches public metadata for a YouTube video using YouTube's official oEmbed API.
        Does not require any API keys.
        """
        oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
        try:
            response = requests.get(oembed_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "title": data.get("title", "Unknown Title"),
                    "author": data.get("author_name", "Unknown Channel"),
                    "thumbnail_url": data.get("thumbnail_url", "")
                }
        except Exception:
            pass
        
        # Fallback metadata if API call fails
        video_id = RAGEngine.extract_video_id(url)
        return {
            "title": f"YouTube Video ({video_id})" if video_id else "YouTube Video",
            "author": "Unknown Channel",
            "thumbnail_url": ""
        }

    def download_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Downloads the transcript for a given video ID.
        Supports both instance-based and class-based API versions of youtube-transcript-api.
        Returns a list of caption segments: [{'text': str, 'start': float, 'duration': float}]
        """
        # 1. Try instance-based API
        try:
            api = YouTubeTranscriptApi()
            try:
                t_list = api.list(video_id)
                # Try finding English transcript
                try:
                    fetched = t_list.find_transcript(['en', 'en-US','hi']).fetch()
                except Exception:
                    # Fallback to the first available transcript
                    first_transcript = next(iter(t_list), None)
                    if first_transcript:
                        fetched = first_transcript.fetch()
                    else:
                        raise Exception("No transcripts available in this video.")
                
                # Check format of fetched object
                if hasattr(fetched, 'to_raw_data'):
                    return fetched.to_raw_data()
                elif isinstance(fetched, list):
                    return fetched
                else:
                    # Convert FetchedTranscript snippets
                    return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched.snippets]
            except Exception as e:
                # Fallback: If instance methods fail, attempt static class-based API
                if hasattr(YouTubeTranscriptApi, 'get_transcript'):
                    try:
                        return YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
                    except Exception:
                        t_list_old = YouTubeTranscriptApi.list_transcripts(video_id)
                        for t in t_list_old:
                            return t.fetch()
                raise e
        except Exception as e:
            raise Exception(
                f"Could not retrieve transcripts for video {video_id}. "
                f"Please verify that captions/subtitles are enabled for this video. Details: {e}"
            )

    def clean_text(self, text: str) -> str:
        """
        Cleans transcript text by removing common sound descriptions like [Music], [Laughter], etc.
        """
        # Remove bracketed noises e.g. [Music], [Laughter], (Applause)
        text = re.sub(r'\[[a-zA-Z\s]+\]', '', text)
        text = re.sub(r'\([a-zA-Z\s]+\)', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_transcript(self, transcript: List[Dict[str, Any]], chunk_size_words: int = 500, chunk_overlap_words: int = 100) -> List[Dict[str, Any]]:
        """
        Uses LangChain's SemanticChunker to group transcript sentences/lines into semantically
        coherent paragraphs.
        Estimates the start timestamp and duration of each chunk by matching the chunk's text
        back to the original transcript items.
        """
        if not transcript:
            return []
            
        # Clean and join the transcript texts into a single block
        clean_transcript = []
        for item in transcript:
            cleaned_text = self.clean_text(item['text'])
            if cleaned_text:
                clean_transcript.append({
                    'text': cleaned_text,
                    'start': item['start'],
                    'duration': item['duration']
                })
                
        if not clean_transcript:
            return []
            
        full_text = " ".join([item['text'] for item in clean_transcript])
        
        # Split using the SemanticChunker
        try:
            chunks = self.text_splitter.split_text(full_text)
        except Exception as e:
            # Fallback if splitting fails for any reason
            chunks = [full_text]
            
        # Align chunk texts back to original timestamps
        aligned_chunks = []
        transcript_idx = 0
        
        for chunk_text in chunks:
            chunk_words = chunk_text.split()
            if not chunk_words:
                continue
                
            items_in_chunk = []
            accumulated_words_count = 0
            
            while transcript_idx < len(clean_transcript):
                item = clean_transcript[transcript_idx]
                items_in_chunk.append(item)
                accumulated_words_count += len(item['text'].split())
                transcript_idx += 1
                
                # Stop if we've consumed enough words to match this chunk
                if accumulated_words_count >= len(chunk_words):
                    break
                    
            if items_in_chunk:
                start_time = items_in_chunk[0]['start']
                end_time = items_in_chunk[-1]['start'] + items_in_chunk[-1]['duration']
                aligned_chunks.append({
                    'text': chunk_text,
                    'start_time': start_time,
                    'duration': max(0.0, end_time - start_time)
                })
                
        return aligned_chunks

    def index_video(self, url: str) -> Dict[str, Any]:
        """
        Core pipeline to download transcript, metadata, chunk text, generate embeddings,
        and save into ChromaDB.
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {url}")
            
        # 1. Fetch metadata
        metadata = self.fetch_video_metadata(url)
        title = metadata["title"]
        author = metadata["author"]
        
        # 2. Download transcript
        transcript = self.download_transcript(video_id)
        
        # 3. Create chunks
        chunks = self.chunk_transcript(transcript)
        if not chunks:
            raise Exception(f"No text extracted from transcript for video: {title}")
            
        # 4. Generate local embeddings for each chunk
        texts = [c['text'] for c in chunks]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # 5. Insert into ChromaDB
        ids = [f"{video_id}_chunk_{idx}" for idx in range(len(chunks))]
        metadatas = []
        for idx, chunk in enumerate(chunks):
            metadatas.append({
                "video_id": video_id,
                "url": url,
                "title": title,
                "author": author,
                "start_time": chunk["start_time"],
                "duration": chunk["duration"]
            })
            
        # Remove existing documents for this video to avoid duplicates/overwrite
        self.delete_video(video_id)
        
        # Insert new chunks
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        
        return {
            "video_id": video_id,
            "title": title,
            "author": author,
            "chunks_count": len(chunks)
        }

    def list_videos(self) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB to list all indexed videos with chunk counts and metadata.
        """
        # Fetch all metadata from database
        results = self.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        
        unique_videos = {}
        for m in metadatas:
            v_id = m["video_id"]
            if v_id not in unique_videos:
                unique_videos[v_id] = {
                    "video_id": v_id,
                    "url": m["url"],
                    "title": m["title"],
                    "author": m["author"],
                    "chunks": 0
                }
            unique_videos[v_id]["chunks"] += 1
            
        return list(unique_videos.values())

    def delete_video(self, video_id: str) -> bool:
        """
        Deletes a video and its transcript chunks from the ChromaDB collection.
        """
        try:
            self.collection.delete(where={"video_id": video_id})
            return True
        except Exception:
            return False

    def search_relevant_chunks(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Performs similarity search in ChromaDB using query embeddings.
        Returns the top_k most relevant chunks.
        """
        # Check if database is empty
        if self.collection.count() == 0:
            return []
            
        query_vector = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            
            for doc, meta, dist in zip(docs, metas, dists):
                # Cosine distance to similarity: similarity = 1 - distance
                similarity = 1.0 - dist
                formatted.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": similarity
                })
        
        # Sort by similarity descending
        formatted.sort(key=lambda x: x["similarity"], reverse=True)
        return formatted

    @staticmethod
    def get_ollama_status() -> Dict[str, Any]:
        """
        Queries the local Ollama instance to check connection status and retrieve available models.
        """
        url = "http://localhost:11434/api/tags"
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return {
                    "running": True,
                    "models": models,
                    "error": None
                }
        except Exception as e:
            return {
                "running": False,
                "models": [],
                "error": str(e)
            }
        
        return {
            "running": False,
            "models": [],
            "error": f"Ollama API returned HTTP {response.status_code}"
        }

    def query_ollama_stream(self, query: str, context_chunks: List[Dict[str, Any]], model: str = "ministral:3b"):
        """
        Streams answers from the local Ollama model using the retrieved context chunks.
        Yields text chunks dynamically as they are generated.
        """
        if not context_chunks:
            yield "No relevant transcript context found in the database. Please add video URLs first."
            return
            
        # Build context prompt string
        context_str = ""
        for idx, chunk in enumerate(context_chunks):
            title = chunk["metadata"]["title"]
            start_time = chunk["metadata"]["start_time"]
            
            # Format timestamp to hh:mm:ss or mm:ss
            m, s = divmod(int(start_time), 60)
            h, m = divmod(m, 60)
            timestamp_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
            
            context_str += f"--- Source Chunk {idx+1} | Video: '{title}' (at {timestamp_str}) ---\n"
            context_str += f"{chunk['text']}\n\n"
            
        system_prompt = (
            "You are a highly precise and helpful AI assistant answering questions based on YouTube video transcripts.\n"
            "You will be given a set of transcript chunks retrieved from the videos. These chunks are relevant to the user's question.\n"
            "Please answer the user's question accurately and concisely, basing your response ONLY on the provided transcript context.\n"
            "If the answer cannot be found or inferred from the provided context, state clearly: 'I cannot find the answer in the provided video transcripts.'\n"
            "Cite the source video titles when referring to details from specific chunks in your response. Keep the tone professional."
        )
        
        prompt = (
            f"Context:\n{context_str}\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "options": {
                "temperature": 0.2  # Keep answers factual and aligned with context
            }
        }
        
        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True, timeout=30)
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        chunk_data = line.decode('utf-8')
                        try:
                            # Ollama returns JSON lines with {"response": "...", "done": bool}
                            import json
                            data = json.loads(chunk_data)
                            yield data.get("response", "")
                        except Exception:
                            pass
            else:
                yield f"\n[Error: Ollama returned status code {response.status_code}]"
        except Exception as e:
            yield f"\n[Error communicating with Ollama: {e}]"
