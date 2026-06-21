import os
import sys
import requests
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.align import Align

from rag_engine_vid import RAGEngine

# Initialize console
console = Console()

class YouTubeRAGCLI:
    def __init__(self):
        # Default target model
        self.default_model = "ministral:3b"
        self.selected_model = self.default_model
        
        console.print("[bold cyan]Starting YouTube RAG CLI System...[/bold cyan]")
        
        # Initialize the RAG engine
        try:
            self.engine = RAGEngine(db_path="./chroma_db", embedding_model_name="all-MiniLM-L6-v2")
        except Exception as e:
            console.print(f"[bold red]Failed to initialize RAG Engine: {e}[/bold red]")
            sys.exit(1)
            
        # Check Ollama status
        self.check_ollama()

    def check_ollama(self) -> None:
        """
        Check local Ollama server status and update active model.
        """
        status = self.engine.get_ollama_status()
        self.ollama_running = status["running"]
        self.available_models = status["models"]
        
        if self.ollama_running:
            # Check if our default model is available
            # Ollama tags can include ':latest', so check for exact match or prefix
            model_names = [m.split(":")[0] for m in self.available_models]
            
            if "ministral" in model_names or "ministral:3b" in self.available_models:
                # Find the exact tag name
                exact_tag = next((m for m in self.available_models if m.startswith("ministral")), "ministral:3b")
                self.selected_model = exact_tag
            elif self.available_models:
                # Fallback to the first available model if Ministral isn't pulled yet
                first_model = self.available_models[0]
                self.selected_model = first_model
        else:
            self.selected_model = "None (Ollama offline)"

    def print_banner(self) -> None:
        """
        Renders a beautiful welcome header panel.
        """
        console.clear()
        
        # Determine status styling
        if self.ollama_running:
            ollama_status_text = "[bold green]ONLINE[/bold green]"
            model_names = [m.split(":")[0] for m in self.available_models]
            if "ministral" in model_names or "ministral:3b" in self.available_models:
                model_status_text = f"[bold green]{self.selected_model}[/bold green]"
            else:
                model_status_text = f"[bold yellow]{self.selected_model}[/bold yellow] [dim](Ministral 3B not found - using fallback)[/dim]"
        else:
            ollama_status_text = "[bold red]OFFLINE (Answer generation disabled)[/bold red]"
            model_status_text = "[bold red]N/A (Ollama offline)[/bold red]"
            
        video_count = 0
        try:
            videos = self.engine.list_videos()
            video_count = len(videos)
        except Exception:
            pass

        banner_content = (
            f"  [bold cyan]DATABASE DIRECTORY:[/bold cyan]  ./chroma_db\n"
            f"  [bold cyan]INDEXED VIDEOS :[/bold cyan]  {video_count} video(s)\n"
            f"  [bold cyan]OLLAMA SERVER  :[/bold cyan]  {ollama_status_text}\n"
            f"  [bold cyan]ACTIVE MODEL   :[/bold cyan]  {model_status_text}"
        )
        
        # Display instructions for pulling Ministral 3B if Ollama is running but model is missing
        if self.ollama_running:
            model_names = [m.split(":")[0] for m in self.available_models]
            if "ministral" not in model_names and "ministral:3b" not in self.available_models:
                banner_content += (
                    "\n\n[bold yellow]⚠️  Ministral 3B model not found in Ollama.[/bold yellow]\n"
                    "   Run: [bold white]ollama pull ministral:3b[/bold white] in your terminal to download it.\n"
                    "   Or select one of your other available models in Option [5]."
                )
        else:
            banner_content += (
                "\n\n[bold yellow]⚠️  Ollama is not running locally.[/bold yellow]\n"
                "   Make sure Ollama is installed and run: [bold white]ollama serve[/bold white]\n"
                "   You can still index videos and perform similarity chunk search offline."
            )

        panel = Panel(
            Align.left(banner_content),
            title="🎥 [bold magenta]YOUTUBE SEARCHABLE KNOWLEDGE BASE (RAG)[/bold magenta] 🎥",
            subtitle="[dim]Powered by ChromaDB, SentenceTransformers & Ollama[/dim]",
            border_style="magenta",
            expand=False,
            padding=(1, 4)
        )
        console.print(panel)
        console.print()

    def run(self) -> None:
        """
        Main REPL Loop.
        """
        while True:
            self.check_ollama()
            self.print_banner()
            
            console.print("[bold white]Choose an action:[/bold white]")
            console.print("  [bold green]1.[/bold green] 📥 Add / Index YouTube Video")
            console.print("  [bold green]2.[/bold green] 🔍 Search & Ask Questions (RAG Q&A)")
            console.print("  [bold green]3.[/bold green] 📋 List Indexed Videos")
            console.print("  [bold green]4.[/bold green] ❌ Delete Video from Database")
            console.print("  [bold green]5.[/bold green] ⚙️  Ollama Settings & Model Selection")
            console.print("  [bold green]6.[/bold green] 🚪 Exit")
            console.print()
            
            choice = Prompt.ask("[bold yellow]Enter choice (1-6)[/bold yellow]", choices=["1", "2", "3", "4", "5", "6"], default="2")
            
            if choice == "1":
                self.index_video_menu()
            elif choice == "2":
                self.search_qa_menu()
            elif choice == "3":
                self.list_videos_menu()
            elif choice == "4":
                self.delete_video_menu()
            elif choice == "5":
                self.settings_menu()
            elif choice == "6":
                console.print("\n[bold cyan]Goodbye![/bold cyan]")
                sys.exit(0)
                
            input("\n[dim]Press Enter to return to main menu...[/dim]")

    def index_video_menu(self) -> None:
        """
        CLI option to add and index a YouTube video URL.
        """
        console.print("\n[bold magenta]--- INDEX A NEW YOUTUBE VIDEO ---[/bold magenta]\n")
        url = Prompt.ask("[bold white]Paste YouTube URL[/bold white]")
        url = url.strip()
        if not url:
            console.print("[bold red]URL cannot be empty![/bold red]")
            return
            
        video_id = self.engine.extract_video_id(url)
        if not video_id:
            console.print("[bold red]Could not extract a valid YouTube Video ID from the URL. Please check the URL.[/bold red]")
            return
            
        console.print(f"[cyan]Target Video ID: {video_id}[/cyan]")
        
        # Run indexing with a beautiful Rich progress spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(description="Fetching video metadata...", total=100)
            
            # Fetch metadata
            metadata = self.engine.fetch_video_metadata(url)
            progress.update(task, description=f"Video Found: [bold white]{metadata['title']}[/bold white] by [cyan]{metadata['author']}[/cyan]")
            
            # Download transcript
            progress.update(task, description="Downloading video transcripts/captions...")
            try:
                transcript = self.engine.download_transcript(video_id)
            except Exception as e:
                progress.stop()
                console.print(f"\n[bold red]❌ Transcript Error: {e}[/bold red]")
                return
                
            # Chunking
            progress.update(task, description="Chunking transcript text (500-1000 tokens/chunk)...")
            chunks = self.engine.chunk_transcript(transcript)
            progress.update(task, description=f"Split transcript into [bold white]{len(chunks)}[/bold white] chunks.")
            
            # Generate Embeddings & Save to ChromaDB
            progress.update(task, description="Computing local embeddings & writing to ChromaDB...")
            try:
                result = self.engine.index_video(url)
                progress.update(task, completed=100, description="Indexing complete!")
            except Exception as e:
                progress.stop()
                console.print(f"\n[bold red]❌ Database/Embedding Error: {e}[/bold red]")
                return
                
        # Show success panel
        success_info = (
            f"[bold green]Successfully indexed video![/bold green]\n\n"
            f"[bold]Title:[/bold]  {result['title']}\n"
            f"[bold]Channel:[/bold] {result['author']}\n"
            f"[bold]Chunks:[/bold]  {result['chunks_count']} chunks saved to database."
        )
        console.print(Panel(success_info, border_style="green", title="📥 SUCCESS"))

    def search_qa_menu(self) -> None:
        """
        CLI option to query the knowledge base and run RAG.
        """
        console.print("\n[bold magenta]--- SEARCH & ASK QUESTIONS (RAG Q&A) ---[/bold magenta]\n")
        
        # Check if database has anything
        try:
            videos = self.engine.list_videos()
            if not videos:
                console.print("[bold yellow]The database is currently empty. Please index some videos first (Option 1).[/bold yellow]")
                return
        except Exception as e:
            console.print(f"[bold red]Error checking database status: {e}[/bold red]")
            return
            
        query = Prompt.ask("[bold white]Ask a question about the videos[/bold white]")
        query = query.strip()
        if not query:
            return
            
        # 1. Search ChromaDB for relevant chunks
        console.print("\n[cyan]Searching ChromaDB for relevant transcript segments...[/cyan]")
        try:
            chunks = self.engine.search_relevant_chunks(query, top_k=4)
        except Exception as e:
            console.print(f"[bold red]Search error: {e}[/bold red]")
            return
            
        if not chunks:
            console.print("[bold yellow]No matching transcript segments found. The database might be empty or query is out of bounds.[/bold yellow]")
            return
            
        # 2. Display the relevant chunks in a clean layout
        console.print(f"\n[bold cyan]Found {len(chunks)} matching segments in database:[/bold cyan]")
        
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("No.", width=4, justify="center")
        table.add_column("Video Source", ratio=2)
        table.add_column("Time", width=8, justify="center")
        table.add_column("Snippet Preview", ratio=5)
        table.add_column("Match", width=8, justify="center")
        
        for i, chunk in enumerate(chunks):
            title = chunk["metadata"]["title"]
            start_time = chunk["metadata"]["start_time"]
            similarity = chunk["similarity"]
            snippet = chunk["text"][:120].replace("\n", " ") + "..."
            
            # Format time
            m, s = divmod(int(start_time), 60)
            h, m = divmod(m, 60)
            timestamp_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
            
            table.add_row(
                str(i + 1),
                title,
                timestamp_str,
                snippet,
                f"{similarity:.1%}"
            )
            
        console.print(table)
        console.print()
        
        # 3. Call local Ollama LLM to answer
        if not self.ollama_running:
            console.print("[bold yellow]⚠️ Ollama is not running. Answer generation is skipped. You can view the raw matching segments above.[/bold yellow]")
            
            # Offer to display full text of matching chunks
            if Confirm.ask("Do you want to see the full text of the matched chunks?"):
                for idx, chunk in enumerate(chunks):
                    title = chunk["metadata"]["title"]
                    start_time = chunk["metadata"]["start_time"]
                    
                    m, s = divmod(int(start_time), 60)
                    h, m = divmod(m, 60)
                    timestamp_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
                    
                    console.print(Panel(
                        chunk["text"], 
                        title=f"[bold]Chunk {idx+1} | {title} at {timestamp_str}[/bold]", 
                        border_style="cyan"
                    ))
            return
            
        console.print(f"[cyan]Prompting local model [bold white]{self.selected_model}[/bold white] to answer... (streaming response)[/cyan]\n")
        
        console.print("[bold green]Answer:[/bold green]")
        
        answer_text = ""
        # Create live console context to render streaming output cleanly
        with Live(Text(""), refresh_per_second=10, console=console) as live:
            for stream_chunk in self.engine.query_ollama_stream(query, chunks, model=self.selected_model):
                answer_text += stream_chunk
                # Render markdown progressively
                live.update(Markdown(answer_text))
                
        console.print("\n[dim]--- End of Answer ---[/dim]\n")
        
        # Option to show source chunk details
        if Confirm.ask("Do you want to see the full text of the retrieved chunks?"):
            for idx, chunk in enumerate(chunks):
                title = chunk["metadata"]["title"]
                start_time = chunk["metadata"]["start_time"]
                
                # Make deep-linked URL
                url = chunk["metadata"]["url"]
                clean_url = url.split('&t=')[0].split('?t=')[0]
                timed_url = f"{clean_url}&t={int(start_time)}s"
                
                m, s = divmod(int(start_time), 60)
                h, m = divmod(m, 60)
                timestamp_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
                
                info = (
                    f"{chunk['text']}\n\n"
                    f"[bold dim]Link to video time:[/bold dim] [blue underline]{timed_url}[/blue underline]"
                )
                
                console.print(Panel(
                    info, 
                    title=f"[bold]Chunk {idx+1} | {title} at {timestamp_str}[/bold]", 
                    border_style="cyan"
                ))

    def list_videos_menu(self) -> None:
        """
        CLI option to list all videos currently in the database.
        """
        console.print("\n[bold magenta]--- INDEXED VIDEOS IN DATABASE ---[/bold magenta]\n")
        try:
            videos = self.engine.list_videos()
            if not videos:
                console.print("[bold yellow]No videos indexed yet. Go to Option 1 to add some videos![/bold yellow]")
                return
                
            table = Table(show_header=True, header_style="bold magenta", expand=True)
            table.add_column("Video ID", width=13, style="dim")
            table.add_column("Video Title", ratio=3)
            table.add_column("Channel / Author", ratio=1)
            table.add_column("Database Chunks", width=18, justify="right")
            
            for v in videos:
                table.add_row(
                    v["video_id"],
                    v["title"],
                    v["author"],
                    f"{v['chunks']} chunks"
                )
                
            console.print(table)
            
        except Exception as e:
            console.print(f"[bold red]Failed to retrieve video list: {e}[/bold red]")

    def delete_video_menu(self) -> None:
        """
        CLI option to delete a video's transcript from the database.
        """
        console.print("\n[bold magenta]--- DELETE VIDEO FROM DATABASE ---[/bold magenta]\n")
        try:
            videos = self.engine.list_videos()
            if not videos:
                console.print("[bold yellow]No videos indexed to delete.[/bold yellow]")
                return
                
            # List videos for user choice
            for idx, v in enumerate(videos):
                console.print(f"  [{idx + 1}] [bold white]{v['title']}[/bold white] [dim]({v['video_id']})[/dim]")
            console.print()
            
            choice_str = Prompt.ask(
                "[bold yellow]Select video index to delete (or press Enter to cancel)[/bold yellow]",
                choices=[str(i+1) for i in range(len(videos))] + [""]
            )
            
            if not choice_str:
                console.print("[yellow]Deletion cancelled.[/yellow]")
                return
                
            selected_video = videos[int(choice_str) - 1]
            title = selected_video["title"]
            video_id = selected_video["video_id"]
            
            if Confirm.ask(f"[bold red]Are you absolutely sure you want to delete the transcript data for '{title}'?[/bold red]"):
                success = self.engine.delete_video(video_id)
                if success:
                    console.print(f"[bold green]Successfully deleted video '{title}' from database.[/bold green]")
                else:
                    console.print("[bold red]Failed to delete video from database.[/bold red]")
                    
        except Exception as e:
            console.print(f"[bold red]Error deleting video: {e}[/bold red]")

    def settings_menu(self) -> None:
        """
        CLI option to view available models and change the active LLM.
        """
        console.print("\n[bold magenta]--- OLLAMA SETTINGS ---[/bold magenta]\n")
        
        self.check_ollama()
        
        if not self.ollama_running:
            console.print("[bold red]Ollama server is currently OFFLINE.[/bold red]")
            console.print("Make sure the Ollama application is running and listening on http://localhost:11434.")
            return
            
        console.print(f"[bold]Ollama server status:[/bold] [bold green]Running (Online)[/bold green]")
        console.print(f"[bold]Active Model:[/bold] {self.selected_model}\n")
        
        if not self.available_models:
            console.print("[bold yellow]No local models found in your Ollama installation.[/bold yellow]")
            console.print("Please pull a model first, for example: [bold white]ollama pull ministral:3b[/bold white]")
            return
            
        console.print("[bold white]Available Local Models:[/bold white]")
        for idx, model in enumerate(self.available_models):
            star = "★" if model == self.selected_model else " "
            console.print(f"  [{idx + 1}] {star} {model}")
            
        console.print()
        choice = Prompt.ask(
            "[bold yellow]Select model number to switch active model (or press Enter to keep current)[/bold yellow]",
            choices=[str(i+1) for i in range(len(self.available_models))] + [""]
        )
        
        if choice:
            self.selected_model = self.available_models[int(choice) - 1]
            console.print(f"[bold green]Active model switched to: {self.selected_model}[/bold green]")


if __name__ == "__main__":
    cli = YouTubeRAGCLI()
    try:
        cli.run()
    except KeyboardInterrupt:
        console.print("\n\n[bold cyan]Program interrupted. Exiting YouTube RAG CLI...[/bold cyan]")
        sys.exit(0)
