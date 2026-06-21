from docling.document_converter import DocumentConverter

source_url = "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf" # Example PDF URL

converter = DocumentConverter()
result = converter.convert(source_url)

# Print as Markdown (or JSON/Text)
print(result.document.export_to_markdown())
