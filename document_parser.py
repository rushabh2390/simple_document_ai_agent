import os
import io
import gc
import logging
from pathlib import Path
from typing import Dict, Any, List
from pypdf import PdfReader, PdfWriter

# Docling Core Imports
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption, ExcelFormatOption
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# LangChain text splitters for high-fidelity Markdown chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MultiModalParser")


class MultiModalDocumentParser:
    def __init__(self, base_output_dir: str = "./processed_data", batch_size: int = 3):
        """Initializes the document processing engine with structural markdown capabilities."""
        self.base_dir = Path(base_output_dir)
        self.image_dir = self.base_dir / "images"
        self.table_dir = self.base_dir / "tables"
        self.batch_size = batch_size  # Crucial to prevent std::bad_alloc crashes
        
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initializing Docling Pipeline Options...")
        pdf_options = PdfPipelineOptions()
        pdf_options.images_scale = 0.5             
        pdf_options.generate_picture_images = True  
        pdf_options.do_table_structure = True       
        pdf_options.table_structure_options.do_cell_matching = True
        pdf_options.do_ocr = False                 
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
                InputFormat.DOCX: WordFormatOption(),
                InputFormat.XLSX: ExcelFormatOption()
            }
        )
        logger.info(f"Parser engine successfully active. Safe Batch Size: {batch_size}")

    def parse_file(self, file_bytes: bytes, filename: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Converts documents to layout-aware Markdown using page slicing windows 
        to guarantee low memory footprints, then chunks text appropriately.
        """
        clean_name = Path(filename).stem.replace(" ", "_")
        logger.info(f"🚀 Starting Markdown-backed memory-safe workflow for: {filename}")
        
        all_compiled_chunks = []
        markdown_segments = []
        table_paths = []

        # Case A: If it's a PDF, apply the sequential micro-batch strategy
        if filename.lower().endswith('.pdf'):
            try:
                with io.BytesIO(file_bytes) as file_stream:
                    reader = PdfReader(file_stream)
                    total_pages = len(reader.pages)
                    logger.info(f"📚 PDF page structure discovered: {total_pages} total pages.")
                    
                    # Process page window intervals incrementally
                    for start_page in range(0, total_pages, self.batch_size):
                        end_page = min(start_page + self.batch_size, total_pages)
                        logger.info(f" -> Safely parsing page window range [{start_page} to {end_page}]...")
                        
                        writer = PdfWriter()
                        for page_idx in range(start_page, end_page):
                            writer.add_page(reader.pages[page_idx])
                            
                        slice_buffer = io.BytesIO()
                        writer.write(slice_buffer)
                        slice_buffer.seek(0)
                        
                        try:
                            source_stream = DocumentStream(name=f"slice_{start_page}_{end_page}.pdf", stream=slice_buffer)
                            conversion_result = self.converter.convert(source_stream)
                            doc = conversion_result.document
                            
                            # Append structural markdown string safely
                            markdown_segments.append(doc.export_to_markdown())
                            
                            # Isolate tables found inside this slice
                            for idx, element in enumerate(getattr(doc, "tables", [])):
                                try:
                                    df = element.export_to_dataframe()
                                    if not df.empty:
                                        csv_filename = f"table_{clean_name}_p{start_page}_{idx}.csv"
                                        csv_filepath = self.table_dir / csv_filename
                                        df.to_csv(csv_filepath, index=False)
                                        table_paths.append(str(csv_filepath.resolve()))
                                except Exception as t_err:
                                    logger.warning(f"⚠️ Skipping table export index {idx}: {t_err}")
                                    
                        except Exception as slice_err:
                            logger.error(f"💥 Failed processing slice window [{start_page}-{end_page}]: {slice_err}")
                        finally:
                            slice_buffer.close()
                            del writer
                            gc.collect() # Enforce memory recovery cycles
                            
            except Exception as e:
                logger.critical(f"💥 PDF reader extraction pipeline failure on {filename}: {str(e)}")
                return []
        
        # Case B: Non-PDF files (processed directly in a single block)
        else:
            try:
                with io.BytesIO(file_bytes) as file_stream:
                    source_stream = DocumentStream(name=filename, stream=file_stream)
                    conversion_result = self.converter.convert(source_stream)
                    doc = conversion_result.document
                    
                    markdown_segments.append(doc.export_to_markdown())
                    
                    for idx, element in enumerate(getattr(doc, "tables", [])):
                        try:
                            df = element.export_to_dataframe()
                            if not df.empty:
                                csv_filename = f"table_{clean_name}_{idx}.csv"
                                csv_filepath = self.table_dir / csv_filename
                                df.to_csv(csv_filepath, index=False)
                                table_paths.append(str(csv_filepath.resolve()))
                        except Exception as t_err:
                            logger.warning(f"⚠️ Skipping table export index {idx}: {t_err}")
            except Exception as e:
                logger.critical(f"💥 Failed formatting stream conversion for {filename}: {str(e)}")
                return []

        # 2. Re-combine layout fragments and compute recursive chunks
        full_markdown_content = "\n\n".join(markdown_segments)
        
        if full_markdown_content:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            splits = text_splitter.split_text(full_markdown_content)
            
            for index, split_text in enumerate(splits):
                primary_table = table_paths[0] if table_paths else None
                
                all_compiled_chunks.append({
                    "chunk_id": f"{clean_name}_chunk_{index}",
                    "text": split_text,
                    "table_path": primary_table,
                    "image_path": None
                })
                
            gc.collect()

        logger.info(f"✅ Finished parsing workflow. Generated {len(all_compiled_chunks)} markdown chunks for {filename}.")
        return all_compiled_chunks