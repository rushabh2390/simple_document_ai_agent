import os
import io
import gc
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd  # High-performance tabular data extraction
from pypdf import PdfReader, PdfWriter

# Docling Core Framework Imports
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
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
        """Initializes the multi-format document routing and processing engine."""
        self.base_dir = Path(base_output_dir)
        self.image_dir = self.base_dir / "images"
        self.table_dir = self.base_dir / "tables"
        self.batch_size = batch_size  
        
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initializing Docling Engine for layout documents...")
        pdf_options = PdfPipelineOptions()
        pdf_options.images_scale = 1.0              
        pdf_options.generate_picture_images = True  
        pdf_options.do_table_structure = True       
        pdf_options.table_structure_options.do_cell_matching = True
        pdf_options.do_ocr = False                 
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
                InputFormat.DOCX: WordFormatOption()
            }
        )
        logger.info(f"Parser engine successfully active. Safe Batch Size: {batch_size}")

    def parse_file(self, file_bytes: bytes, filename: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Routes files dynamically based on extension. Tabular data gets converted to 
        structural markdown tables, while documents use layout analysis.
        """
        clean_name = Path(filename).stem.replace(" ", "_")
        ext = Path(filename).suffix.lower()
        logger.info(f"🚀 Routing incoming file [{ext}]: {filename}")
        
        all_compiled_chunks = []
        markdown_segments = []
        table_paths = []
        image_paths = []

        # =====================================================================
        # ROUTE 1: TABULAR DATA PIPELINES (CSV & EXCEL)
        # =====================================================================
        if ext in ['.csv', '.xlsx', '.xls']:
            try:
                if ext == '.csv':
                    csv_text = file_bytes.decode('utf-8', errors='replace')
                    df = pd.read_csv(io.StringIO(csv_text))
                else:
                    df = pd.read_excel(io.BytesIO(file_bytes))
                
                if df.empty:
                    logger.warning(f"⚠️ Tabular file {filename} is empty.")
                    return []

                # Group rows dynamically (e.g., chunks of 15-20 records each)
                row_chunk_size = 15 
                headers_str = ", ".join(df.columns.tolist())
                
                for i in range(0, len(df), row_chunk_size):
                    sub_df = df.iloc[i : i + row_chunk_size]
                    
                    # Store sub-slice as markdown representation for retrieval
                    markdown_table = sub_df.to_markdown(index=False)
                    
                    # 🔥 INJECT CONTEXT CRITERIA DIRECTLY TO MAKE BM25 SEARCH WORK PERFECTLY
                    semantic_text = (
                        f"Dataset File: {filename}\n"
                        f"Columns Present: {headers_str}\n"
                        f"Data Rows:\n{markdown_table}"
                    )
                    
                    # Persist this specific structural slice to a localized CSV matrix
                    chunk_uid = f"{clean_name}_chunk_row_{i}"
                    csv_filename = f"table_{clean_name}_slice_{i}.csv"
                    csv_filepath = self.table_dir / csv_filename
                    sub_df.to_csv(csv_filepath, index=False)
                    
                    all_compiled_chunks.append({
                        "chunk_id": chunk_uid,
                        "text": semantic_text,
                        "table_path": str(csv_filepath.resolve()),
                        "image_path": None
                    })
                
                logger.info(f"✅ Finished tabular parsing workflow. Generated {len(all_compiled_chunks)} row chunks.")
                return all_compiled_chunks
                
            except Exception as e:
                logger.error(f"💥 Native tabular pandas processor failed on {filename}: {str(e)}")
                return []

        # =====================================================================
        # ROUTE 2: BATCH-SLICED PDF WORKFLOWS
        # =====================================================================
        elif ext == '.pdf':
            try:
                with io.BytesIO(file_bytes) as file_stream:
                    reader = PdfReader(file_stream)
                    total_pages = len(reader.pages)
                    logger.info(f"📚 PDF page structure discovered: {total_pages} total pages.")
                    
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
                            
                            markdown_segments.append(doc.export_to_markdown())
                            
                            for idx, element in enumerate(getattr(doc, "tables", [])):
                                try:
                                    tdf = element.export_to_dataframe()
                                    if not tdf.empty:
                                        csv_filename = f"table_{clean_name}_p{start_page}_{idx}.csv"
                                        csv_filepath = self.table_dir / csv_filename
                                        tdf.to_csv(csv_filepath, index=False)
                                        table_paths.append(str(csv_filepath.resolve()))
                                except Exception as t_err:
                                    logger.warning(f"⚠️ Skipping table export index {idx}: {t_err}")

                            for idx, element in enumerate(getattr(doc, "pictures", [])):
                                try:
                                    if hasattr(element, "image") and element.image:
                                        img_filename = f"image_{clean_name}_p{start_page}_{idx}.png"
                                        img_filepath = self.image_dir / img_filename
                                        element.image.save(img_filepath, "PNG")
                                        image_paths.append(str(img_filepath.resolve()))
                                except Exception as img_err:
                                    logger.warning(f"⚠️ Skipping picture export index {idx}: {img_err}")
                                    
                        except Exception as slice_err:
                            logger.error(f"💥 Failed processing slice window [{start_page}-{end_page}]: {slice_err}")
                        finally:
                            slice_buffer.close()
                            del writer
                            gc.collect() 
                            
            except Exception as e:
                logger.critical(f"💥 PDF reader extraction pipeline failure on {filename}: {str(e)}")
                return []

        # =====================================================================
        # ROUTE 3: LAYOUT TEXT / WORD / MARKDOWN PROCESSING
        # =====================================================================
        elif ext in ['.docx', '.txt', '.md']:
            try:
                # Text/Markdown files can be read directly or wrapped in native layout streams
                if ext in ['.txt', '.md']:
                    text_content = file_bytes.decode('utf-8', errors='replace')
                    markdown_segments.append(text_content)
                else:
                    # Word Document processing via Docling
                    with io.BytesIO(file_bytes) as file_stream:
                        source_stream = DocumentStream(name=filename, stream=file_stream)
                        conversion_result = self.converter.convert(source_stream)
                        doc = conversion_result.document
                        markdown_segments.append(doc.export_to_markdown())
            except Exception as e:
                logger.critical(f"💥 Layout text processor failed on {filename}: {str(e)}")
                return []
        
        else:
            logger.warning(f"🚫 Unsupported file format encountered: {ext}")
            return []

        # =====================================================================
        # 3. CHUNKING & MULTIMEDIA MAPPING EXTENSION
        # =====================================================================
        if ext not in ['.csv', '.xlsx', '.xls'] and markdown_segments:
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
                    # 🔥 FIX: Only link assets if they are referenced or correspond to this text slice
                    assigned_table = None
                    for t_path in table_paths:
                        t_name = Path(t_path).stem
                        if t_name in split_text or (len(table_paths) == 1 and ext in ['.csv', '.xlsx']):
                            assigned_table = t_path
                            break
                    
                    assigned_image = None
                    for img_path in image_paths:
                        img_name = Path(img_path).stem
                        if img_name in split_text:
                            assigned_image = img_path
                            break
                    
                    all_compiled_chunks.append({
                        "chunk_id": f"{clean_name}_chunk_{index}",
                        "text": split_text,
                        "table_path": assigned_table,
                        "image_path": assigned_image  
                    })

        gc.collect()
        logger.info(f"✅ Finished parsing workflow. Generated {len(all_compiled_chunks)} chunks for {filename}.")
        return all_compiled_chunks