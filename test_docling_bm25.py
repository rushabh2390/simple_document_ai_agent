import os
import re
import io
import gc
from pypdf import PdfReader, PdfWriter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
from rank_bm25 import BM25Okapi

# Create a local directory to hold the output images
IMAGE_OUTPUT_DIR = "./extracted_images"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# 1. Pipeline Settings
pdf_options = PdfPipelineOptions()
pdf_options.images_scale = 0.5            
pdf_options.generate_picture_images = True # Populate image crops safely
pdf_options.do_table_structure = False      
pdf_options.do_ocr = False                  

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
    }
)

def tokenize_text(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()

# Target file configuration
pdf_path = r"E:\Deep-Learning-with-PyTorch.pdf"

if not os.path.exists(pdf_path):
    print(f"❌ Error: Cannot find file at '{pdf_path}'")
    exit(1)

chunks_pool = []  
chunker = HybridChunker(max_tokens=5000)

print(f"⏳ Reading PDF page structure using pypdf...")
reader = PdfReader(pdf_path)
total_pages = len(reader.pages)
print(f"📚 Total pages discovered: {total_pages}")

# Batch size of 3 avoids memory overflow on heavy graphic page segments
BATCH_SIZE = 3

print(f"⏳ Step 1 & 2: Batch processing document pages in micro-segments of {BATCH_SIZE}...")

for start_page in range(0, total_pages, BATCH_SIZE):
    end_page = min(start_page + BATCH_SIZE, total_pages)
    print(f" -> Processing pages [{start_page} to {end_page}]...")
    
    writer = PdfWriter()
    for page_idx in range(start_page, end_page):
        writer.add_page(reader.pages[page_idx])
        
    pdf_buffer = io.BytesIO()
    writer.write(pdf_buffer)
    pdf_buffer.seek(0)
    
    try:
        source_stream = DocumentStream(
            name=f"batch_{start_page}_{end_page}.pdf", 
            stream=pdf_buffer
        )
        
        conversion_result = converter.convert(source_stream)
        doc = conversion_result.document
        
        # ─── FIXED: CORRECT V2 UNIVERSAL ITEM ITERATION ───
        saved_images_map = {}
        
        # Iterate through structural items and look for populated item images
        for item, _level in doc.iterate_items():
            if hasattr(item, "image") and item.image and getattr(item.image, "pil_image", None):
                img_filename = f"img_page_{start_page}_{item.self_ref.replace('/', '_')}.png"
                img_filepath = os.path.join(IMAGE_OUTPUT_DIR, img_filename)
                
                # Save the cropped layout image to disk
                item.image.pil_image.save(img_filepath, format="PNG")
                saved_images_map[item.self_ref] = img_filepath

        # Native semantic chunk split
        doc_chunks = chunker.chunk(doc)
        for chunk in doc_chunks:
            chunk_text = chunker.contextualize(chunk)
            
            matched_image_path = None
            if hasattr(chunk, "meta") and chunk.meta.doc_items:
                for meta_item in chunk.meta.doc_items:
                    if meta_item.self_ref in saved_images_map:
                        matched_image_path = saved_images_map[meta_item.self_ref]
                        break
            
            chunks_pool.append({
                "text": chunk_text,
                "image_path": matched_image_path
            })
            
    except Exception as e:
        print(f" ⚠️ Warning: Failed processing page slice range [{start_page}-{end_page}]: {e}")
        continue
    finally:
        # Explicit clean-up logic to release back-end C++ buffers on every loop execution
        pdf_buffer.close()
        del writer
        gc.collect()

print(f"✅ Successfully compiled {len(chunks_pool)} total chunks across all page batches.")

# 3. Build Search Index
if len(chunks_pool) == 0:
    print("❌ Error: No chunks were compiled. Index compilation aborted.")
    exit(1)

print("⏳ Step 3: Compiling local BM25 matrix indexes...")
tokenized_corpus = [tokenize_text(item["text"]) for item in chunks_pool]
bm25_index = BM25Okapi(tokenized_corpus)
print("🚀 Search engine index built successfully!")

# 4. Verification Test
print("\n" + "="*50)
print("🔍 SYSTEM CORE VERIFICATION QUERY")
print("="*50)

test_query = "alexnet convolutional networks training specifications"
print(f"User Query: '{test_query}'\n")

tokenized_query = tokenize_text(test_query)
matched_nodes = bm25_index.get_top_n(tokenized_query, chunks_pool, n=2)

if matched_nodes:
    for idx, node in enumerate(matched_nodes):
        print(f"--- MATCHING SEGMENT {idx+1} ---")
        print(f"TEXT CONTENT:\n{node['text']}")
        if node['image_path']:
            print(f"🖼️ ASSOCIATED DIAGRAM LOCATION: {node['image_path']}")
        else:
            print("🖼️ ASSOCIATED DIAGRAM LOCATION: None (Pure Text Node)")
        print("-" * 40)
else:
    print("⚠️ No direct text intersections discovered for the given tokens.")