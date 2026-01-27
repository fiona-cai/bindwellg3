import pdfplumber
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd


@dataclass
class DocumentSection:
    """Represents a structured section of the document"""
    section_id: str
    page_number: int
    section_type: str  # 'text', 'table', 'header', 'footer'
    content: Any
    metadata: Dict[str, Any]


@dataclass
class TableData:
    """Represents extracted table data"""
    table_id: str
    page_number: int
    headers: List[str]
    rows: List[List[str]]
    metadata: Dict[str, Any]


class PDFDocumentProcessor:
    """Comprehensive PDF processor for ML/RAG pipeline"""
    
    def __init__(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.sections: List[DocumentSection] = []
        self.tables: List[TableData] = []
        
    def is_header_or_footer(self, text: str, y_position: float, page_height: float) -> bool:
        """Detect if text is likely a header or footer"""
        text = text.strip().lower()
        # Check position (top 10% or bottom 10% of page)
        is_top = y_position > page_height * 0.9
        is_bottom = y_position < page_height * 0.1
        
        # Common header/footer patterns
        header_patterns = [
            r'^\d+$',  # Page numbers
            r'^page \d+',
            r'^chapter \d+',
            r'^section \d+',
            r'^table of contents',
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, text):
                return True
                
        return is_top or is_bottom
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might be artifacts
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        return text.strip()
    
    def extract_tables_from_page(self, page, page_num: int) -> List[TableData]:
        """Extract all tables from a page with robust error handling"""
        extracted_tables = []
        
        try:
            # Method 1: Use pdfplumber's table detection
            tables = page.find_tables()
            
            for idx, table in enumerate(tables):
                try:
                    # Extract table with settings for better accuracy
                    table_data = table.extract()
                    
                    if not table_data or len(table_data) < 2:
                        continue
                    
                    # Clean and process table data
                    cleaned_data = []
                    for row in table_data:
                        if row:
                            cleaned_row = [self.clean_text(str(cell)) if cell else "" for cell in row]
                            # Skip completely empty rows
                            if any(cell.strip() for cell in cleaned_row):
                                cleaned_data.append(cleaned_row)
                    
                    if len(cleaned_data) < 2:
                        continue
                    
                    # Identify headers (usually first row or rows with consistent formatting)
                    headers = cleaned_data[0] if cleaned_data else []
                    rows = cleaned_data[1:] if len(cleaned_data) > 1 else []
                    
                    # Handle multi-row headers
                    if len(cleaned_data) > 1:
                        # Check if second row might be part of header (common in complex tables)
                        second_row = cleaned_data[1] if len(cleaned_data) > 1 else []
                        if second_row and all(not cell.strip() or cell.strip().isupper() for cell in second_row[:3]):
                            # Merge header rows
                            headers = [f"{h1} {h2}".strip() if h1 and h2 else (h1 or h2 or "")
                                      for h1, h2 in zip(headers, second_row)]
                            rows = cleaned_data[2:] if len(cleaned_data) > 2 else []
                    
                    # Clean headers
                    headers = [self.clean_text(h) for h in headers]
                    
                    # Remove empty columns
                    if headers:
                        non_empty_cols = [i for i, h in enumerate(headers) if h.strip()]
                        if non_empty_cols:
                            headers = [headers[i] for i in non_empty_cols]
                            rows = [[row[col_idx] if col_idx < len(row) else "" 
                                    for col_idx in non_empty_cols] 
                                   for row in rows]
                    
                    # Handle broken rows (rows with fewer columns)
                    max_cols = len(headers)
                    normalized_rows = []
                    for row in rows:
                        # Pad or truncate to match header count
                        normalized_row = row[:max_cols] + [""] * (max_cols - len(row))
                        normalized_rows.append(normalized_row)
                    
                    table_obj = TableData(
                        table_id=f"table_p{page_num}_t{idx+1}",
                        page_number=page_num,
                        headers=headers,
                        rows=normalized_rows,
                        metadata={
                            "bbox": table.bbox if hasattr(table, 'bbox') else None,
                            "row_count": len(normalized_rows),
                            "column_count": len(headers)
                        }
                    )
                    
                    extracted_tables.append(table_obj)
                    
                except Exception as e:
                    print(f"Error extracting table {idx+1} from page {page_num}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error finding tables on page {page_num}: {e}")
        
        return extracted_tables
    
    def extract_text_sections_from_page(self, page, page_num: int) -> List[DocumentSection]:
        """Extract text sections from a page, handling inconsistent layouts"""
        sections = []
        page_height = page.height
        
        try:
            # Get all text objects with their positions
            chars = page.chars
            
            # Group characters into words and lines
            words = []
            current_word = {"text": "", "x0": 0, "y0": 0, "x1": 0, "y1": 0}
            
            for char in chars:
                if char.get('text', '').strip():
                    if not current_word["text"]:
                        current_word = {
                            "text": char.get('text', ''),
                            "x0": char.get('x0', 0),
                            "y0": char.get('y0', 0),
                            "x1": char.get('x1', 0),
                            "y1": char.get('y1', 0)
                        }
                    else:
                        # Check if character is part of same word (close horizontally)
                        if abs(char.get('y0', 0) - current_word["y0"]) < 5:
                            current_word["text"] += char.get('text', '')
                            current_word["x1"] = char.get('x1', 0)
                        else:
                            # New word
                            words.append(current_word)
                            current_word = {
                                "text": char.get('text', ''),
                                "x0": char.get('x0', 0),
                                "y0": char.get('y0', 0),
                                "x1": char.get('x1', 0),
                                "y1": char.get('y1', 0)
                            }
            
            if current_word["text"]:
                words.append(current_word)
            
            # Extract full text for chunking
            full_text = page.extract_text()
            
            if full_text:
                # Check if this looks like a header/footer
                first_word_y = words[0]["y0"] if words else page_height
                is_header_footer = self.is_header_or_footer(
                    full_text[:100], first_word_y, page_height
                )
                
                if not is_header_footer:
                    # Clean and chunk the text
                    cleaned_text = self.clean_text(full_text)
                    
                    if cleaned_text:
                        # Split into chunks for RAG
                        chunks = self.text_splitter.split_text(cleaned_text)
                        
                        for chunk_idx, chunk in enumerate(chunks):
                            section = DocumentSection(
                                section_id=f"text_p{page_num}_c{chunk_idx+1}",
                                page_number=page_num,
                                section_type="text",
                                content=chunk,
                                metadata={
                                    "chunk_index": chunk_idx,
                                    "chunk_size": len(chunk),
                                    "word_count": len(chunk.split())
                                }
                            )
                            sections.append(section)
                            
        except Exception as e:
            print(f"Error extracting text from page {page_num}: {e}")
            # Fallback to simple text extraction
            try:
                text = page.extract_text()
                if text:
                    cleaned_text = self.clean_text(text)
                    chunks = self.text_splitter.split_text(cleaned_text)
                    for chunk_idx, chunk in enumerate(chunks):
                        section = DocumentSection(
                            section_id=f"text_p{page_num}_c{chunk_idx+1}",
                            page_number=page_num,
                            section_type="text",
                            content=chunk,
                            metadata={"chunk_index": chunk_idx}
                        )
                        sections.append(section)
            except:
                pass
        
        return sections
    
    def process_pdf(self) -> Dict[str, Any]:
        """Process the entire PDF and extract all structured data"""
        print(f"Processing PDF: {self.file_path}")
        
        with pdfplumber.open(self.file_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"Processing page {page_num}/{total_pages}...")
                
                # Extract tables first (before text to avoid overlap)
                tables = self.extract_tables_from_page(page, page_num)
                self.tables.extend(tables)
                print(f"  Found {len(tables)} table(s)")
                
                # Extract text sections
                text_sections = self.extract_text_sections_from_page(page, page_num)
                self.sections.extend(text_sections)
                print(f"  Created {len(text_sections)} text chunk(s)")
        
        print(f"\nProcessing complete!")
        print(f"Total sections: {len(self.sections)}")
        print(f"Total tables: {len(self.tables)}")
        
        return self.get_structured_output()
    
    def get_structured_output(self) -> Dict[str, Any]:
        """Get structured output for ML/RAG pipeline"""
        return {
            "document_metadata": {
                "file_path": self.file_path,
                "total_sections": len(self.sections),
                "total_tables": len(self.tables),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            },
            "sections": [asdict(section) for section in self.sections],
            "tables": [asdict(table) for table in self.tables]
        }
    
    def save_to_json(self, output_path: str = "processed_document.json"):
        """Save processed document to JSON file"""
        output = self.get_structured_output()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved structured output to: {output_path}")
        
        # Also save tables separately for easy access
        tables_output = {
            "tables": [asdict(table) for table in self.tables]
        }
        
        tables_path = output_path.replace('.json', '_tables.json')
        with open(tables_path, 'w', encoding='utf-8') as f:
            json.dump(tables_output, f, indent=2, ensure_ascii=False)
        
        print(f"Saved tables to: {tables_path}")


def main():
    """Main execution function"""
    file_path = "./2026-pgp.pdf"
    
    # Initialize processor
    processor = PDFDocumentProcessor(
        file_path=file_path,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Process the PDF
    structured_output = processor.process_pdf()
    
    # Save to JSON
    processor.save_to_json("processed_document.json")
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total text sections: {len(processor.sections)}")
    print(f"Total tables extracted: {len(processor.tables)}")
    
    if processor.tables:
        print("\nTable Summary:")
        for table in processor.tables:
            print(f"  - {table.table_id}: {len(table.rows)} rows, {len(table.headers)} columns")
            print(f"    Headers: {', '.join(table.headers[:5])}{'...' if len(table.headers) > 5 else ''}")
    
    if processor.sections:
        print(f"\nSample text section (first 200 chars):")
        print(processor.sections[0].content[:200] + "...")


if __name__ == "__main__":
    main()
