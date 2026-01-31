import re
import json
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

    def fill_down_merged_cells(
        self,
        rows: List[List[str]],
        col_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Handle tables that visually use merged cells (e.g., left-most label spanning many rows).

        pdfplumber often returns continuation rows with an empty string in the merged column.
        For RAG + downstream structuring, it's usually better to "fill down" that value so
        each row is self-contained.
        """
        if not rows:
            return {"filled": 0, "applied": False}

        # Heuristic: only apply if there are blanks in the column AND there is at least one
        # non-empty value before a blank (typical merged-cell extraction artifact).
        saw_value = False
        blanks_after_value = 0
        for r in rows:
            cell = r[col_idx].strip() if col_idx < len(r) and r[col_idx] else ""
            if cell:
                saw_value = True
            elif saw_value:
                blanks_after_value += 1

        if blanks_after_value == 0:
            return {"filled": 0, "applied": False}

        filled = 0
        last_val = ""
        for r in rows:
            if col_idx >= len(r):
                continue
            cell = r[col_idx].strip() if r[col_idx] else ""
            if cell:
                last_val = r[col_idx]
            else:
                if last_val:
                    r[col_idx] = last_val
                    filled += 1

        return {"filled": filled, "applied": True}
    
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

                    # Handle merged-cell style layouts (like 1.1.1(a) spanning multiple rows)
                    filldown_stats = self.fill_down_merged_cells(normalized_rows, col_idx=0)
                    
                    table_obj = TableData(
                        table_id=f"table_p{page_num}_t{idx+1}",
                        page_number=page_num,
                        headers=headers,
                        rows=normalized_rows,
                        metadata={
                            "bbox": table.bbox if hasattr(table, 'bbox') else None,
                            "row_count": len(normalized_rows),
                            "column_count": len(headers),
                            "merged_cell_filldown": {
                                "column_index": 0,
                                **filldown_stats,
                            },
                            # table_title will be filled below if a caption is found
                            "table_title": None,
                        }
                    )

                    # Attempt to find a nearby caption (e.g., "Table 1: Title") just above the table
                    try:
                        bbox = table.bbox if hasattr(table, 'bbox') else None
                        if bbox:
                            # bbox format: (x0, top, x1, bottom)
                            table_top = bbox[1]
                            table_x0, table_x1 = bbox[0], bbox[2]

                            # Collect words that are above the table within a reasonable vertical window
                            words = page.extract_words()
                            # Candidate words: those whose bottom is <= table_top + small_margin and >= table_top - search_height
                            search_height = 80
                            small_margin = 8
                            candidate_words = [w for w in words
                                               if w.get('bottom', 0) <= table_top + small_margin
                                               and w.get('bottom', 0) >= table_top - search_height
                                               and (w.get('x0', 0) <= table_x1 and w.get('x1', 0) >= table_x0)]

                            # Group words by approximate line (top coordinate)
                            lines = {}
                            for w in candidate_words:
                                top_key = int(round(w.get('top', 0)))
                                lines.setdefault(top_key, []).append(w)

                            # Search each line (closest to table first) for a caption pattern
                            caption = None
                            for top in sorted(lines.keys(), reverse=True):
                                line_words = sorted(lines[top], key=lambda x: x.get('x0', 0))
                                line_text = " ".join([w.get('text', '') for w in line_words]).strip()
                                # Common caption patterns: Table 1: Title, TABLE 1 - Title, Table 1 Title
                                m = re.match(r'(?i)^\s*(table)\s*\d+\s*[:\-–—]?\s*(.+)', line_text)
                                if m:
                                    caption = m.group(2).strip()
                                    break
                                # Sometimes captions are written like 'Table: Title' or just 'Table Title'
                                m2 = re.match(r'(?i)^\s*(table)\s*[:\-–—]?\s*(.+)', line_text)
                                if m2:
                                    caption = m2.group(2).strip()
                                    break

                            if caption:
                                table_obj.metadata['table_title'] = caption

                    except Exception:
                        # Non-fatal: if caption detection fails, keep metadata.table_title as None
                        pass
                    
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
        # Aggregate detected table titles
        titles = []
        for t in self.tables:
            tt = t.metadata.get('table_title') if isinstance(t.metadata, dict) else None
            if tt and tt.strip():
                titles.append(tt.strip())

        # Keep unique while preserving order
        seen = set()
        unique_titles = [x for x in titles if not (x in seen or seen.add(x))]

        doc_meta = {
            "file_path": self.file_path,
            "total_sections": len(self.sections),
            "total_tables": len(self.tables),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "table_titles": unique_titles[0],
        }

        return {
            "document_metadata": doc_meta,
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
    #chunk_by_heading()
    main()
