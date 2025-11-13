import os
import PyPDF2
import docx
import re
from pathlib import Path

class FileProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.doc', '.txt']
    
    def extract_text(self, file_path):
        """Extract text from various file formats"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif extension in ['.docx', '.doc']:
            return self._extract_from_docx(file_path)
        elif extension == '.txt':
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _extract_from_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def _extract_from_docx(self, file_path):
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")
    
    def _extract_from_txt(self, file_path):
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            raise Exception(f"Error extracting text from TXT: {str(e)}")
    
    def validate_file(self, file_path, max_size_mb=16):
        """Validate file format and size"""
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            return False, "File does not exist"
        
        # Check file extension
        if file_path.suffix.lower() not in self.supported_formats:
            return False, f"Unsupported file format. Supported: {', '.join(self.supported_formats)}"
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
        
        return True, "File is valid"
    
    def clean_filename(self, filename):
        """Clean filename for safe storage"""
        # Remove special characters and spaces
        clean_name = re.sub(r'[^\w\-_\.]', '_', filename)
        # Remove multiple underscores
        clean_name = re.sub(r'_+', '_', clean_name)
        return clean_name
    
    def get_file_info(self, file_path):
        """Get file information"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None
        
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'size': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'extension': file_path.suffix.lower(),
            'modified': stat.st_mtime
        }
