import spacy
import re
import json
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_cleaner import TextCleaner

class NERParser:
    def __init__(self, model_path="ml_models/trained_ner_model"):
        self.model_path = Path(model_path)
        self.nlp = None
        self.text_cleaner = TextCleaner()
        self.load_model()
    
    def load_model(self):
        """Load the trained NER model"""
        try:
            if self.model_path.exists():
                self.nlp = spacy.load(self.model_path)
                print(f"Loaded trained NER model from {self.model_path}")
            else:
                # Fallback to base model if trained model not found
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    print("Loaded base spaCy model (trained model not found)")
                except OSError:
                    # Create blank model as last resort
                    self.nlp = spacy.blank("en")
                    print("Created blank spaCy model (no models found)")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.nlp = spacy.blank("en")
    
    def parse_resume(self, text):
        """Parse resume text and extract structured information"""
        if not text:
            return {}
        
        # Clean text
        cleaned_text = self.text_cleaner.clean_text(text)
        
        # Process with NER model
        doc = self.nlp(cleaned_text)
        
        # Initialize result structure
        parsed_data = {
            'name': None,
            'email': None,
            'phone': None,
            'location': None,
            'skills': [],
            'experience_years': None,
            'education': [],
            'certifications': [],
            'linkedin': None,
            'github': None,
            'summary': None,
            'experience': [],
            'projects': []
        }
        
        # Extract entities using NER
        for ent in doc.ents:
            if ent.label_ == "PERSON" and not parsed_data['name']:
                parsed_data['name'] = ent.text.strip()
            elif ent.label_ == "EMAIL":
                parsed_data['email'] = ent.text.strip()
            elif ent.label_ == "PHONE":
                parsed_data['phone'] = ent.text.strip()
            elif ent.label_ == "LOCATION":
                parsed_data['location'] = ent.text.strip()
            elif ent.label_ == "SKILLS":
                skills = self.text_cleaner.normalize_skills(ent.text)
                parsed_data['skills'].extend(skills)
            elif ent.label_ == "EXPERIENCE_YEARS":
                years = re.findall(r'\d+', ent.text)
                if years:
                    parsed_data['experience_years'] = int(years[0])
            elif ent.label_ in ["DEGREE", "COLLEGE"]:
                parsed_data['education'].append(ent.text.strip())
            elif ent.label_ == "CERTIFICATION":
                parsed_data['certifications'].append(ent.text.strip())
        
        # Fallback extraction using regex patterns
        self._fallback_extraction(text, parsed_data)
        
        # Clean and deduplicate
        parsed_data['skills'] = list(set(parsed_data['skills']))
        parsed_data['education'] = list(set(parsed_data['education']))
        parsed_data['certifications'] = list(set(parsed_data['certifications']))
        
        return parsed_data
    
    def _fallback_extraction(self, text, parsed_data):
        """Fallback extraction using regex patterns"""
        
        # Extract email if not found
        if not parsed_data['email']:
            emails = self.text_cleaner.extract_emails(text)
            if emails:
                parsed_data['email'] = emails[0]
        
        # Extract phone if not found
        if not parsed_data['phone']:
            phones = self.text_cleaner.extract_phones(text)
            if phones:
                parsed_data['phone'] = phones[0]
        
        # Extract LinkedIn profile
        linkedin_profiles = self.text_cleaner.extract_linkedin_profiles(text)
        if linkedin_profiles:
            parsed_data['linkedin'] = f"https://{linkedin_profiles[0]}"
        
        # Extract GitHub profile
        github_profiles = self.text_cleaner.extract_github_profiles(text)
        if github_profiles:
            parsed_data['github'] = f"https://{github_profiles[0]}"
        
        # Extract years of experience if not found
        if not parsed_data['experience_years']:
            years = self.text_cleaner.extract_years_of_experience(text)
            if years:
                parsed_data['experience_years'] = years
        
        # Extract education information
        education_info = self.text_cleaner.extract_education_info(text)
        parsed_data['education'].extend(education_info)
        
        # Extract certifications
        certifications = self.text_cleaner.extract_certifications(text)
        parsed_data['certifications'].extend(certifications)
        
        # Extract name from first line if not found
        if not parsed_data['name']:
            lines = text.split('\n')
            for line in lines[:3]:  # Check first 3 lines
                line = line.strip()
                if line and len(line.split()) <= 4 and not any(char in line for char in '@+()'):
                    # Likely a name
                    parsed_data['name'] = line
                    break
        
        # Extract skills from common sections
        if not parsed_data['skills']:
            skills_section = self._extract_skills_section(text)
            if skills_section:
                parsed_data['skills'] = self.text_cleaner.normalize_skills(skills_section)
        
        # Extract summary/objective
        summary = self._extract_summary_section(text)
        if summary:
            parsed_data['summary'] = summary
        
        # Extract experience entries
        experience = self._extract_experience_section(text)
        if experience:
            parsed_data['experience'] = experience
        
        # Extract projects
        projects = self._extract_projects_section(text)
        if projects:
            parsed_data['projects'] = projects
    
    def _extract_skills_section(self, text):
        """Extract skills from skills section"""
        skills_patterns = [
            r'(?i)skills?\s*:?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\s*\n|\n[A-Z]|$)',
            r'(?i)technical\s+skills?\s*:?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\s*\n|\n[A-Z]|$)',
            r'(?i)technologies?\s*:?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\s*\n|\n[A-Z]|$)'
        ]
        
        for pattern in skills_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_summary_section(self, text):
        """Extract summary/objective section"""
        summary_patterns = [
            r'(?i)(?:professional\s+)?summary\s*:?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\s*\n|\n[A-Z]|$)',
            r'(?i)objective\s*:?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\s*\n|\n[A-Z]|$)',
            r'(?i)profile\s*:?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\s*\n|\n[A-Z]|$)'
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_experience_section(self, text):
        """Extract work experience entries"""
        experience_patterns = [
            r'(?i)(?:work\s+)?experience\s*:?\s*(.*?)(?=\n\s*(?:education|skills|projects|certifications)|$)',
            r'(?i)employment\s+history\s*:?\s*(.*?)(?=\n\s*(?:education|skills|projects|certifications)|$)'
        ]
        
        experiences = []
        
        for pattern in experience_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                exp_text = match.group(1).strip()
                # Split by job entries (look for patterns like "Company Name" or "Job Title")
                job_entries = re.split(r'\n(?=[A-Z][^a-z]*(?:\n|$))', exp_text)
                
                for entry in job_entries:
                    entry = entry.strip()
                    if len(entry) > 20:  # Filter out short entries
                        experiences.append({
                            'description': entry,
                            'title': self._extract_job_title(entry),
                            'company': self._extract_company_name(entry),
                            'duration': self._extract_duration(entry)
                        })
        
        return experiences
    
    def _extract_projects_section(self, text):
        """Extract projects section"""
        projects_patterns = [
            r'(?i)projects?\s*:?\s*(.*?)(?=\n\s*(?:education|skills|experience|certifications)|$)'
        ]
        
        projects = []
        
        for pattern in projects_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                proj_text = match.group(1).strip()
                # Split by project entries
                project_entries = re.split(r'\n(?=[A-Z][^a-z]*)', proj_text)
                
                for entry in project_entries:
                    entry = entry.strip()
                    if len(entry) > 20:
                        projects.append({
                            'name': self._extract_project_name(entry),
                            'description': entry,
                            'technologies': self._extract_technologies(entry)
                        })
        
        return projects
    
    def _extract_job_title(self, text):
        """Extract job title from experience entry"""
        lines = text.split('\n')
        for line in lines[:2]:  # Check first 2 lines
            if any(keyword in line.lower() for keyword in ['engineer', 'developer', 'manager', 'analyst', 'specialist']):
                return line.strip()
        return lines[0].strip() if lines else "Position"
    
    def _extract_company_name(self, text):
        """Extract company name from experience entry"""
        # Look for patterns like "at Company Name" or "Company Name"
        company_match = re.search(r'(?:at\s+)?([A-Z][a-zA-Z\s&.,]+(?:Inc|LLC|Corp|Ltd)?)', text)
        if company_match:
            return company_match.group(1).strip()
        return "Company"
    
    def _extract_duration(self, text):
        """Extract duration from experience entry"""
        duration_patterns = [
            r'(\d{4}\s*-\s*\d{4})',
            r'(\d{4}\s*-\s*present)',
            r'(\w+\s+\d{4}\s*-\s*\w+\s+\d{4})',
            r'(\d+\s+(?:years?|months?))'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    def _extract_project_name(self, text):
        """Extract project name"""
        lines = text.split('\n')
        return lines[0].strip() if lines else "Project"
    
    def _extract_technologies(self, text):
        """Extract technologies from project description"""
        tech_keywords = [
            'python', 'java', 'javascript', 'react', 'node.js', 'angular', 'vue',
            'django', 'flask', 'spring', 'express', 'mongodb', 'mysql', 'postgresql',
            'aws', 'azure', 'docker', 'kubernetes', 'git', 'html', 'css'
        ]
        
        technologies = []
        text_lower = text.lower()
        
        for tech in tech_keywords:
            if tech in text_lower:
                technologies.append(tech.title())
        
        return technologies
    
    def extract_contact_info(self, text):
        """Extract only contact information"""
        return {
            'email': self.text_cleaner.extract_emails(text),
            'phone': self.text_cleaner.extract_phones(text),
            'linkedin': self.text_cleaner.extract_linkedin_profiles(text),
            'github': self.text_cleaner.extract_github_profiles(text)
        }
    
    def validate_parsed_data(self, parsed_data):
        """Validate and clean parsed data"""
        # Ensure required fields have default values
        defaults = {
            'name': 'Unknown',
            'email': '',
            'phone': '',
            'location': '',
            'skills': [],
            'experience_years': 0,
            'education': [],
            'certifications': [],
            'linkedin': '',
            'github': '',
            'summary': '',
            'experience': [],
            'projects': []
        }
        
        for key, default_value in defaults.items():
            if key not in parsed_data or parsed_data[key] is None:
                parsed_data[key] = default_value
        
        return parsed_data
