import re
import string

class TextCleaner:
    def __init__(self):
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s@.+-]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_emails(self, text):
        """Extract email addresses from text"""
        return self.email_pattern.findall(text)
    
    def extract_phones(self, text):
        """Extract phone numbers from text"""
        phones = self.phone_pattern.findall(text)
        # Clean phone numbers
        cleaned_phones = []
        for phone in phones:
            # Remove non-digit characters except +
            cleaned = re.sub(r'[^\d+]', '', phone)
            if len(cleaned) >= 10:
                cleaned_phones.append(cleaned)
        return cleaned_phones
    
    def extract_urls(self, text):
        """Extract URLs from text"""
        return self.url_pattern.findall(text)
    
    def extract_linkedin_profiles(self, text):
        """Extract LinkedIn profile URLs"""
        linkedin_pattern = re.compile(r'linkedin\.com/in/[a-zA-Z0-9-]+')
        return linkedin_pattern.findall(text)
    
    def extract_github_profiles(self, text):
        """Extract GitHub profile URLs"""
        github_pattern = re.compile(r'github\.com/[a-zA-Z0-9-]+')
        return github_pattern.findall(text)
    
    def normalize_skills(self, skills_text):
        """Normalize and extract skills from text"""
        if not skills_text:
            return []
        
        # Common skill separators
        skills = re.split(r'[,;|•\n]', skills_text)
        
        # Clean each skill
        normalized_skills = []
        for skill in skills:
            skill = skill.strip()
            if skill and len(skill) > 1:
                # Remove common prefixes
                skill = re.sub(r'^[-•\s]*', '', skill)
                skill = re.sub(r'\s+', ' ', skill)
                normalized_skills.append(skill.title())
        
        return list(set(normalized_skills))  # Remove duplicates
    
    def extract_years_of_experience(self, text):
        """Extract years of experience from text"""
        patterns = [
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(?:experience|exp).*?(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return int(matches[0])
        
        return None
    
    def extract_education_info(self, text):
        """Extract education information"""
        education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'diploma',
            'b.s.', 'b.a.', 'm.s.', 'm.a.', 'mba', 'ph.d.',
            'bs', 'ba', 'ms', 'ma', 'university', 'college',
            'institute', 'school'
        ]
        
        education_info = []
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in education_keywords):
                education_info.append(line.strip())
        
        return education_info
    
    def extract_certifications(self, text):
        """Extract certifications from text"""
        cert_keywords = [
            'certified', 'certification', 'certificate', 'aws', 'azure',
            'google cloud', 'cisco', 'microsoft', 'oracle', 'pmp',
            'scrum master', 'agile'
        ]
        
        certifications = []
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in cert_keywords):
                certifications.append(line.strip())
        
        return certifications
    
    def preprocess_for_ml(self, text):
        """Preprocess text for ML models"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
