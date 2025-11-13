import os
import pandas as pd
import json
import requests
import zipfile
from pathlib import Path

class DatasetLoader:
    def __init__(self, data_folder='data'):
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)
    
    def download_kaggle_dataset(self, dataset_name, filename):
        """Download dataset from Kaggle using kaggle API"""
        try:
            import kaggle
            kaggle.api.authenticate()
            
            # Download dataset
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=self.data_folder, 
                unzip=True
            )
            print(f"Downloaded {dataset_name} successfully")
            return True
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            return False
    
    def load_resume_entities_ner(self):
        """Load Resume Entities for NER dataset"""
        # Try to download from Kaggle
        if not self.download_kaggle_dataset('dataturks/resume-entities-for-ner', 'resume_entities_ner.json'):
            # Create sample data if download fails
            sample_data = self._create_sample_ner_data()
            with open(self.data_folder / 'resume_entities_ner.json', 'w') as f:
                json.dump(sample_data, f, indent=2)
        
        # Load the data
        try:
            with open(self.data_folder / 'resume_entities_ner.json', 'r') as f:
                return json.load(f)
        except:
            return self._create_sample_ner_data()
    
    def load_resume_dataset(self):
        """Load Resume Dataset with Classifications"""
        if not self.download_kaggle_dataset('jillianseed/resume-data', 'resume_dataset.csv'):
            # Create sample data if download fails
            sample_data = self._create_sample_resume_data()
            df = pd.DataFrame(sample_data)
            df.to_csv(self.data_folder / 'resume_dataset.csv', index=False)
        
        try:
            return pd.read_csv(self.data_folder / 'resume_dataset.csv')
        except:
            sample_data = self._create_sample_resume_data()
            df = pd.DataFrame(sample_data)
            df.to_csv(self.data_folder / 'resume_dataset.csv', index=False)
            return df
    
    def load_job_descriptions(self):
        """Load Job Descriptions Dataset"""
        if not self.download_kaggle_dataset('andrewmvd/adzuna-job-salary-predictions', 'job_descriptions.csv'):
            # Create sample data if download fails
            sample_data = self._create_sample_job_data()
            df = pd.DataFrame(sample_data)
            df.to_csv(self.data_folder / 'job_descriptions.csv', index=False)
        
        try:
            return pd.read_csv(self.data_folder / 'job_descriptions.csv')
        except:
            sample_data = self._create_sample_job_data()
            df = pd.DataFrame(sample_data)
            df.to_csv(self.data_folder / 'job_descriptions.csv', index=False)
            return df
    
    def load_skills_dataset(self):
        """Load Skills Dataset"""
        if not self.download_kaggle_dataset('nigelsmithdigital/global-skill-mapping', 'skills.csv'):
            # Create sample data if download fails
            sample_data = self._create_sample_skills_data()
            df = pd.DataFrame(sample_data)
            df.to_csv(self.data_folder / 'skills.csv', index=False)
        
        try:
            return pd.read_csv(self.data_folder / 'skills.csv')
        except:
            sample_data = self._create_sample_skills_data()
            df = pd.DataFrame(sample_data)
            df.to_csv(self.data_folder / 'skills.csv', index=False)
            return df
    
    def _create_sample_ner_data(self):
        """Create sample NER training data"""
        return [
            {
                "content": "John Doe\nSoftware Engineer\nEmail: john.doe@email.com\nPhone: +1-555-0123\nLocation: San Francisco, CA\nSkills: Python, JavaScript, React, Node.js\nExperience: 5 years\nEducation: BS Computer Science, Stanford University\nCertification: AWS Certified Developer",
                "annotation": [
                    {"label": ["PERSON"], "points": [{"start": 0, "end": 8, "text": "John Doe"}]},
                    {"label": ["EMAIL"], "points": [{"start": 32, "end": 51, "text": "john.doe@email.com"}]},
                    {"label": ["PHONE"], "points": [{"start": 59, "end": 71, "text": "+1-555-0123"}]},
                    {"label": ["LOCATION"], "points": [{"start": 82, "end": 98, "text": "San Francisco, CA"}]},
                    {"label": ["SKILLS"], "points": [{"start": 107, "end": 140, "text": "Python, JavaScript, React, Node.js"}]},
                    {"label": ["EXPERIENCE_YEARS"], "points": [{"start": 153, "end": 160, "text": "5 years"}]},
                    {"label": ["DEGREE"], "points": [{"start": 172, "end": 190, "text": "BS Computer Science"}]},
                    {"label": ["COLLEGE"], "points": [{"start": 192, "end": 210, "text": "Stanford University"}]},
                    {"label": ["CERTIFICATION"], "points": [{"start": 225, "end": 248, "text": "AWS Certified Developer"}]}
                ]
            }
        ] * 50  # Repeat for training data
    
    def _create_sample_resume_data(self):
        """Create sample resume data"""
        return [
            {
                'name': 'John Doe',
                'email': 'john.doe@email.com',
                'phone': '+1-555-0123',
                'location': 'San Francisco, CA',
                'skills': 'Python, JavaScript, React, Node.js, AWS',
                'experience_years': 5,
                'education': 'BS Computer Science, Stanford University',
                'category': 'IT',
                'resume_text': 'Experienced software engineer with 5 years of experience in full-stack development...'
            },
            {
                'name': 'Jane Smith',
                'email': 'jane.smith@email.com',
                'phone': '+1-555-0124',
                'location': 'New York, NY',
                'skills': 'Java, Spring Boot, MySQL, Docker, Kubernetes',
                'experience_years': 7,
                'education': 'MS Computer Science, MIT',
                'category': 'IT',
                'resume_text': 'Senior backend developer with expertise in microservices architecture...'
            }
        ] * 100  # Create more sample data
    
    def _create_sample_job_data(self):
        """Create sample job descriptions data"""
        return [
            {
                'title': 'Senior Software Engineer',
                'company': 'Tech Corp',
                'location': 'San Francisco, CA',
                'description': 'We are looking for a senior software engineer with experience in Python, React, and AWS...',
                'required_skills': 'Python, React, AWS, Docker',
                'experience_required': 5,
                'salary_range': '$120,000 - $160,000',
                'employment_type': 'Full-time'
            },
            {
                'title': 'Full Stack Developer',
                'company': 'StartupXYZ',
                'location': 'Austin, TX',
                'description': 'Join our growing team as a full stack developer. Experience with Node.js and React required...',
                'required_skills': 'JavaScript, Node.js, React, MongoDB',
                'experience_required': 3,
                'salary_range': '$80,000 - $120,000',
                'employment_type': 'Full-time'
            }
        ] * 50
    
    def _create_sample_skills_data(self):
        """Create sample skills data"""
        return [
            {'skill': 'Python', 'category': 'Programming Language', 'popularity': 95},
            {'skill': 'JavaScript', 'category': 'Programming Language', 'popularity': 92},
            {'skill': 'React', 'category': 'Frontend Framework', 'popularity': 88},
            {'skill': 'Node.js', 'category': 'Backend Framework', 'popularity': 85},
            {'skill': 'AWS', 'category': 'Cloud Platform', 'popularity': 90},
            {'skill': 'Docker', 'category': 'DevOps Tool', 'popularity': 82},
            {'skill': 'Kubernetes', 'category': 'DevOps Tool', 'popularity': 78},
            {'skill': 'Java', 'category': 'Programming Language', 'popularity': 89},
            {'skill': 'Spring Boot', 'category': 'Backend Framework', 'popularity': 75},
            {'skill': 'MySQL', 'category': 'Database', 'popularity': 80}
        ]
