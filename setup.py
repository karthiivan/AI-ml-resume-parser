#!/usr/bin/env python3
"""
Setup script for AI Resume Parser
This script will:
1. Install required packages
2. Download datasets
3. Train ML models
4. Initialize database
5. Create necessary directories
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'ml_models/trained_ner_model',
        'ml_models/trained_matcher',
        'ml_models/trained_classifier',
        'uploads/resumes',
        'uploads/generated',
        'uploads/reports',
        'database',
        'static/images'
    ]
    
    print("\n📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {directory}")
    
    return True

def install_requirements():
    """Install Python requirements"""
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python requirements"
    )

def download_spacy_model():
    """Download spaCy English model"""
    return run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Downloading spaCy English model"
    )

def setup_datasets():
    """Setup datasets"""
    print("\n📊 Setting up datasets...")
    
    try:
        from utils.dataset_loader import DatasetLoader
        loader = DatasetLoader()
        
        print("   📥 Loading Resume Entities NER dataset...")
        ner_data = loader.load_resume_entities_ner()
        print(f"   ✓ Loaded {len(ner_data)} NER training samples")
        
        print("   📥 Loading Resume dataset...")
        resume_df = loader.load_resume_dataset()
        print(f"   ✓ Loaded {len(resume_df)} resume samples")
        
        print("   📥 Loading Job descriptions dataset...")
        job_df = loader.load_job_descriptions()
        print(f"   ✓ Loaded {len(job_df)} job samples")
        
        print("   📥 Loading Skills dataset...")
        skills_df = loader.load_skills_dataset()
        print(f"   ✓ Loaded {len(skills_df)} skills")
        
        return True
    except Exception as e:
        print(f"   ❌ Dataset setup failed: {e}")
        return False

def train_ner_model():
    """Train NER model"""
    return run_command(
        f"{sys.executable} ml_models/train_ner.py",
        "Training NER model (this may take a few minutes)"
    )

def train_matcher_model():
    """Train job matcher model"""
    return run_command(
        f"{sys.executable} ml_models/train_matcher.py",
        "Training job matcher model (this may take several minutes)"
    )

def initialize_database():
    """Initialize database"""
    return run_command(
        f"{sys.executable} init_db.py",
        "Initializing database with sample data"
    )

def create_sample_files():
    """Create sample files for testing"""
    print("\n📄 Creating sample files...")
    
    # Create sample resume
    sample_resume = """
John Doe
Software Engineer

Email: john.doe@email.com
Phone: +1-555-0123
Location: San Francisco, CA
LinkedIn: https://linkedin.com/in/johndoe
GitHub: https://github.com/johndoe

PROFESSIONAL SUMMARY
Experienced software engineer with 5 years of full-stack development experience.
Passionate about building scalable web applications and working with cutting-edge technologies.

SKILLS
• Programming Languages: Python, JavaScript, Java, TypeScript
• Frontend: React, Vue.js, HTML5, CSS3, Bootstrap
• Backend: Node.js, Django, Flask, Express.js
• Databases: PostgreSQL, MongoDB, Redis
• Cloud: AWS, Docker, Kubernetes
• Tools: Git, Jenkins, JIRA

EXPERIENCE
Senior Software Engineer | TechCorp Inc. | 2021 - Present
• Led development of microservices architecture serving 1M+ users
• Implemented CI/CD pipelines reducing deployment time by 60%
• Mentored junior developers and conducted code reviews

Software Engineer | StartupXYZ | 2019 - 2021
• Developed responsive web applications using React and Node.js
• Optimized database queries improving performance by 40%
• Collaborated with cross-functional teams in Agile environment

EDUCATION
Bachelor of Science in Computer Science
Stanford University | 2015 - 2019
GPA: 3.8/4.0

CERTIFICATIONS
• AWS Certified Developer Associate
• Certified Scrum Master (CSM)
"""
    
    sample_file = Path('uploads/resumes/sample_resume.txt')
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(sample_file, 'w') as f:
        f.write(sample_resume)
    
    print(f"   ✓ Created sample resume: {sample_file}")
    return True

def run_tests():
    """Run basic tests to verify setup"""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test NER parser
        from ml_models.ner_parser import NERParser
        parser = NERParser()
        
        sample_text = "John Doe, Software Engineer, john@email.com, +1-555-0123, San Francisco, CA"
        result = parser.parse_resume(sample_text)
        
        if result and 'name' in result:
            print("   ✓ NER parser working correctly")
        else:
            print("   ⚠️  NER parser may need attention")
        
        # Test job matcher
        from ml_models.job_matcher import JobMatcher
        matcher = JobMatcher()
        
        resume_data = {'skills': ['Python', 'JavaScript'], 'experience_years': 5}
        job_data = {'required_skills': ['Python', 'React'], 'experience_required': 3}
        
        score = matcher.calculate_match_score(resume_data, job_data)
        
        if score:
            print("   ✓ Job matcher working correctly")
        else:
            print("   ⚠️  Job matcher may need attention")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Tests failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 AI Resume Parser Setup")
    print("=" * 60)
    
    start_time = time.time()
    
    steps = [
        ("Creating directories", create_directories),
        ("Installing requirements", install_requirements),
        ("Downloading spaCy model", download_spacy_model),
        ("Setting up datasets", setup_datasets),
        ("Training NER model", train_ner_model),
        ("Training matcher model", train_matcher_model),
        ("Initializing database", initialize_database),
        ("Creating sample files", create_sample_files),
        ("Running tests", run_tests)
    ]
    
    success_count = 0
    
    for description, func in steps:
        if func():
            success_count += 1
        else:
            print(f"\n⚠️  {description} failed, but continuing with setup...")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"🎉 Setup completed!")
    print(f"{'='*60}")
    print(f"✅ {success_count}/{len(steps)} steps completed successfully")
    print(f"⏱️  Total time: {duration:.1f} seconds")
    
    if success_count >= len(steps) - 2:  # Allow 2 failures
        print("\n🎯 Setup successful! You can now run the application:")
        print("   python app.py")
        print("\n🌐 The application will be available at: http://localhost:5000")
        print("\n👤 Demo credentials:")
        print("   Job Seeker: john.doe@email.com / password123")
        print("   HR User: hr@techcorp.com / password123")
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("   The application may still work with basic functionality.")
        print("   Check the error messages above for details.")
    
    return success_count >= len(steps) - 2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
