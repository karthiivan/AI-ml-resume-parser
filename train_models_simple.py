#!/usr/bin/env python3
"""
Simplified ML model training script for AI Resume Parser
"""

import spacy
from spacy.training import Example
import random
import json
from pathlib import Path
import os

def create_sample_training_data():
    """Create sample training data for NER"""
    training_data = [
        ("John Doe is a Software Engineer. Email: john.doe@email.com Phone: +1-555-0123 Location: San Francisco, CA Skills: Python, JavaScript, React Experience: 5 years Education: BS Computer Science, Stanford University", 
         {"entities": [(0, 8, "PERSON"), (32, 51, "EMAIL"), (59, 71, "PHONE"), (82, 98, "LOCATION"), (107, 140, "SKILLS"), (153, 160, "EXPERIENCE_YEARS"), (172, 210, "EDUCATION")]}),
        
        ("Jane Smith, Senior Developer jane.smith@company.com +1-555-0124 New York, NY Java, Spring Boot, AWS 7 years MS Computer Science, MIT",
         {"entities": [(0, 10, "PERSON"), (30, 52, "EMAIL"), (53, 65, "PHONE"), (66, 78, "LOCATION"), (79, 104, "SKILLS"), (105, 112, "EXPERIENCE_YEARS"), (113, 140, "EDUCATION")]}),
        
        ("Mike Johnson mike.j@tech.com (555) 123-4567 Austin, TX React, Node.js, MongoDB 3 years Bachelor of Engineering, University of Texas",
         {"entities": [(0, 12, "PERSON"), (13, 29, "EMAIL"), (30, 44, "PHONE"), (45, 55, "LOCATION"), (56, 80, "SKILLS"), (81, 88, "EXPERIENCE_YEARS"), (89, 139, "EDUCATION")]}),
        
        ("Sarah Wilson sarah.wilson@startup.io 555.987.6543 Seattle, WA Python, Django, PostgreSQL 4 years PhD Computer Science, Stanford",
         {"entities": [(0, 12, "PERSON"), (13, 37, "EMAIL"), (38, 50, "PHONE"), (51, 62, "LOCATION"), (63, 91, "SKILLS"), (92, 99, "EXPERIENCE_YEARS"), (100, 133, "EDUCATION")]}),
        
        ("David Brown david@company.org +1 (555) 234-5678 Boston, MA JavaScript, Angular, Docker 6 years MS Software Engineering, Harvard",
         {"entities": [(0, 11, "PERSON"), (12, 30, "EMAIL"), (31, 47, "PHONE"), (48, 58, "LOCATION"), (59, 86, "SKILLS"), (87, 94, "EXPERIENCE_YEARS"), (95, 133, "EDUCATION")]})
    ]
    
    return training_data * 10  # Repeat to have more training data

def train_ner_model():
    """Train a simple NER model"""
    print("🤖 Training NER Model...")
    
    # Load base model
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✓ Loaded base spaCy model")
    except OSError:
        nlp = spacy.blank("en")
        print("✓ Created blank spaCy model")
    
    # Get or create NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add labels
    labels = ["PERSON", "EMAIL", "PHONE", "LOCATION", "SKILLS", "EXPERIENCE_YEARS", "EDUCATION"]
    for label in labels:
        ner.add_label(label)
    
    print(f"✓ Added labels: {labels}")
    
    # Get training data
    training_data = create_sample_training_data()
    print(f"✓ Created {len(training_data)} training examples")
    
    # Convert to spaCy format
    examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    
    # Train the model
    print("🏋️ Training model...")
    
    # Disable other pipes during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        # Initialize with examples
        nlp.initialize(lambda: examples)
        
        # Training loop
        for i in range(10):  # Reduced iterations for speed
            random.shuffle(examples)
            losses = {}
            nlp.update(examples, losses=losses, drop=0.2)
            print(f"   Iteration {i+1}/10, Loss: {losses.get('ner', 0):.2f}")
    
    # Save model
    model_path = Path("ml_models/trained_ner_model")
    model_path.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(model_path)
    print(f"✅ NER model saved to {model_path}")
    
    # Test the model
    test_text = "Alice Johnson alice@test.com +1-555-9999 Chicago, IL Python, React 2 years BS Computer Science"
    doc = nlp(test_text)
    print("\n🧪 Testing model:")
    print(f"Text: {test_text}")
    print("Entities found:")
    for ent in doc.ents:
        print(f"  {ent.text} -> {ent.label_}")
    
    return True

def create_simple_job_matcher():
    """Create a simple rule-based job matcher"""
    print("\n🎯 Creating Job Matcher...")
    
    matcher_code = '''
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

class SimpleJobMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
    def calculate_match_score(self, resume_data, job_data):
        """Calculate match score between resume and job"""
        try:
            # Skills matching (40% weight)
            skills_score = self._calculate_skills_match(resume_data, job_data)
            
            # Experience matching (30% weight)
            experience_score = self._calculate_experience_match(resume_data, job_data)
            
            # Education matching (20% weight)
            education_score = self._calculate_education_match(resume_data, job_data)
            
            # Location matching (10% weight)
            location_score = self._calculate_location_match(resume_data, job_data)
            
            # Calculate weighted average
            overall_score = (
                skills_score * 0.4 +
                experience_score * 0.3 +
                education_score * 0.2 +
                location_score * 0.1
            )
            
            return {
                'overall_score': overall_score,
                'skills_match': skills_score,
                'experience_match': experience_score,
                'education_match': education_score,
                'location_match': location_score
            }
        except Exception as e:
            print(f"Error in matching: {e}")
            return {'overall_score': 50, 'skills_match': 50, 'experience_match': 50, 'education_match': 50, 'location_match': 50}
    
    def _calculate_skills_match(self, resume_data, job_data):
        """Calculate skills matching percentage"""
        resume_skills = set()
        job_skills = set()
        
        # Extract resume skills
        if isinstance(resume_data.get('skills'), list):
            resume_skills = set(skill.lower().strip() for skill in resume_data['skills'])
        elif isinstance(resume_data.get('skills'), str):
            resume_skills = set(skill.lower().strip() for skill in resume_data['skills'].split(','))
        
        # Extract job skills
        if isinstance(job_data.get('required_skills'), list):
            job_skills = set(skill.lower().strip() for skill in job_data['required_skills'])
        elif isinstance(job_data.get('required_skills'), str):
            job_skills = set(skill.lower().strip() for skill in job_data['required_skills'].split(','))
        
        if not job_skills:
            return 50.0
        
        # Calculate matches
        matches = resume_skills.intersection(job_skills)
        score = (len(matches) / len(job_skills)) * 100
        
        return min(score, 100.0)
    
    def _calculate_experience_match(self, resume_data, job_data):
        """Calculate experience matching percentage"""
        resume_exp = resume_data.get('experience_years', 0)
        required_exp = job_data.get('experience_required', 0)
        
        if required_exp == 0:
            return 100.0
        
        if resume_exp >= required_exp:
            return 100.0
        else:
            return (resume_exp / required_exp) * 100
    
    def _calculate_education_match(self, resume_data, job_data):
        """Calculate education matching percentage"""
        if resume_data.get('education'):
            return 80.0  # Basic education match
        return 50.0
    
    def _calculate_location_match(self, resume_data, job_data):
        """Calculate location matching percentage"""
        resume_location = resume_data.get('location', '').lower()
        job_location = job_data.get('location', '').lower()
        
        if not resume_location or not job_location:
            return 50.0
        
        if resume_location in job_location or job_location in resume_location:
            return 100.0
        
        return 30.0
    
    def get_matching_analysis(self, resume_data, job_data):
        """Get detailed matching analysis"""
        result = self.calculate_match_score(resume_data, job_data)
        
        # Get matching and missing skills
        resume_skills = set()
        job_skills = set()
        
        if isinstance(resume_data.get('skills'), list):
            resume_skills = set(skill.lower().strip() for skill in resume_data['skills'])
        
        if isinstance(job_data.get('required_skills'), list):
            job_skills = set(skill.lower().strip() for skill in job_data['required_skills'])
        
        matching_skills = list(resume_skills.intersection(job_skills))
        missing_skills = list(job_skills - resume_skills)
        
        score = result['overall_score']
        if score >= 80:
            recommendation = "Highly Recommended - Shortlist"
        elif score >= 60:
            recommendation = "Good Fit - Consider Interview"
        else:
            recommendation = "Below Requirements - Consider Rejecting"
        
        result.update({
            'breakdown': result,
            'matching_skills': matching_skills,
            'missing_skills': missing_skills,
            'recommendation': recommendation
        })
        
        return result
'''
    
    # Save the matcher
    matcher_path = Path("ml_models/trained_matcher")
    matcher_path.mkdir(parents=True, exist_ok=True)
    
    with open(matcher_path / "job_matcher.py", "w") as f:
        f.write(matcher_code)
    
    # Create __init__.py
    with open(matcher_path / "__init__.py", "w") as f:
        f.write("from .job_matcher import SimpleJobMatcher")
    
    print(f"✅ Job matcher saved to {matcher_path}")
    return True

def initialize_database():
    """Initialize database with sample data"""
    print("\n💾 Initializing Database...")
    
    try:
        from models import db, User, JobSeeker, HRUser, Job, Application
        from flask_bcrypt import Bcrypt
        from app import create_app
        
        app = create_app()
        bcrypt = Bcrypt()
        
        with app.app_context():
            # Create tables
            db.create_all()
            print("✓ Database tables created")
            
            # Check if users already exist
            existing_jobseeker = User.query.filter_by(email='john.doe@email.com').first()
            existing_hr = User.query.filter_by(email='hr@techcorp.com').first()
            
            if not existing_jobseeker:
                # Create job seeker user
                jobseeker_user = User(
                    email='john.doe@email.com',
                    password=bcrypt.generate_password_hash('password123').decode('utf-8'),
                    user_type='jobseeker'
                )
                db.session.add(jobseeker_user)
                db.session.flush()
                
                jobseeker_profile = JobSeeker(
                    user_id=jobseeker_user.id,
                    name='John Doe',
                    phone='+1-555-0123',
                    location='San Francisco, CA',
                    experience_years=5,
                    education='BS Computer Science, Stanford University'
                )
                jobseeker_profile.set_skills(['Python', 'JavaScript', 'React', 'Node.js', 'AWS'])
                jobseeker_profile.set_parsed_data({
                    'name': 'John Doe',
                    'email': 'john.doe@email.com',
                    'phone': '+1-555-0123',
                    'location': 'San Francisco, CA',
                    'skills': ['Python', 'JavaScript', 'React', 'Node.js', 'AWS'],
                    'experience_years': 5,
                    'education': ['BS Computer Science, Stanford University'],
                    'summary': 'Experienced software engineer with 5 years of full-stack development experience.'
                })
                db.session.add(jobseeker_profile)
                print("✓ Created Job Seeker demo account")
            else:
                print("✓ Job Seeker demo account already exists")
            
            if not existing_hr:
                # Create HR user
                hr_user = User(
                    email='hr@techcorp.com',
                    password=bcrypt.generate_password_hash('password123').decode('utf-8'),
                    user_type='hr'
                )
                db.session.add(hr_user)
                db.session.flush()
                
                hr_profile = HRUser(
                    user_id=hr_user.id,
                    company_name='TechCorp Inc.',
                    contact_name='Jane Smith',
                    phone='+1-555-0100'
                )
                db.session.add(hr_profile)
                print("✓ Created HR demo account")
            else:
                print("✓ HR demo account already exists")
            
            # Create sample jobs if they don't exist
            existing_job = Job.query.filter_by(title='Senior Software Engineer').first()
            if not existing_job and existing_hr:
                hr_profile = HRUser.query.filter_by(user_id=existing_hr.id).first()
                if hr_profile:
                    job = Job(
                        hr_id=hr_profile.id,
                        title='Senior Software Engineer',
                        company='TechCorp Inc.',
                        location='San Francisco, CA',
                        description='We are looking for a senior software engineer with experience in Python, React, and AWS.',
                        experience_required=3,
                        salary_range='$120,000 - $160,000',
                        employment_type='Full-time',
                        status='active'
                    )
                    job.set_required_skills(['Python', 'React', 'AWS', 'JavaScript'])
                    db.session.add(job)
                    print("✓ Created sample job")
            
            db.session.commit()
            print("✅ Database initialized with demo data")
            
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 AI Resume Parser - Full ML Setup")
    print("=" * 50)
    
    success_count = 0
    total_steps = 3
    
    # Step 1: Train NER model
    if train_ner_model():
        success_count += 1
    
    # Step 2: Create job matcher
    if create_simple_job_matcher():
        success_count += 1
    
    # Step 3: Initialize database
    if initialize_database():
        success_count += 1
    
    print(f"\n{'='*50}")
    print(f"🎉 Setup Complete!")
    print(f"✅ {success_count}/{total_steps} steps completed successfully")
    
    if success_count == total_steps:
        print("\n🌟 Full ML setup successful!")
        print("\n📋 Demo Login Credentials:")
        print("   👤 Job Seeker:")
        print("      Email: john.doe@email.com")
        print("      Password: password123")
        print("\n   🏢 HR User:")
        print("      Email: hr@techcorp.com") 
        print("      Password: password123")
        print("\n🌐 Access the application at: http://localhost:5000")
        print("   The ML models are now active and ready to use!")
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("   Check the error messages above for details.")

if __name__ == "__main__":
    main()
