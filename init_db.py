#!/usr/bin/env python3
"""
Database initialization script for Resume Parser application
"""

import os
from pathlib import Path
from flask import Flask
from flask_bcrypt import Bcrypt
from models import db, User, JobSeeker, HRUser, Job, Application
from config import config

def create_app():
    """Create Flask application"""
    app = Flask(__name__)
    app.config.from_object(config['development'])
    
    # Initialize extensions
    db.init_app(app)
    
    return app

def init_database():
    """Initialize database with tables"""
    app = create_app()
    
    with app.app_context():
        # Create database directory if it doesn't exist
        db_path = Path(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create all tables (will create database file if it doesn't exist)
        print("Creating database tables...")
        db.create_all()
        
        # Create sample data
        create_sample_data()
        
        print("Database initialized successfully!")

def create_sample_data():
    """Create sample data for testing"""
    bcrypt = Bcrypt()
    
    # Check if sample data already exists
    existing_hr = User.query.filter_by(email='hr@techcorp.com').first()
    if existing_hr:
        print("Sample data already exists. Skipping creation.")
        return
    
    # Create sample HR user
    hr_user = User(
        email='hr@techcorp.com',
        password=bcrypt.generate_password_hash('password123').decode('utf-8'),
        user_type='hr'
    )
    db.session.add(hr_user)
    db.session.flush()  # Get the ID
    
    hr_profile = HRUser(
        user_id=hr_user.id,
        company_name='TechCorp Inc.',
        contact_name='Jane Smith',
        phone='+1-555-0100'
    )
    db.session.add(hr_profile)
    
    # Create sample job seeker
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
        education='BS Computer Science, Stanford University',
        linkedin='https://linkedin.com/in/johndoe',
        github='https://github.com/johndoe'
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
    db.session.flush()
    
    # Create sample job
    job = Job(
        hr_id=hr_profile.id,
        title='Senior Software Engineer',
        company='TechCorp Inc.',
        location='San Francisco, CA',
        description='We are looking for a senior software engineer with experience in Python, React, and AWS. The ideal candidate will have 3+ years of experience in full-stack development.',
        experience_required=3,
        salary_range='$120,000 - $160,000',
        employment_type='Full-time',
        status='active'
    )
    job.set_required_skills(['Python', 'React', 'AWS', 'JavaScript'])
    db.session.add(job)
    db.session.flush()
    
    # Create sample application
    application = Application(
        job_id=job.id,
        jobseeker_id=jobseeker_profile.id,
        ai_score=85.5,
        skills_match=90.0,
        experience_match=85.0,
        education_match=80.0,
        status='pending'
    )
    db.session.add(application)
    
    # Commit all changes
    db.session.commit()
    
    print("Sample data created:")
    print(f"- HR User: hr@techcorp.com (password: password123)")
    print(f"- Job Seeker: john.doe@email.com (password: password123)")
    print(f"- Sample job: {job.title}")
    print(f"- Sample application with AI score: {application.ai_score}%")

if __name__ == '__main__':
    init_database()
