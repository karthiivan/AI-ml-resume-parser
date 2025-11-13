from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    user_type = db.Column(db.String(20), nullable=False)  # 'jobseeker' or 'hr'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    jobseeker = db.relationship('JobSeeker', backref='user', uselist=False)
    hr_user = db.relationship('HRUser', backref='user', uselist=False)

class JobSeeker(db.Model):
    __tablename__ = 'jobseekers'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    location = db.Column(db.String(100))
    skills = db.Column(db.Text)  # JSON string
    experience_years = db.Column(db.Integer)
    education = db.Column(db.Text)
    resume_path = db.Column(db.String(200))
    linkedin = db.Column(db.String(200))
    github = db.Column(db.String(200))
    parsed_data = db.Column(db.Text)  # JSON string of all extracted info
    
    # Relationships
    applications = db.relationship('Application', backref='jobseeker', lazy=True)
    
    def get_skills(self):
        return json.loads(self.skills) if self.skills else []
    
    def set_skills(self, skills_list):
        self.skills = json.dumps(skills_list)
    
    def get_parsed_data(self):
        return json.loads(self.parsed_data) if self.parsed_data else {}
    
    def set_parsed_data(self, data):
        self.parsed_data = json.dumps(data)

class HRUser(db.Model):
    __tablename__ = 'hr_users'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    company_name = db.Column(db.String(100))
    contact_name = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    
    # Relationships
    jobs = db.relationship('Job', backref='hr_user', lazy=True)

class Job(db.Model):
    __tablename__ = 'jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    hr_id = db.Column(db.Integer, db.ForeignKey('hr_users.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    company = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100))
    description = db.Column(db.Text)
    required_skills = db.Column(db.Text)  # JSON array
    experience_required = db.Column(db.Integer)
    salary_range = db.Column(db.String(50))
    employment_type = db.Column(db.String(20))
    status = db.Column(db.String(20), default='active')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    applications = db.relationship('Application', backref='job', lazy=True)
    
    def get_required_skills(self):
        return json.loads(self.required_skills) if self.required_skills else []
    
    def set_required_skills(self, skills_list):
        self.required_skills = json.dumps(skills_list)

class Application(db.Model):
    __tablename__ = 'applications'
    
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'), nullable=False)
    jobseeker_id = db.Column(db.Integer, db.ForeignKey('jobseekers.id'), nullable=False)
    resume_path = db.Column(db.String(200))
    ai_score = db.Column(db.Float)  # 0-100
    skills_match = db.Column(db.Float)  # percentage
    experience_match = db.Column(db.Float)  # percentage
    education_match = db.Column(db.Float)  # percentage
    status = db.Column(db.String(20), default='pending')  # 'pending', 'shortlisted', 'rejected', 'selected'
    applied_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_score_color(self):
        if self.ai_score >= 80:
            return 'success'
        elif self.ai_score >= 50:
            return 'warning'
        else:
            return 'danger'
