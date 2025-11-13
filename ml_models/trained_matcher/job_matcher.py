
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
