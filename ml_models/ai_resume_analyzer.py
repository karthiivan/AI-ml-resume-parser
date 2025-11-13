"""
AI-Powered Resume Analyzer
Analyzes resumes and provides intelligent matching scores for job positions
"""

import re
import json
from typing import Dict, List, Tuple
from datetime import datetime

class AIResumeAnalyzer:
    def __init__(self):
        """Initialize the AI Resume Analyzer"""
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'matplotlib', 'seaborn'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'sqlite'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins'],
            'mobile': ['android', 'ios', 'react native', 'flutter', 'xamarin', 'kotlin', 'swift'],
            'design': ['photoshop', 'illustrator', 'figma', 'sketch', 'ui/ux', 'adobe creative suite'],
            'project_management': ['agile', 'scrum', 'kanban', 'jira', 'trello', 'asana', 'project management']
        }
        
        self.experience_keywords = [
            'years', 'experience', 'worked', 'developed', 'managed', 'led', 'created', 
            'implemented', 'designed', 'built', 'maintained', 'optimized', 'improved'
        ]
        
        self.education_levels = {
            'phd': 10, 'doctorate': 10, 'ph.d': 10,
            'masters': 8, 'master': 8, 'mba': 8, 'm.s': 8, 'm.a': 8,
            'bachelors': 6, 'bachelor': 6, 'b.s': 6, 'b.a': 6, 'b.tech': 6, 'b.e': 6,
            'associates': 4, 'associate': 4, 'diploma': 3, 'certificate': 2
        }
        
        print("✓ AI Resume Analyzer initialized")
    
    def analyze_resume(self, resume_text: str, job_requirements: Dict) -> Dict:
        """
        Comprehensive AI analysis of resume against job requirements
        
        Args:
            resume_text: Extracted text from resume
            job_requirements: Job requirements including skills, experience, etc.
            
        Returns:
            Detailed analysis with scores and recommendations
        """
        try:
            # Extract structured data from resume
            extracted_data = self._extract_resume_data(resume_text)
            
            # Calculate various matching scores
            skills_analysis = self._analyze_skills_match(extracted_data['skills'], job_requirements.get('required_skills', []))
            experience_analysis = self._analyze_experience_match(extracted_data['experience'], job_requirements.get('experience_required', 0))
            education_analysis = self._analyze_education_match(extracted_data['education'], job_requirements.get('education_required', ''))
            
            # Calculate overall AI score
            overall_score = self._calculate_overall_score(skills_analysis, experience_analysis, education_analysis)
            
            # Generate AI recommendations
            recommendations = self._generate_recommendations(skills_analysis, experience_analysis, education_analysis)
            
            # Determine candidate worthiness
            worthiness = self._determine_worthiness(overall_score, skills_analysis, experience_analysis)
            
            return {
                'overall_score': overall_score,
                'worthiness': worthiness,
                'skills_analysis': skills_analysis,
                'experience_analysis': experience_analysis,
                'education_analysis': education_analysis,
                'extracted_data': extracted_data,
                'recommendations': recommendations,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in AI resume analysis: {e}")
            return self._get_fallback_analysis()
    
    def _extract_resume_data(self, resume_text: str) -> Dict:
        """Extract structured data from resume text using AI techniques"""
        resume_text = resume_text.lower()
        
        # Extract skills
        skills = self._extract_skills(resume_text)
        
        # Extract experience
        experience = self._extract_experience(resume_text)
        
        # Extract education
        education = self._extract_education(resume_text)
        
        # Extract contact info
        contact_info = self._extract_contact_info(resume_text)
        
        return {
            'skills': skills,
            'experience': experience,
            'education': education,
            'contact_info': contact_info
        }
    
    def _extract_skills(self, resume_text: str) -> List[str]:
        """Extract skills from resume text"""
        found_skills = []
        
        for category, skills_list in self.skill_categories.items():
            for skill in skills_list:
                if skill.lower() in resume_text:
                    found_skills.append(skill)
        
        # Also look for skills in common patterns
        skill_patterns = [
            r'skills?[:\-\s]+(.*?)(?:\n|$)',
            r'technologies?[:\-\s]+(.*?)(?:\n|$)',
            r'programming languages?[:\-\s]+(.*?)(?:\n|$)'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            for match in matches:
                # Split by common separators
                skills_in_match = re.split(r'[,;|•\n]', match)
                for skill in skills_in_match:
                    skill = skill.strip()
                    if skill and len(skill) > 2:
                        found_skills.append(skill)
        
        return list(set(found_skills))  # Remove duplicates
    
    def _extract_experience(self, resume_text: str) -> Dict:
        """Extract experience information from resume"""
        # Look for years of experience
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience[:\-\s]*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in\s*\w+'
        ]
        
        years_experience = 0
        for pattern in experience_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            if matches:
                years_experience = max(years_experience, int(matches[0]))
        
        # Count job positions (rough estimate)
        job_indicators = len(re.findall(r'(?:software engineer|developer|analyst|manager|specialist|consultant)', resume_text, re.IGNORECASE))
        
        return {
            'years': years_experience,
            'positions_count': job_indicators,
            'has_leadership': any(word in resume_text for word in ['led', 'managed', 'supervised', 'directed'])
        }
    
    def _extract_education(self, resume_text: str) -> Dict:
        """Extract education information from resume"""
        education_info = {
            'highest_degree': '',
            'score': 0,
            'institutions': []
        }
        
        # Find education levels
        for degree, score in self.education_levels.items():
            if degree in resume_text:
                if score > education_info['score']:
                    education_info['highest_degree'] = degree
                    education_info['score'] = score
        
        # Find institutions
        university_patterns = [
            r'university of \w+',
            r'\w+ university',
            r'\w+ institute of technology',
            r'\w+ college'
        ]
        
        for pattern in university_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            education_info['institutions'].extend(matches)
        
        return education_info
    
    def _extract_contact_info(self, resume_text: str) -> Dict:
        """Extract contact information from resume"""
        contact_info = {}
        
        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, resume_text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Phone
        phone_pattern = r'[\+]?[1-9]?[0-9]{7,15}'
        phones = re.findall(phone_pattern, resume_text)
        if phones:
            contact_info['phone'] = phones[0]
        
        return contact_info
    
    def _analyze_skills_match(self, candidate_skills: List[str], required_skills: List[str]) -> Dict:
        """Analyze how well candidate skills match job requirements"""
        if not required_skills:
            return {'score': 75, 'matched_skills': [], 'missing_skills': [], 'bonus_skills': candidate_skills}
        
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        required_skills_lower = [skill.lower() for skill in required_skills]
        
        matched_skills = []
        missing_skills = []
        
        for req_skill in required_skills_lower:
            if any(req_skill in cand_skill or cand_skill in req_skill for cand_skill in candidate_skills_lower):
                matched_skills.append(req_skill)
            else:
                missing_skills.append(req_skill)
        
        # Calculate score
        if len(required_skills_lower) == 0:
            score = 75
        else:
            score = (len(matched_skills) / len(required_skills_lower)) * 100
        
        # Bonus for additional relevant skills
        bonus_skills = [skill for skill in candidate_skills if skill.lower() not in required_skills_lower]
        
        return {
            'score': min(100, score + len(bonus_skills) * 2),  # Bonus points for extra skills
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'bonus_skills': bonus_skills
        }
    
    def _analyze_experience_match(self, candidate_experience: Dict, required_experience: int) -> Dict:
        """Analyze experience match"""
        candidate_years = candidate_experience.get('years', 0)
        
        if required_experience == 0:
            score = 80
        elif candidate_years >= required_experience:
            # Bonus for exceeding requirements
            excess_years = candidate_years - required_experience
            score = min(100, 85 + (excess_years * 3))
        else:
            # Penalty for not meeting requirements
            shortage = required_experience - candidate_years
            score = max(30, 85 - (shortage * 15))
        
        return {
            'score': score,
            'candidate_years': candidate_years,
            'required_years': required_experience,
            'has_leadership': candidate_experience.get('has_leadership', False),
            'positions_count': candidate_experience.get('positions_count', 0)
        }
    
    def _analyze_education_match(self, candidate_education: Dict, required_education: str) -> Dict:
        """Analyze education match"""
        candidate_score = candidate_education.get('score', 0)
        
        if not required_education:
            return {'score': 70, 'candidate_degree': candidate_education.get('highest_degree', ''), 'meets_requirement': True}
        
        required_education_lower = required_education.lower()
        
        # Determine required education score
        required_score = 0
        for degree, score in self.education_levels.items():
            if degree in required_education_lower:
                required_score = max(required_score, score)
        
        if candidate_score >= required_score:
            score = min(100, 80 + (candidate_score - required_score) * 2)
            meets_requirement = True
        else:
            score = max(40, 80 - (required_score - candidate_score) * 10)
            meets_requirement = False
        
        return {
            'score': score,
            'candidate_degree': candidate_education.get('highest_degree', ''),
            'meets_requirement': meets_requirement
        }
    
    def _calculate_overall_score(self, skills_analysis: Dict, experience_analysis: Dict, education_analysis: Dict) -> float:
        """Calculate weighted overall score"""
        # Weights: Skills 50%, Experience 30%, Education 20%
        overall_score = (
            skills_analysis['score'] * 0.5 +
            experience_analysis['score'] * 0.3 +
            education_analysis['score'] * 0.2
        )
        
        return round(overall_score, 1)
    
    def _determine_worthiness(self, overall_score: float, skills_analysis: Dict, experience_analysis: Dict) -> Dict:
        """Determine if candidate is worthy based on AI analysis"""
        
        # Define worthiness criteria
        if overall_score >= 85:
            status = "Highly Recommended"
            confidence = "Very High"
            reason = "Exceptional match with strong skills and experience alignment"
        elif overall_score >= 75:
            status = "Recommended"
            confidence = "High"
            reason = "Good match with solid qualifications"
        elif overall_score >= 65:
            status = "Consider"
            confidence = "Medium"
            reason = "Decent match but may need additional evaluation"
        elif overall_score >= 50:
            status = "Weak Match"
            confidence = "Low"
            reason = "Limited alignment with job requirements"
        else:
            status = "Not Recommended"
            confidence = "Very Low"
            reason = "Poor match with significant gaps in requirements"
        
        return {
            'status': status,
            'confidence': confidence,
            'reason': reason,
            'is_worthy': overall_score >= 65  # Threshold for worthiness
        }
    
    def _generate_recommendations(self, skills_analysis: Dict, experience_analysis: Dict, education_analysis: Dict) -> List[str]:
        """Generate AI-powered recommendations"""
        recommendations = []
        
        # Skills recommendations
        if skills_analysis['score'] < 70:
            missing_count = len(skills_analysis['missing_skills'])
            if missing_count > 0:
                recommendations.append(f"Candidate lacks {missing_count} key required skills")
        
        # Experience recommendations
        if experience_analysis['score'] < 70:
            if experience_analysis['candidate_years'] < experience_analysis['required_years']:
                shortage = experience_analysis['required_years'] - experience_analysis['candidate_years']
                recommendations.append(f"Candidate has {shortage} years less experience than required")
        
        # Education recommendations
        if not education_analysis['meets_requirement']:
            recommendations.append("Education level may not meet job requirements")
        
        # Positive recommendations
        if skills_analysis['score'] >= 85:
            recommendations.append("Strong technical skills alignment")
        
        if experience_analysis['has_leadership']:
            recommendations.append("Candidate has leadership experience")
        
        if len(skills_analysis['bonus_skills']) > 3:
            recommendations.append("Candidate brings additional valuable skills")
        
        return recommendations
    
    def _get_fallback_analysis(self) -> Dict:
        """Return fallback analysis in case of errors"""
        return {
            'overall_score': 50.0,
            'worthiness': {
                'status': 'Needs Manual Review',
                'confidence': 'Low',
                'reason': 'Automated analysis failed',
                'is_worthy': False
            },
            'skills_analysis': {'score': 50, 'matched_skills': [], 'missing_skills': [], 'bonus_skills': []},
            'experience_analysis': {'score': 50, 'candidate_years': 0, 'required_years': 0},
            'education_analysis': {'score': 50, 'meets_requirement': False},
            'extracted_data': {'skills': [], 'experience': {}, 'education': {}, 'contact_info': {}},
            'recommendations': ['Manual review required due to analysis error'],
            'analysis_timestamp': datetime.now().isoformat()
        }
