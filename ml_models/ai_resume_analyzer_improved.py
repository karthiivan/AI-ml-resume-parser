#!/usr/bin/env python3
"""
Enhanced AI Resume Analyzer with Improved Models
Uses the improved skills extractor and NER model for better accuracy
"""

import pickle
import re
import json
import numpy as np
from pathlib import Path

class AIResumeAnalyzer:
    def __init__(self):
        self.skills_model = None
        self.ner_model = None
        self.load_improved_models()
    
    def load_improved_models(self):
        """Load the improved ML models"""
        try:
            # Load improved skills extractor
            with open('ml_models/trained/skills_extractor_improved.pkl', 'rb') as f:
                self.skills_model = pickle.load(f)
            print("✅ Loaded improved skills extractor")
            
            # Load improved NER model
            with open('ml_models/trained/ner_model_improved.pkl', 'rb') as f:
                self.ner_model = pickle.load(f)
            print("✅ Loaded improved NER model")
            
        except Exception as e:
            print(f"⚠️ Could not load improved models: {e}")
            print("   Falling back to basic analysis...")
            self.skills_model = None
            self.ner_model = None
    
    def extract_skills(self, text):
        """Extract skills using improved model"""
        if not self.skills_model:
            return self._basic_skills_extraction(text)
        
        detected_skills = []
        confidence_scores = {}
        
        text_lower = text.lower()
        
        # ML-based extraction
        ml_models = self.skills_model.get('ml_models', {})
        for skill, model_data in ml_models.items():
            try:
                model = model_data['model']
                vectorizer = model_data['vectorizer']
                
                # Transform text
                text_vec = vectorizer.transform([text_lower])
                
                # Predict
                prediction = model.predict(text_vec)[0]
                confidence = max(model.predict_proba(text_vec)[0])
                
                if prediction == 1 and confidence > 0.7:
                    detected_skills.append(skill)
                    confidence_scores[skill] = confidence
                    
            except Exception as e:
                continue
        
        # Rule-based extraction
        rule_patterns = self.skills_model.get('rule_patterns', {})
        for skill, patterns in rule_patterns.items():
            if skill not in detected_skills:
                for pattern in patterns:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        detected_skills.append(skill)
                        confidence_scores[skill] = 0.85
                        break
        
        return detected_skills, confidence_scores
    
    def extract_entities(self, text):
        """Extract entities using improved NER model"""
        if not self.ner_model:
            return self._basic_entity_extraction(text)
        
        entities = {}
        patterns = self.ner_model.get('patterns', {})
        
        for entity_type, entity_patterns in patterns.items():
            found_entities = []
            
            for pattern in entity_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if match.groups():
                        entity_text = match.group(1).strip()
                    else:
                        entity_text = match.group(0).strip()
                    
                    if entity_text and len(entity_text) > 1:
                        found_entities.append({
                            'text': entity_text,
                            'start': match.start(),
                            'end': match.end(),
                            'confidence': 0.9
                        })
            
            if found_entities:
                # Remove duplicates
                unique_entities = []
                seen_texts = set()
                for entity in found_entities:
                    if entity['text'].lower() not in seen_texts:
                        unique_entities.append(entity)
                        seen_texts.add(entity['text'].lower())
                
                entities[entity_type] = unique_entities
        
        return entities
    
    def analyze_resume(self, resume_text, job_requirements=None):
        """Comprehensive resume analysis with improved models"""
        analysis = {
            'skills_analysis': {},
            'entity_extraction': {},
            'experience_analysis': {},
            'education_analysis': {},
            'overall_score': 0,
            'worthiness': {},
            'recommendations': []
        }
        
        # Extract skills
        detected_skills, skill_confidences = self.extract_skills(resume_text)
        
        # Extract entities
        entities = self.extract_entities(resume_text)
        
        # Skills analysis
        if job_requirements and 'required_skills' in job_requirements:
            required_skills = job_requirements['required_skills']
            if isinstance(required_skills, str):
                required_skills = [s.strip() for s in required_skills.split(',')]
            
            matched_skills = []
            for req_skill in required_skills:
                for detected_skill in detected_skills:
                    if req_skill.lower() in detected_skill.lower() or detected_skill.lower() in req_skill.lower():
                        matched_skills.append(detected_skill)
                        break
            
            skills_score = len(matched_skills) / len(required_skills) if required_skills else 0
            
            analysis['skills_analysis'] = {
                'detected_skills': detected_skills,
                'required_skills': required_skills,
                'matched_skills': matched_skills,
                'score': skills_score,
                'confidence_scores': skill_confidences
            }
        else:
            analysis['skills_analysis'] = {
                'detected_skills': detected_skills,
                'score': min(len(detected_skills) / 10, 1.0),  # Normalize to 0-1
                'confidence_scores': skill_confidences
            }
        
        # Entity extraction
        analysis['entity_extraction'] = entities
        
        # Experience analysis
        experience_years = self._extract_experience_years(resume_text)
        required_experience = job_requirements.get('experience_required', 0) if job_requirements else 0
        
        if required_experience > 0:
            experience_score = min(experience_years / required_experience, 1.0)
        else:
            experience_score = min(experience_years / 5, 1.0)  # Normalize to 5 years
        
        analysis['experience_analysis'] = {
            'years_detected': experience_years,
            'years_required': required_experience,
            'score': experience_score
        }
        
        # Education analysis
        education_info = entities.get('EDUCATION', [])
        education_score = 0.7 if education_info else 0.3  # Basic scoring
        
        analysis['education_analysis'] = {
            'education_found': education_info,
            'score': education_score
        }
        
        # Overall score calculation
        weights = {
            'skills': 0.4,
            'experience': 0.3,
            'education': 0.2,
            'completeness': 0.1
        }
        
        completeness_score = len(entities) / 9  # 9 entity types
        
        overall_score = (
            analysis['skills_analysis']['score'] * weights['skills'] +
            analysis['experience_analysis']['score'] * weights['experience'] +
            analysis['education_analysis']['score'] * weights['education'] +
            completeness_score * weights['completeness']
        )
        
        analysis['overall_score'] = overall_score
        
        # Worthiness determination
        if overall_score >= 0.8:
            worthiness_status = "Highly Recommended"
            worthiness_reason = "Excellent match with strong skills and experience"
        elif overall_score >= 0.6:
            worthiness_status = "Recommended"
            worthiness_reason = "Good match with relevant skills and experience"
        elif overall_score >= 0.4:
            worthiness_status = "Consider"
            worthiness_reason = "Partial match, may need additional evaluation"
        else:
            worthiness_status = "Not Recommended"
            worthiness_reason = "Limited match with requirements"
        
        analysis['worthiness'] = {
            'status': worthiness_status,
            'reason': worthiness_reason,
            'score': overall_score
        }
        
        # Generate recommendations
        recommendations = []
        
        if analysis['skills_analysis']['score'] < 0.6:
            recommendations.append("Consider developing additional technical skills")
        
        if analysis['experience_analysis']['score'] < 0.5:
            recommendations.append("Gain more relevant work experience")
        
        if not education_info:
            recommendations.append("Include educational background information")
        
        if len(detected_skills) < 5:
            recommendations.append("Highlight more technical skills and competencies")
        
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def _extract_experience_years(self, text):
        """Extract years of experience from text"""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)\b',
            r'(?:experience|exp):\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*(?:in|with|of)\s+\w+',
            r'(?:over|more than)\s+(\d+)\s+years?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return int(matches[0])
        
        return 0
    
    def _basic_skills_extraction(self, text):
        """Basic skills extraction fallback"""
        basic_skills = [
            'Python', 'Java', 'JavaScript', 'React', 'Node.js', 'AWS', 'Docker',
            'MySQL', 'PostgreSQL', 'MongoDB', 'Git', 'Linux', 'HTML', 'CSS'
        ]
        
        detected = []
        confidences = {}
        
        text_lower = text.lower()
        for skill in basic_skills:
            if skill.lower() in text_lower:
                detected.append(skill)
                confidences[skill] = 0.8
        
        return detected, confidences
    
    def _basic_entity_extraction(self, text):
        """Basic entity extraction fallback"""
        entities = {}
        
        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            entities['EMAIL'] = [{'text': email, 'confidence': 0.9} for email in emails]
        
        # Phone
        phone_pattern = r'\+?1[-\.\s]?\(?[0-9]{3}\)?[-\.\s]?[0-9]{3}[-\.\s]?[0-9]{4}'
        phones = re.findall(phone_pattern, text)
        if phones:
            entities['PHONE'] = [{'text': phone, 'confidence': 0.9} for phone in phones]
        
        return entities

# Global instance
ai_analyzer = AIResumeAnalyzer()

def analyze_resume_with_improved_models(resume_text, job_requirements=None):
    """Main function to analyze resume with improved models"""
    return ai_analyzer.analyze_resume(resume_text, job_requirements)
