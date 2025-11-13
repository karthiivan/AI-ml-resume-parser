import re

class JobMatcher:
    def __init__(self, model_path="ml_models/trained_matcher"):
        self.model_path = model_path
        print("✓ Simple Job Matcher initialized")
    
    def load_model(self):
        """Load the trained matcher model"""
        try:
            if self.model_path.exists():
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                
                # Load model config
                with open(self.model_path / 'config.json', 'r') as f:
                    config = json.load(f)
                
                # Initialize model architecture
                from .train_matcher import ResumeJobMatcher
                self.model = ResumeJobMatcher(config.get('model_name', 'bert-base-uncased'))
                
                # Load trained weights
                self.model.load_state_dict(torch.load(
                    self.model_path / 'model.pt',
                    map_location=self.device
                ))
                self.model.to(self.device)
                self.model.eval()
                
                print(f"Loaded trained matcher model from {self.model_path}")
            else:
                # Fallback to BERT-based similarity
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                self.model = AutoModel.from_pretrained('bert-base-uncased')
                self.model.to(self.device)
                print("Using BERT-based similarity matching (trained model not found)")
        
        except Exception as e:
            print(f"Error loading matcher model: {e}")
            # Fallback to basic similarity
            self.tokenizer = None
            self.model = None
    
    def calculate_match_score(self, resume_data, job_data):
        """Calculate overall match score between resume and job"""
        try:
            return self._calculate_rule_based_score(resume_data, job_data)
        except Exception as e:
            print(f"Error calculating match score: {e}")
            return {'overall_score': 50, 'skills_match': 50, 'experience_match': 50, 'education_match': 50, 'location_match': 50}
    
    def _calculate_ml_score(self, resume_data, job_data):
        """Calculate score using trained ML model"""
        # Prepare text inputs
        resume_text = self._prepare_resume_text(resume_data)
        job_text = self._prepare_job_text(job_data)
        
        # Tokenize inputs
        resume_encoding = self.tokenizer(
            resume_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        job_encoding = self.tokenizer(
            job_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        resume_input_ids = resume_encoding['input_ids'].to(self.device)
        resume_attention_mask = resume_encoding['attention_mask'].to(self.device)
        job_input_ids = job_encoding['input_ids'].to(self.device)
        job_attention_mask = job_encoding['attention_mask'].to(self.device)
        
        # Predict score
        with torch.no_grad():
            score = self.model(resume_input_ids, resume_attention_mask, job_input_ids, job_attention_mask)
        
        return float(score.item())
    
    def _calculate_rule_based_score(self, resume_data, job_data):
        """Calculate score using rule-based approach"""
        scores = {}
        
        # Skills matching (40% weight)
        scores['skills'] = self._calculate_skills_match(resume_data, job_data)
        
        # Experience matching (30% weight)
        scores['experience'] = self._calculate_experience_match(resume_data, job_data)
        
        # Education matching (20% weight)
        scores['education'] = self._calculate_education_match(resume_data, job_data)
        
        # Location matching (10% weight)
        scores['location'] = self._calculate_location_match(resume_data, job_data)
        
        # Calculate weighted average
        weights = {
            'skills': 0.4,
            'experience': 0.3,
            'education': 0.2,
            'location': 0.1
        }
        
        overall_score = sum(scores[key] * weights[key] for key in scores)
        
        return {
            'overall_score': overall_score,
            'skills_match': scores['skills'],
            'experience_match': scores['experience'],
            'education_match': scores['education'],
            'location_match': scores['location']
        }
    
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
            return 50.0  # Neutral score if no skills specified
        
        # Calculate exact matches
        exact_matches = resume_skills.intersection(job_skills)
        exact_score = (len(exact_matches) / len(job_skills)) * 100
        
        # Calculate semantic matches (simple keyword matching)
        semantic_matches = 0
        for job_skill in job_skills:
            if job_skill not in exact_matches:
                for resume_skill in resume_skills:
                    if self._are_skills_similar(job_skill, resume_skill):
                        semantic_matches += 1
                        break
        
        semantic_score = (semantic_matches / len(job_skills)) * 100
        
        # Combine scores (70% exact, 30% semantic)
        total_score = (exact_score * 0.7) + (semantic_score * 0.3)
        
        return min(total_score, 100.0)
    
    def _calculate_experience_match(self, resume_data, job_data):
        """Calculate experience matching percentage"""
        resume_exp = resume_data.get('experience_years', 0)
        required_exp = job_data.get('experience_required', 0)
        
        if required_exp == 0:
            return 100.0  # No experience required
        
        if resume_exp >= required_exp:
            # Bonus for extra experience, but cap at 100%
            bonus = min((resume_exp - required_exp) * 5, 20)
            return min(100.0 + bonus, 100.0)
        else:
            # Penalty for insufficient experience
            ratio = resume_exp / required_exp
            return ratio * 100
    
    def _calculate_education_match(self, resume_data, job_data):
        """Calculate education matching percentage"""
        resume_education = resume_data.get('education', [])
        job_description = job_data.get('description', '').lower()
        
        if not resume_education:
            return 30.0  # Low score for no education info
        
        # Check for degree requirements in job description
        degree_keywords = ['bachelor', 'master', 'phd', 'degree', 'diploma', 'certification']
        
        education_text = ' '.join(str(edu).lower() for edu in resume_education)
        
        # Basic matching
        score = 50.0  # Base score for having education
        
        for keyword in degree_keywords:
            if keyword in job_description and keyword in education_text:
                score += 10
        
        return min(score, 100.0)
    
    def _calculate_location_match(self, resume_data, job_data):
        """Calculate location matching percentage"""
        resume_location = resume_data.get('location', '').lower()
        job_location = job_data.get('location', '').lower()
        
        if not resume_location or not job_location:
            return 50.0  # Neutral score if location not specified
        
        # Check for exact city/state match
        if resume_location in job_location or job_location in resume_location:
            return 100.0
        
        # Check for state match
        resume_parts = resume_location.split(',')
        job_parts = job_location.split(',')
        
        for r_part in resume_parts:
            for j_part in job_parts:
                if r_part.strip() == j_part.strip():
                    return 80.0
        
        # Check for remote work
        if 'remote' in job_location:
            return 100.0
        
        return 20.0  # Different locations
    
    def _are_skills_similar(self, skill1, skill2):
        """Check if two skills are semantically similar"""
        # Simple similarity check
        skill_mappings = {
            'js': 'javascript',
            'py': 'python',
            'react.js': 'react',
            'node.js': 'nodejs',
            'vue.js': 'vue',
            'angular.js': 'angular',
            'c++': 'cpp',
            'c#': 'csharp'
        }
        
        # Normalize skills
        s1 = skill_mappings.get(skill1, skill1)
        s2 = skill_mappings.get(skill2, skill2)
        
        # Check if one is contained in the other
        return s1 in s2 or s2 in s1
    
    def _prepare_resume_text(self, resume_data):
        """Prepare resume text for ML model"""
        parts = []
        
        if resume_data.get('name'):
            parts.append(resume_data['name'])
        
        if resume_data.get('skills'):
            skills = resume_data['skills']
            if isinstance(skills, list):
                parts.append(' '.join(skills))
            else:
                parts.append(str(skills))
        
        if resume_data.get('education'):
            education = resume_data['education']
            if isinstance(education, list):
                parts.append(' '.join(str(edu) for edu in education))
            else:
                parts.append(str(education))
        
        if resume_data.get('experience_years'):
            parts.append(f"{resume_data['experience_years']} years experience")
        
        if resume_data.get('summary'):
            parts.append(resume_data['summary'])
        
        return ' '.join(parts)
    
    def _prepare_job_text(self, job_data):
        """Prepare job text for ML model"""
        parts = []
        
        if job_data.get('title'):
            parts.append(job_data['title'])
        
        if job_data.get('description'):
            parts.append(job_data['description'])
        
        if job_data.get('required_skills'):
            skills = job_data['required_skills']
            if isinstance(skills, list):
                parts.append(' '.join(skills))
            else:
                parts.append(str(skills))
        
        if job_data.get('experience_required'):
            parts.append(f"{job_data['experience_required']} years required")
        
        return ' '.join(parts)
    
    def get_matching_analysis(self, resume_data, job_data):
        """Get detailed matching analysis"""
        result = self.calculate_match_score(resume_data, job_data)
        
        if isinstance(result, dict):
            # Rule-based result with detailed breakdown
            analysis = {
                'overall_score': result['overall_score'],
                'breakdown': {
                    'skills_match': result['skills_match'],
                    'experience_match': result['experience_match'],
                    'education_match': result['education_match'],
                    'location_match': result['location_match']
                },
                'matching_skills': self._get_matching_skills(resume_data, job_data),
                'missing_skills': self._get_missing_skills(resume_data, job_data),
                'recommendation': self._get_recommendation(result['overall_score'])
            }
        else:
            # ML model result
            analysis = {
                'overall_score': result,
                'breakdown': {
                    'skills_match': self._calculate_skills_match(resume_data, job_data),
                    'experience_match': self._calculate_experience_match(resume_data, job_data),
                    'education_match': self._calculate_education_match(resume_data, job_data),
                    'location_match': self._calculate_location_match(resume_data, job_data)
                },
                'matching_skills': self._get_matching_skills(resume_data, job_data),
                'missing_skills': self._get_missing_skills(resume_data, job_data),
                'recommendation': self._get_recommendation(result)
            }
        
        return analysis
    
    def _get_matching_skills(self, resume_data, job_data):
        """Get list of matching skills"""
        resume_skills = set()
        job_skills = set()
        
        if isinstance(resume_data.get('skills'), list):
            resume_skills = set(skill.lower().strip() for skill in resume_data['skills'])
        elif isinstance(resume_data.get('skills'), str):
            resume_skills = set(skill.lower().strip() for skill in resume_data['skills'].split(','))
        
        if isinstance(job_data.get('required_skills'), list):
            job_skills = set(skill.lower().strip() for skill in job_data['required_skills'])
        elif isinstance(job_data.get('required_skills'), str):
            job_skills = set(skill.lower().strip() for skill in job_data['required_skills'].split(','))
        
        return list(resume_skills.intersection(job_skills))
    
    def _get_missing_skills(self, resume_data, job_data):
        """Get list of missing skills"""
        resume_skills = set()
        job_skills = set()
        
        if isinstance(resume_data.get('skills'), list):
            resume_skills = set(skill.lower().strip() for skill in resume_data['skills'])
        elif isinstance(resume_data.get('skills'), str):
            resume_skills = set(skill.lower().strip() for skill in resume_data['skills'].split(','))
        
        if isinstance(job_data.get('required_skills'), list):
            job_skills = set(skill.lower().strip() for skill in job_data['required_skills'])
        elif isinstance(job_data.get('required_skills'), str):
            job_skills = set(skill.lower().strip() for skill in job_data['required_skills'].split(','))
        
        return list(job_skills - resume_skills)
    
    def _get_recommendation(self, score):
        """Get recommendation based on score"""
        if score >= 80:
            return "Highly Recommended - Shortlist"
        elif score >= 60:
            return "Good Fit - Consider Interview"
        elif score >= 40:
            return "Moderate Fit - Review Carefully"
        else:
            return "Below Requirements - Consider Rejecting"
