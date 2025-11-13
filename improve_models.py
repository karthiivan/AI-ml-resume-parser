#!/usr/bin/env python3
"""
Improved Skills Extractor and NER Model
Enhanced algorithms for better performance
"""

import pandas as pd
import json
import numpy as np
import pickle
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ImprovedSkillsExtractor:
    def __init__(self):
        self.skill_models = {}
        self.skill_patterns = {}
        self.skill_keywords = {}
        self.vectorizers = {}
        
    def create_enhanced_training_data(self):
        """Create enhanced training data with better features"""
        print("🔨 Creating enhanced skills training data...")
        
        # Load datasets
        resume_df = pd.read_csv('data/resume_dataset.csv')
        skills_df = pd.read_csv('data/skills.csv')
        
        # Enhanced skill definitions with synonyms and variations
        enhanced_skills = {
            'Python': ['python', 'py', 'django', 'flask', 'fastapi', 'pandas', 'numpy', 'scipy'],
            'JavaScript': ['javascript', 'js', 'node.js', 'nodejs', 'react', 'angular', 'vue', 'jquery'],
            'Java': ['java', 'spring', 'spring boot', 'hibernate', 'maven', 'gradle'],
            'React': ['react', 'reactjs', 'react.js', 'jsx', 'redux', 'react native'],
            'Node.js': ['node.js', 'nodejs', 'node', 'express', 'express.js'],
            'AWS': ['aws', 'amazon web services', 'ec2', 's3', 'lambda', 'cloudformation'],
            'Docker': ['docker', 'containerization', 'dockerfile', 'docker-compose'],
            'Kubernetes': ['kubernetes', 'k8s', 'kubectl', 'helm', 'container orchestration'],
            'MySQL': ['mysql', 'sql', 'database', 'relational database', 'mariadb'],
            'PostgreSQL': ['postgresql', 'postgres', 'psql', 'sql'],
            'MongoDB': ['mongodb', 'mongo', 'nosql', 'document database'],
            'Git': ['git', 'github', 'gitlab', 'version control', 'bitbucket'],
            'Linux': ['linux', 'unix', 'bash', 'shell', 'ubuntu', 'centos'],
            'TensorFlow': ['tensorflow', 'tf', 'keras', 'machine learning', 'deep learning'],
            'PyTorch': ['pytorch', 'torch', 'deep learning', 'neural networks']
        }
        
        # Create training data for each skill
        all_training_data = {}
        
        for skill, keywords in enhanced_skills.items():
            print(f"   Processing {skill}...")
            
            texts = []
            labels = []
            
            for _, row in resume_df.iterrows():
                resume_text = str(row['resume_text']).lower()
                skills_text = str(row['skills']).lower()
                category = str(row['category']).lower()
                
                # Combine all text
                full_text = f"{resume_text} {skills_text} {category}"
                
                # Check if any keyword is present
                has_skill = any(keyword.lower() in full_text for keyword in keywords)
                
                texts.append(full_text)
                labels.append(1 if has_skill else 0)
            
            # Only include skills with sufficient positive examples
            positive_count = sum(labels)
            if positive_count >= 10:  # Minimum 10 positive examples
                all_training_data[skill] = {
                    'texts': texts,
                    'labels': labels,
                    'keywords': keywords,
                    'positive_samples': positive_count
                }
                print(f"     ✅ {skill}: {positive_count} positive samples")
            else:
                print(f"     ❌ {skill}: Only {positive_count} samples (skipped)")
        
        return all_training_data
    
    def train_improved_models(self, training_data):
        """Train improved models with better algorithms"""
        print("\n🚀 Training improved skills extraction models...")
        
        skill_metrics = {}
        
        for skill, data in training_data.items():
            print(f"   Training enhanced model for: {skill}")
            
            texts = data['texts']
            labels = data['labels']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Create enhanced feature pipeline
            # Combine TF-IDF with character n-grams for better feature extraction
            tfidf_word = TfidfVectorizer(
                max_features=1000, 
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams
                min_df=2
            )
            
            tfidf_char = TfidfVectorizer(
                max_features=500,
                analyzer='char',
                ngram_range=(2, 4),  # Character n-grams
                min_df=2
            )
            
            # Transform features
            X_train_word = tfidf_word.fit_transform(X_train)
            X_train_char = tfidf_char.fit_transform(X_train)
            
            X_test_word = tfidf_word.transform(X_test)
            X_test_char = tfidf_char.transform(X_test)
            
            # Combine features
            from scipy.sparse import hstack
            X_train_combined = hstack([X_train_word, X_train_char])
            X_test_combined = hstack([X_test_word, X_test_char])
            
            # Try multiple models and select best
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
            }
            
            best_model = None
            best_score = 0
            best_name = ""
            
            for name, model in models.items():
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_combined, y_train, cv=3, scoring='f1')
                avg_score = np.mean(cv_scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_name = name
            
            # Train best model
            best_model.fit(X_train_combined, y_train)
            y_pred = best_model.predict(X_test_combined)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Store model and vectorizers
            self.skill_models[skill] = {
                'model': best_model,
                'word_vectorizer': tfidf_word,
                'char_vectorizer': tfidf_char,
                'model_name': best_name,
                'keywords': data['keywords']
            }
            
            skill_metrics[skill] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_score': best_score,
                'model_used': best_name,
                'positive_samples': data['positive_samples']
            }
            
            print(f"     ✅ {best_name}: F1={f1:.3f}, CV={best_score:.3f}")
        
        return skill_metrics
    
    def create_rule_based_backup(self):
        """Create rule-based backup for skills not in ML models"""
        print("\n📋 Creating rule-based backup patterns...")
        
        # Extended skill patterns with variations
        skill_patterns = {
            'HTML': [r'\bhtml\b', r'\bhtml5\b', r'\bmarkup\b'],
            'CSS': [r'\bcss\b', r'\bcss3\b', r'\bsass\b', r'\bless\b', r'\bstylesheets\b'],
            'TypeScript': [r'\btypescript\b', r'\bts\b'],
            'PHP': [r'\bphp\b', r'\blaravel\b', r'\bsymfony\b'],
            'Ruby': [r'\bruby\b', r'\brails\b', r'\bruby on rails\b'],
            'Go': [r'\bgolang\b', r'\bgo\b'],
            'Rust': [r'\brust\b'],
            'C++': [r'\bc\+\+\b', r'\bcpp\b'],
            'C#': [r'\bc#\b', r'\bcsharp\b', r'\b\.net\b'],
            'Swift': [r'\bswift\b', r'\bios\b'],
            'Kotlin': [r'\bkotlin\b', r'\bandroid\b'],
            'Scala': [r'\bscala\b'],
            'R': [r'\br\b', r'\brstudio\b'],
            'MATLAB': [r'\bmatlab\b'],
            'Jenkins': [r'\bjenkins\b', r'\bci/cd\b', r'\bcontinuous integration\b'],
            'Terraform': [r'\bterraform\b', r'\binfrastructure as code\b'],
            'Ansible': [r'\bansible\b', r'\bautomation\b'],
            'Redis': [r'\bredis\b', r'\bcaching\b'],
            'Elasticsearch': [r'\belasticsearch\b', r'\belastic\b', r'\bsearch engine\b'],
            'Apache Spark': [r'\bspark\b', r'\bapache spark\b', r'\bbig data\b'],
            'Hadoop': [r'\bhadoop\b', r'\bhdfs\b'],
            'Kafka': [r'\bkafka\b', r'\bmessaging\b'],
            'GraphQL': [r'\bgraphql\b', r'\bapi\b'],
            'REST': [r'\brest\b', r'\brestful\b', r'\bapi\b'],
            'Microservices': [r'\bmicroservices\b', r'\bservice oriented\b'],
            'Agile': [r'\bagile\b', r'\bscrum\b', r'\bkanban\b'],
            'DevOps': [r'\bdevops\b', r'\bsite reliability\b']
        }
        
        self.skill_patterns = skill_patterns
        print(f"   ✅ Created patterns for {len(skill_patterns)} additional skills")
        
        return skill_patterns
    
    def extract_skills_from_text(self, text):
        """Extract skills from text using both ML and rule-based approaches"""
        detected_skills = []
        confidence_scores = {}
        
        text_lower = text.lower()
        
        # ML-based extraction
        for skill, model_data in self.skill_models.items():
            try:
                model = model_data['model']
                word_vec = model_data['word_vectorizer']
                char_vec = model_data['char_vectorizer']
                
                # Transform text
                text_word = word_vec.transform([text_lower])
                text_char = char_vec.transform([text_lower])
                
                # Combine features
                from scipy.sparse import hstack
                text_combined = hstack([text_word, text_char])
                
                # Predict
                prediction = model.predict(text_combined)[0]
                confidence = max(model.predict_proba(text_combined)[0])
                
                if prediction == 1 and confidence > 0.6:  # Higher threshold
                    detected_skills.append(skill)
                    confidence_scores[skill] = confidence
                    
            except Exception as e:
                print(f"Error processing {skill}: {e}")
        
        # Rule-based backup
        for skill, patterns in self.skill_patterns.items():
            if skill not in detected_skills:
                for pattern in patterns:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        detected_skills.append(skill)
                        confidence_scores[skill] = 0.8  # Rule-based confidence
                        break
        
        return detected_skills, confidence_scores
    
    def save_improved_model(self):
        """Save the improved skills extraction model"""
        model_data = {
            'ml_models': self.skill_models,
            'rule_patterns': self.skill_patterns,
            'version': '2.0_improved'
        }
        
        with open('ml_models/trained/skills_extractor_improved.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Improved skills extractor saved")

class ImprovedNERModel:
    def __init__(self):
        self.patterns = {}
        self.ml_models = {}
        
    def create_enhanced_ner_patterns(self):
        """Create enhanced NER patterns with better regex"""
        print("\n🏷️ Creating enhanced NER patterns...")
        
        enhanced_patterns = {
            'PERSON': [
                r'^([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\n|$)',  # First line name
                r'(?:Name|Full Name):\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
                r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b(?=\s*(?:Software|Engineer|Developer|Manager|Analyst))'
            ],
            'EMAIL': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                r'(?:Email|E-mail|Mail):\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
                r'Contact:\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})'
            ],
            'PHONE': [
                r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
                r'(?:Phone|Tel|Mobile|Cell):\s*((?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})',
                r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
            ],
            'LOCATION': [
                r'\b([A-Z][a-z]+,\s*[A-Z]{2})\b',  # City, ST
                r'\b([A-Z][a-z]+\s+[A-Z][a-z]+,\s*[A-Z]{2})\b',  # City Name, ST
                r'(?:Location|Address|Based in):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*[A-Z]{2})',
                r'\b([A-Z][a-z]+,\s*[A-Z][a-z]+)\b'  # City, Country
            ],
            'EDUCATION': [
                r'\b((?:Bachelor|Master|PhD|Doctorate|MBA|BS|MS|BA|MA|B\.Tech|M\.Tech)(?:\s+of\s+\w+)*(?:\s+in\s+[\w\s]+)?)\b',
                r'(?:Education|Degree):\s*((?:Bachelor|Master|PhD|Doctorate|MBA|BS|MS|BA|MA)(?:\s+of\s+\w+)*(?:\s+in\s+[\w\s]+)?)',
                r'\b((?:Bachelor|Master)\'?s?\s+(?:of\s+)?(?:Science|Arts|Engineering)(?:\s+in\s+[\w\s]+)?)\b'
            ],
            'EXPERIENCE_YEARS': [
                r'\b(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)\b',
                r'(?:Experience|Exp):\s*(\d+)\+?\s*years?',
                r'\b(\d+)\+?\s*years?\s*(?:in|with|of)\s+\w+',
                r'(?:Over|More than)\s+(\d+)\s+years?'
            ],
            'SKILLS': [
                r'(?:Skills|Technical Skills|Programming Languages):\s*([^.\n]+)',
                r'(?:Technologies|Tech Stack):\s*([^.\n]+)',
                r'(?:Proficient in|Expert in|Experienced with):\s*([^.\n]+)'
            ],
            'COMPANY': [
                r'(?:Company|Organization|Employer):\s*([A-Z][A-Za-z\s&.,]+)',
                r'(?:at|@)\s+([A-Z][A-Za-z\s&.,]+)(?:\s+(?:Inc|LLC|Corp|Ltd)\.?)?',
                r'\b([A-Z][A-Za-z\s&.,]+(?:\s+(?:Inc|LLC|Corp|Ltd)\.?))\b'
            ],
            'CERTIFICATION': [
                r'\b((?:AWS|Azure|Google|Oracle|Microsoft|Cisco)\s+Certified\s+[\w\s]+)\b',
                r'(?:Certification|Certified in):\s*([\w\s]+)',
                r'\b(PMP|CISSP|CISA|CPA|CFA|FRM)\b'
            ]
        }
        
        self.patterns = enhanced_patterns
        print(f"   ✅ Created enhanced patterns for {len(enhanced_patterns)} entity types")
        
        return enhanced_patterns
    
    def train_ml_ner_models(self):
        """Train ML models for NER as backup to patterns"""
        print("\n🤖 Training ML models for NER...")
        
        # Load NER training data
        with open('data/resume_entities_ner.json', 'r', encoding='utf-8') as f:
            ner_data = json.load(f)
        
        # Create training data for each entity type
        entity_models = {}
        
        for entity_type in ['PERSON', 'EMAIL', 'PHONE', 'LOCATION', 'SKILLS', 'EDUCATION']:
            print(f"   Training ML model for: {entity_type}")
            
            texts = []
            labels = []
            
            for item in ner_data:
                text = item['content']
                has_entity = False
                
                for annotation in item['annotation']:
                    if annotation['label'][0] == entity_type:
                        has_entity = True
                        break
                
                texts.append(text)
                labels.append(1 if has_entity else 0)
            
            # Check if we have enough positive samples
            positive_count = sum(labels)
            if positive_count < 5:
                print(f"     ❌ {entity_type}: Only {positive_count} samples (skipped)")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            
            # Create pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=500, stop_words='english')),
                ('classifier', LogisticRegression(random_state=42))
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            entity_models[entity_type] = {
                'model': pipeline,
                'accuracy': accuracy,
                'f1_score': f1,
                'positive_samples': positive_count
            }
            
            print(f"     ✅ {entity_type}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
        
        self.ml_models = entity_models
        return entity_models
    
    def extract_entities(self, text):
        """Extract entities using enhanced patterns and ML backup"""
        entities = {}
        
        # Pattern-based extraction (primary)
        for entity_type, patterns in self.patterns.items():
            found_entities = []
            
            for pattern in patterns:
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
                            'method': 'pattern'
                        })
            
            if found_entities:
                entities[entity_type] = found_entities
        
        # ML-based backup for missing entities
        for entity_type, model_data in self.ml_models.items():
            if entity_type not in entities:
                try:
                    model = model_data['model']
                    prediction = model.predict([text])[0]
                    confidence = max(model.predict_proba([text])[0])
                    
                    if prediction == 1 and confidence > 0.7:
                        entities[entity_type] = [{
                            'text': f'{entity_type}_detected',
                            'start': 0,
                            'end': len(text),
                            'method': 'ml',
                            'confidence': confidence
                        }]
                except Exception as e:
                    print(f"Error in ML extraction for {entity_type}: {e}")
        
        return entities
    
    def save_improved_model(self):
        """Save the improved NER model"""
        model_data = {
            'patterns': self.patterns,
            'ml_models': self.ml_models,
            'version': '2.0_improved'
        }
        
        with open('ml_models/trained/ner_model_improved.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Improved NER model saved")

def main():
    """Main improvement function"""
    print("🚀 IMPROVING SKILLS EXTRACTOR AND NER MODEL")
    print("=" * 60)
    
    # Improve Skills Extractor
    print("\n📊 PHASE 1: IMPROVING SKILLS EXTRACTOR")
    print("-" * 40)
    
    skills_extractor = ImprovedSkillsExtractor()
    
    # Create enhanced training data
    training_data = skills_extractor.create_enhanced_training_data()
    
    # Train improved models
    skill_metrics = skills_extractor.train_improved_models(training_data)
    
    # Create rule-based backup
    skills_extractor.create_rule_based_backup()
    
    # Save improved model
    skills_extractor.save_improved_model()
    
    # Improve NER Model
    print("\n🏷️ PHASE 2: IMPROVING NER MODEL")
    print("-" * 40)
    
    ner_model = ImprovedNERModel()
    
    # Create enhanced patterns
    ner_model.create_enhanced_ner_patterns()
    
    # Train ML backup models
    ner_model.train_ml_ner_models()
    
    # Save improved model
    ner_model.save_improved_model()
    
    # Performance Summary
    print("\n📊 IMPROVEMENT SUMMARY")
    print("-" * 40)
    
    print("✅ Skills Extractor Improvements:")
    print(f"   • Enhanced feature extraction (word + character n-grams)")
    print(f"   • Multiple model comparison (RF, GB, LR)")
    print(f"   • Cross-validation for model selection")
    print(f"   • Rule-based backup for {len(skills_extractor.skill_patterns)} additional skills")
    print(f"   • Improved models for {len(skill_metrics)} skills")
    
    avg_f1 = np.mean([m['f1_score'] for m in skill_metrics.values()])
    print(f"   • Average F1 Score: {avg_f1:.3f}")
    
    print("\n✅ NER Model Improvements:")
    print(f"   • Enhanced regex patterns for {len(ner_model.patterns)} entity types")
    print(f"   • ML backup models for {len(ner_model.ml_models)} entities")
    print(f"   • Multi-method extraction (pattern + ML)")
    print(f"   • Better entity coverage and accuracy")
    
    print(f"\n🎉 IMPROVEMENT COMPLETE!")
    print(f"📁 Saved improved models:")
    print(f"   • ml_models/trained/skills_extractor_improved.pkl")
    print(f"   • ml_models/trained/ner_model_improved.pkl")

if __name__ == "__main__":
    main()
