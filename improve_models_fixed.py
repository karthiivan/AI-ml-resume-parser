#!/usr/bin/env python3
"""
Improved Skills Extractor and NER Model - Fixed Version
Enhanced algorithms with better error handling
"""

import pandas as pd
import json
import numpy as np
import pickle
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class ImprovedSkillsExtractor:
    def __init__(self):
        self.skill_models = {}
        self.skill_patterns = {}
        
    def create_enhanced_training_data(self):
        """Create enhanced training data with better features"""
        print("🔨 Creating enhanced skills training data...")
        
        # Load datasets
        resume_df = pd.read_csv('data/resume_dataset.csv')
        
        # Enhanced skill definitions with synonyms and variations
        enhanced_skills = {
            'Python': ['python', 'py', 'django', 'flask', 'fastapi', 'pandas', 'numpy', 'scipy'],
            'JavaScript': ['javascript', 'js', 'node.js', 'nodejs', 'react', 'angular', 'vue', 'jquery'],
            'Java': ['java', 'spring', 'spring boot', 'hibernate', 'maven', 'gradle'],
            'React': ['react', 'reactjs', 'react.js', 'jsx', 'redux', 'react native'],
            'Node.js': ['node.js', 'nodejs', 'node', 'express', 'express.js'],
            'AWS': ['aws', 'amazon web services', 'ec2', 's3', 'lambda', 'cloudformation'],
            'Docker': ['docker', 'containerization', 'dockerfile', 'docker-compose'],
            'MySQL': ['mysql', 'sql', 'database', 'relational database', 'mariadb'],
            'PostgreSQL': ['postgresql', 'postgres', 'psql'],
            'MongoDB': ['mongodb', 'mongo', 'nosql', 'document database'],
            'Git': ['git', 'github', 'gitlab', 'version control', 'bitbucket'],
            'Linux': ['linux', 'unix', 'bash', 'shell', 'ubuntu', 'centos'],
            'TensorFlow': ['tensorflow', 'tf', 'keras', 'machine learning', 'deep learning'],
            'Angular': ['angular', 'angularjs', 'typescript'],
            'Spring Boot': ['spring boot', 'spring', 'java framework']
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
            
            # Check for balanced data
            positive_count = sum(labels)
            negative_count = len(labels) - positive_count
            
            if positive_count >= 10 and negative_count >= 10:  # Need both classes
                all_training_data[skill] = {
                    'texts': texts,
                    'labels': labels,
                    'keywords': keywords,
                    'positive_samples': positive_count,
                    'negative_samples': negative_count
                }
                print(f"     ✅ {skill}: {positive_count} positive, {negative_count} negative samples")
            else:
                print(f"     ❌ {skill}: {positive_count} positive, {negative_count} negative (skipped - imbalanced)")
        
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
            
            # Enhanced TF-IDF with n-grams
            vectorizer = TfidfVectorizer(
                max_features=2000, 
                stop_words='english',
                ngram_range=(1, 3),  # Include trigrams
                min_df=2,
                max_df=0.95
            )
            
            # Transform features
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            
            # Try multiple models and select best
            models = {
                'RandomForest': RandomForestClassifier(
                    n_estimators=200, 
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                ),
                'GradientBoosting': GradientBoostingClassifier(
                    n_estimators=100, 
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'LogisticRegression': LogisticRegression(
                    random_state=42, 
                    max_iter=1000,
                    C=1.0
                )
            }
            
            best_model = None
            best_score = 0
            best_name = ""
            
            for name, model in models.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=3, scoring='f1')
                    avg_score = np.mean(cv_scores)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                        best_name = name
                except Exception as e:
                    print(f"     ⚠️ {name} failed: {e}")
                    continue
            
            if best_model is None:
                print(f"     ❌ No model worked for {skill}")
                continue
            
            # Train best model
            best_model.fit(X_train_vec, y_train)
            y_pred = best_model.predict(X_test_vec)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Store model and vectorizer
            self.skill_models[skill] = {
                'model': best_model,
                'vectorizer': vectorizer,
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
    
    def create_comprehensive_patterns(self):
        """Create comprehensive rule-based patterns"""
        print("\n📋 Creating comprehensive skill patterns...")
        
        # Comprehensive skill patterns
        skill_patterns = {
            # Programming Languages
            'HTML': [r'\bhtml\b', r'\bhtml5\b', r'\bmarkup\b'],
            'CSS': [r'\bcss\b', r'\bcss3\b', r'\bsass\b', r'\bless\b', r'\bstylesheets\b'],
            'TypeScript': [r'\btypescript\b', r'\bts\b'],
            'PHP': [r'\bphp\b', r'\blaravel\b', r'\bsymfony\b', r'\bcodeigniter\b'],
            'Ruby': [r'\bruby\b', r'\brails\b', r'\bruby on rails\b'],
            'Go': [r'\bgolang\b', r'\bgo\b(?!\s+to)'],  # Avoid "go to"
            'Rust': [r'\brust\b'],
            'C++': [r'\bc\+\+\b', r'\bcpp\b'],
            'C#': [r'\bc#\b', r'\bcsharp\b', r'\b\.net\b'],
            'Swift': [r'\bswift\b', r'\bios development\b'],
            'Kotlin': [r'\bkotlin\b', r'\bandroid development\b'],
            'Scala': [r'\bscala\b'],
            'R': [r'\br programming\b', r'\brstudio\b'],
            'MATLAB': [r'\bmatlab\b'],
            'Perl': [r'\bperl\b'],
            'Shell': [r'\bshell scripting\b', r'\bbash\b', r'\bzsh\b'],
            
            # Frameworks & Libraries
            'Vue.js': [r'\bvue\b', r'\bvue\.js\b', r'\bvuejs\b'],
            'Express.js': [r'\bexpress\b', r'\bexpress\.js\b'],
            'Django': [r'\bdjango\b'],
            'Flask': [r'\bflask\b'],
            'Laravel': [r'\blaravel\b'],
            'Symfony': [r'\bsymfony\b'],
            'Rails': [r'\brails\b', r'\bruby on rails\b'],
            'ASP.NET': [r'\basp\.net\b', r'\basp net\b'],
            'jQuery': [r'\bjquery\b'],
            'Bootstrap': [r'\bbootstrap\b'],
            
            # Databases
            'SQLite': [r'\bsqlite\b'],
            'Oracle': [r'\boracle\b', r'\boracle database\b'],
            'SQL Server': [r'\bsql server\b', r'\bmssql\b'],
            'Cassandra': [r'\bcassandra\b'],
            'DynamoDB': [r'\bdynamodb\b'],
            'Neo4j': [r'\bneo4j\b', r'\bgraph database\b'],
            
            # Cloud & DevOps
            'Azure': [r'\bazure\b', r'\bmicrosoft azure\b'],
            'Google Cloud': [r'\bgoogle cloud\b', r'\bgcp\b'],
            'Kubernetes': [r'\bkubernetes\b', r'\bk8s\b'],
            'Jenkins': [r'\bjenkins\b', r'\bci/cd\b'],
            'Terraform': [r'\bterraform\b', r'\binfrastructure as code\b'],
            'Ansible': [r'\bansible\b', r'\bautomation\b'],
            'Chef': [r'\bchef\b'],
            'Puppet': [r'\bpuppet\b'],
            
            # Data & Analytics
            'Pandas': [r'\bpandas\b'],
            'NumPy': [r'\bnumpy\b'],
            'Scikit-learn': [r'\bscikit-learn\b', r'\bsklearn\b'],
            'PyTorch': [r'\bpytorch\b', r'\btorch\b'],
            'Keras': [r'\bkeras\b'],
            'Apache Spark': [r'\bspark\b', r'\bapache spark\b'],
            'Hadoop': [r'\bhadoop\b', r'\bhdfs\b'],
            'Kafka': [r'\bkafka\b', r'\bmessaging\b'],
            'Elasticsearch': [r'\belasticsearch\b', r'\belastic\b'],
            'Redis': [r'\bredis\b', r'\bcaching\b'],
            'Tableau': [r'\btableau\b'],
            'Power BI': [r'\bpower bi\b', r'\bpowerbi\b'],
            
            # Mobile Development
            'React Native': [r'\breact native\b'],
            'Flutter': [r'\bflutter\b'],
            'Xamarin': [r'\bxamarin\b'],
            'Ionic': [r'\bionic\b'],
            
            # Testing
            'Jest': [r'\bjest\b'],
            'Mocha': [r'\bmocha\b'],
            'Selenium': [r'\bselenium\b'],
            'Cypress': [r'\bcypress\b'],
            'JUnit': [r'\bjunit\b'],
            'PyTest': [r'\bpytest\b'],
            
            # Methodologies
            'Agile': [r'\bagile\b', r'\bscrum\b', r'\bkanban\b'],
            'DevOps': [r'\bdevops\b', r'\bsite reliability\b'],
            'Microservices': [r'\bmicroservices\b', r'\bservice oriented\b'],
            'REST API': [r'\brest\b', r'\brestful\b', r'\bapi\b'],
            'GraphQL': [r'\bgraphql\b'],
            'OAuth': [r'\boauth\b'],
            'JWT': [r'\bjwt\b', r'\bjson web token\b']
        }
        
        self.skill_patterns = skill_patterns
        print(f"   ✅ Created patterns for {len(skill_patterns)} skills")
        
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
                vectorizer = model_data['vectorizer']
                
                # Transform text
                text_vec = vectorizer.transform([text_lower])
                
                # Predict
                prediction = model.predict(text_vec)[0]
                confidence = max(model.predict_proba(text_vec)[0])
                
                if prediction == 1 and confidence > 0.7:  # Higher threshold
                    detected_skills.append(skill)
                    confidence_scores[skill] = confidence
                    
            except Exception as e:
                print(f"Error processing {skill}: {e}")
        
        # Rule-based extraction
        for skill, patterns in self.skill_patterns.items():
            if skill not in detected_skills:
                for pattern in patterns:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        detected_skills.append(skill)
                        confidence_scores[skill] = 0.85  # Rule-based confidence
                        break
        
        return detected_skills, confidence_scores
    
    def save_improved_model(self):
        """Save the improved skills extraction model"""
        model_data = {
            'ml_models': self.skill_models,
            'rule_patterns': self.skill_patterns,
            'version': '2.0_improved',
            'total_skills': len(self.skill_models) + len(self.skill_patterns)
        }
        
        with open('ml_models/trained/skills_extractor_improved.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Improved skills extractor saved")

class ImprovedNERModel:
    def __init__(self):
        self.patterns = {}
        
    def create_enhanced_ner_patterns(self):
        """Create enhanced NER patterns with better regex"""
        print("\n🏷️ Creating enhanced NER patterns...")
        
        enhanced_patterns = {
            'PERSON': [
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?:\n|$)',  # First line name
                r'(?:Name|Full Name):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b(?=\s*(?:\n|Software|Engineer|Developer|Manager|Analyst))'
            ],
            'EMAIL': [
                r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',
                r'(?:Email|E-mail|Mail):\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
                r'Contact:\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})'
            ],
            'PHONE': [
                r'(\+?1[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})',
                r'(?:Phone|Tel|Mobile|Cell):\s*(\+?1[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})',
                r'\b(\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b'
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
                r'(?:at|@)\s+([A-Z][A-Za-z\s&.,]+(?:\s+(?:Inc|LLC|Corp|Ltd)\.?)?)',
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
    
    def extract_entities(self, text):
        """Extract entities using enhanced patterns"""
        entities = {}
        
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
    
    def save_improved_model(self):
        """Save the improved NER model"""
        model_data = {
            'patterns': self.patterns,
            'version': '2.0_improved',
            'entity_types': len(self.patterns)
        }
        
        with open('ml_models/trained/ner_model_improved.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Improved NER model saved")

def test_improved_models():
    """Test the improved models"""
    print("\n🧪 TESTING IMPROVED MODELS")
    print("-" * 40)
    
    # Test Skills Extractor
    print("🛠️ Testing Improved Skills Extractor...")
    
    with open('ml_models/trained/skills_extractor_improved.pkl', 'rb') as f:
        skills_data = pickle.load(f)
    
    test_text = "Senior Python developer with 5 years experience in Django, React, AWS, Docker, and PostgreSQL. Proficient in JavaScript, Node.js, and Git version control."
    
    # Simple rule-based test
    detected_skills = []
    for skill, patterns in skills_data['rule_patterns'].items():
        for pattern in patterns:
            if re.search(pattern, test_text.lower(), re.IGNORECASE):
                detected_skills.append(skill)
                break
    
    print(f"   Test Text: {test_text[:60]}...")
    print(f"   Detected Skills: {detected_skills[:10]}")  # Show first 10
    print(f"   Total Skills Detected: {len(detected_skills)}")
    
    # Test NER Model
    print("\n🏷️ Testing Improved NER Model...")
    
    with open('ml_models/trained/ner_model_improved.pkl', 'rb') as f:
        ner_data = pickle.load(f)
    
    test_text_ner = """
    John Smith
    Senior Software Engineer
    Email: john.smith@techcorp.com
    Phone: +1-555-123-4567
    Location: San Francisco, CA
    Education: MS Computer Science, Stanford University
    Experience: 8 years of software development
    Skills: Python, JavaScript, AWS, Docker
    """
    
    entities = {}
    for entity_type, patterns in ner_data['patterns'].items():
        found = []
        for pattern in patterns:
            matches = re.finditer(pattern, test_text_ner, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if match.groups():
                    text = match.group(1).strip()
                else:
                    text = match.group(0).strip()
                if text and len(text) > 1:
                    found.append(text)
        if found:
            entities[entity_type] = found[:3]  # Top 3 matches
    
    print(f"   Test Text: NER test resume")
    print(f"   Detected Entities:")
    for entity_type, values in entities.items():
        print(f"     {entity_type}: {values}")

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
    
    # Create comprehensive patterns
    skills_extractor.create_comprehensive_patterns()
    
    # Save improved model
    skills_extractor.save_improved_model()
    
    # Improve NER Model
    print("\n🏷️ PHASE 2: IMPROVING NER MODEL")
    print("-" * 40)
    
    ner_model = ImprovedNERModel()
    
    # Create enhanced patterns
    ner_model.create_enhanced_ner_patterns()
    
    # Save improved model
    ner_model.save_improved_model()
    
    # Test improved models
    test_improved_models()
    
    # Performance Summary
    print("\n📊 IMPROVEMENT SUMMARY")
    print("-" * 40)
    
    print("✅ Skills Extractor Improvements:")
    print(f"   • Enhanced ML models for {len(skill_metrics)} skills")
    print(f"   • Comprehensive patterns for {len(skills_extractor.skill_patterns)} additional skills")
    print(f"   • Total skills coverage: {len(skill_metrics) + len(skills_extractor.skill_patterns)}")
    print(f"   • Advanced feature extraction (n-grams, better vectorization)")
    print(f"   • Multiple model comparison with cross-validation")
    
    if skill_metrics:
        avg_f1 = np.mean([m['f1_score'] for m in skill_metrics.values()])
        print(f"   • Average F1 Score: {avg_f1:.3f}")
    
    print("\n✅ NER Model Improvements:")
    print(f"   • Enhanced regex patterns for {len(ner_model.patterns)} entity types")
    print(f"   • Better pattern matching with multiple variations")
    print(f"   • Improved entity extraction accuracy")
    print(f"   • Comprehensive coverage: PERSON, EMAIL, PHONE, LOCATION, EDUCATION, etc.")
    
    print(f"\n🎉 IMPROVEMENT COMPLETE!")
    print(f"📁 Saved improved models:")
    print(f"   • ml_models/trained/skills_extractor_improved.pkl")
    print(f"   • ml_models/trained/ner_model_improved.pkl")

if __name__ == "__main__":
    main()
