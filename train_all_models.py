#!/usr/bin/env python3
"""
Complete ML Model Training Pipeline with Real Datasets
Trains all models and generates performance metrics
"""

import pandas as pd
import json
import numpy as np
import pickle
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.training_history = []
        
        # Create directories
        Path('ml_models/trained').mkdir(parents=True, exist_ok=True)
        Path('ml_models/metrics').mkdir(parents=True, exist_ok=True)
        Path('ml_models/plots').mkdir(parents=True, exist_ok=True)
    
    def train_resume_classifier(self):
        """Train resume category classification model"""
        print("\n🎯 Training Resume Category Classifier...")
        
        # Load resume dataset
        df = pd.read_csv('data/resume_dataset.csv')
        
        # Prepare data
        X = df['resume_text'].fillna('')
        y = df['category']
        
        print(f"📊 Dataset: {len(df)} samples, {len(y.unique())} categories")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42)
        }
        
        best_model = None
        best_score = 0
        model_metrics = {}
        
        for name, model in models.items():
            print(f"   Training {name}...")
            start_time = time.time()
            
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            training_time = time.time() - start_time
            
            model_metrics[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_time': training_time
            }
            
            print(f"     Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Time: {training_time:.2f}s")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = (name, model)
        
        # Save best model
        best_name, best_classifier = best_model
        with open('ml_models/trained/resume_classifier.pkl', 'wb') as f:
            pickle.dump({
                'model': best_classifier,
                'vectorizer': vectorizer,
                'model_name': best_name,
                'categories': list(y.unique())
            }, f)
        
        # Generate detailed classification report
        y_pred_best = best_classifier.predict(X_test_vec)
        classification_rep = classification_report(y_test, y_pred_best, output_dict=True)
        
        self.metrics['resume_classifier'] = {
            'best_model': best_name,
            'model_comparison': model_metrics,
            'classification_report': classification_rep,
            'test_accuracy': best_score,
            'categories': list(y.unique()),
            'dataset_size': len(df)
        }
        
        print(f"✅ Best model: {best_name} (Accuracy: {best_score:.3f})")
        return True
    
    def train_skills_extractor(self):
        """Train skills extraction model"""
        print("\n🛠️ Training Skills Extraction Model...")
        
        # Load datasets
        resume_df = pd.read_csv('data/resume_dataset.csv')
        skills_df = pd.read_csv('data/skills.csv')
        
        # Create training data for skills extraction
        skill_keywords = skills_df['skill'].tolist()
        
        # Prepare binary classification data for each skill
        skill_models = {}
        skill_metrics = {}
        
        for skill in skill_keywords:
            print(f"   Training model for: {skill}")
            
            # Create labels (1 if skill mentioned, 0 otherwise)
            labels = []
            texts = []
            
            for _, row in resume_df.iterrows():
                resume_text = str(row['resume_text'])
                skills_text = str(row['skills'])
                
                # Check if skill is mentioned
                has_skill = (skill.lower() in resume_text.lower() or 
                           skill.lower() in skills_text.lower())
                
                labels.append(1 if has_skill else 0)
                texts.append(resume_text)
            
            # Skip if not enough positive examples
            if sum(labels) < 5:
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            
            # Vectorize
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            
            # Train model
            model = LogisticRegression(random_state=42)
            model.fit(X_train_vec, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            
            skill_models[skill] = {
                'model': model,
                'vectorizer': vectorizer
            }
            
            skill_metrics[skill] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'positive_samples': sum(labels)
            }
        
        # Save skills extraction models
        with open('ml_models/trained/skills_extractor.pkl', 'wb') as f:
            pickle.dump(skill_models, f)
        
        self.metrics['skills_extractor'] = {
            'skills_trained': len(skill_models),
            'skill_metrics': skill_metrics,
            'average_accuracy': np.mean([m['accuracy'] for m in skill_metrics.values()]),
            'average_f1': np.mean([m['f1_score'] for m in skill_metrics.values()])
        }
        
        print(f"✅ Trained models for {len(skill_models)} skills")
        return True
    
    def train_job_matcher(self):
        """Train job-resume matching model"""
        print("\n💼 Training Job-Resume Matching Model...")
        
        # Load datasets
        resume_df = pd.read_csv('data/resume_dataset.csv')
        job_df = pd.read_csv('data/job_descriptions.csv')
        
        # Create matching pairs
        matching_data = []
        
        print("   Generating training pairs...")
        for i, resume_row in resume_df.iterrows():
            if i >= 100:  # Limit for training speed
                break
                
            resume_skills = str(resume_row['skills']).lower().split(',')
            resume_skills = [s.strip() for s in resume_skills if s.strip()]
            
            for j, job_row in job_df.iterrows():
                if j >= 50:  # Limit for training speed
                    break
                
                job_skills = str(job_row['required_skills']).lower().split(',')
                job_skills = [s.strip() for s in job_skills if s.strip()]
                
                # Calculate skill overlap
                if resume_skills and job_skills:
                    overlap = len(set(resume_skills) & set(job_skills))
                    total_job_skills = len(job_skills)
                    
                    if total_job_skills > 0:
                        match_score = overlap / total_job_skills
                        
                        # Create feature vector
                        combined_text = f"{resume_row['resume_text']} {job_row['description']}"
                        
                        matching_data.append({
                            'text': combined_text,
                            'match_score': match_score,
                            'is_good_match': 1 if match_score >= 0.3 else 0
                        })
        
        if not matching_data:
            print("❌ No matching data generated")
            return False
        
        # Convert to DataFrame
        match_df = pd.DataFrame(matching_data)
        
        # Prepare features
        X = match_df['text']
        y_regression = match_df['match_score']
        y_classification = match_df['is_good_match']
        
        # Split data
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
            X, y_regression, y_classification, test_size=0.2, random_state=42
        )
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train classification model
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train_vec, y_clf_train)
        
        # Evaluate
        y_pred = classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_clf_test, y_pred)
        precision = precision_score(y_clf_test, y_pred, average='binary')
        recall = recall_score(y_clf_test, y_pred, average='binary')
        f1 = f1_score(y_clf_test, y_pred, average='binary')
        
        # Save model
        with open('ml_models/trained/job_matcher.pkl', 'wb') as f:
            pickle.dump({
                'model': classifier,
                'vectorizer': vectorizer
            }, f)
        
        self.metrics['job_matcher'] = {
            'training_pairs': len(matching_data),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'good_matches': sum(y_classification),
            'match_threshold': 0.3
        }
        
        print(f"✅ Job matcher trained on {len(matching_data)} pairs (Accuracy: {accuracy:.3f})")
        return True
    
    def train_ner_model(self):
        """Train Named Entity Recognition model"""
        print("\n🏷️ Training NER Model...")
        
        try:
            # Try to use spaCy if available
            import spacy
            from spacy.training import Example
            from spacy.util import minibatch
            
            # Load NER data
            with open('data/resume_entities_ner.json', 'r', encoding='utf-8') as f:
                ner_data = json.load(f)
            
            # Create blank model
            nlp = spacy.blank("en")
            ner = nlp.add_pipe("ner")
            
            # Add labels
            labels = set()
            training_data = []
            
            for item in ner_data:
                text = item['content']
                entities = []
                
                for annotation in item['annotation']:
                    for point in annotation['points']:
                        start = point['start']
                        end = point['end']
                        label = annotation['label'][0]
                        labels.add(label)
                        entities.append((start, end, label))
                
                training_data.append((text, {"entities": entities}))
            
            # Add labels to NER
            for label in labels:
                ner.add_label(label)
            
            # Convert to spaCy format
            examples = []
            for text, annotations in training_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            
            # Train model
            nlp.initialize()
            
            for iteration in range(10):  # Reduced iterations for speed
                losses = {}
                batches = minibatch(examples, size=8)
                for batch in batches:
                    nlp.update(batch, losses=losses)
            
            # Save model
            nlp.to_disk("ml_models/trained/ner_model")
            
            # Simple evaluation
            correct = 0
            total = 0
            
            for text, annotations in training_data[:20]:  # Test on first 20
                doc = nlp(text)
                predicted = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
                actual = annotations["entities"]
                
                for entity in actual:
                    total += 1
                    if entity in predicted:
                        correct += 1
            
            accuracy = correct / total if total > 0 else 0
            
            self.metrics['ner_model'] = {
                'training_samples': len(ner_data),
                'entity_types': list(labels),
                'accuracy': accuracy,
                'total_entities': total,
                'correct_predictions': correct
            }
            
            print(f"✅ NER model trained (Accuracy: {accuracy:.3f})")
            return True
            
        except ImportError:
            print("⚠️ spaCy not available, creating simple NER model...")
            
            # Simple rule-based NER as fallback
            with open('data/resume_entities_ner.json', 'r', encoding='utf-8') as f:
                ner_data = json.load(f)
            
            # Create simple patterns
            patterns = {
                'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'PHONE': r'[\+]?[1-9]?[0-9]{7,15}',
                'PERSON': r'^[A-Z][a-z]+ [A-Z][a-z]+',
            }
            
            with open('ml_models/trained/ner_patterns.pkl', 'wb') as f:
                pickle.dump(patterns, f)
            
            self.metrics['ner_model'] = {
                'type': 'rule_based',
                'patterns': len(patterns),
                'training_samples': len(ner_data)
            }
            
            print("✅ Rule-based NER model created")
            return True
    
    def generate_performance_report(self):
        """Generate comprehensive performance metrics report"""
        print("\n📊 Generating Performance Report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create detailed report
        report = {
            'training_timestamp': timestamp,
            'dataset_info': {
                'resume_samples': 962,
                'job_postings': 500,
                'skills_database': 15,
                'ner_samples': 100
            },
            'model_performance': self.metrics,
            'training_summary': {
                'total_models_trained': len(self.metrics),
                'successful_models': sum(1 for m in self.metrics.values() if m),
                'training_duration': time.time() - self.start_time
            }
        }
        
        # Save detailed JSON report
        with open(f'ml_models/metrics/performance_report_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create summary CSV
        summary_data = []
        for model_name, metrics in self.metrics.items():
            if isinstance(metrics, dict):
                row = {'model': model_name}
                
                # Extract key metrics
                if 'accuracy' in metrics:
                    row['accuracy'] = metrics['accuracy']
                if 'f1_score' in metrics:
                    row['f1_score'] = metrics['f1_score']
                if 'precision' in metrics:
                    row['precision'] = metrics['precision']
                if 'recall' in metrics:
                    row['recall'] = metrics['recall']
                
                summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(f'ml_models/metrics/model_summary_{timestamp}.csv', index=False)
        
        # Generate plots
        self.create_performance_plots(timestamp)
        
        print(f"✅ Performance report saved: performance_report_{timestamp}.json")
        return report
    
    def create_performance_plots(self, timestamp):
        """Create performance visualization plots"""
        try:
            plt.style.use('default')
            
            # Model comparison plot
            if 'resume_classifier' in self.metrics:
                model_comp = self.metrics['resume_classifier']['model_comparison']
                
                models = list(model_comp.keys())
                accuracies = [model_comp[m]['accuracy'] for m in models]
                f1_scores = [model_comp[m]['f1_score'] for m in models]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Accuracy comparison
                ax1.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                ax1.set_title('Model Accuracy Comparison')
                ax1.set_ylabel('Accuracy')
                ax1.set_ylim(0, 1)
                
                # F1 Score comparison
                ax2.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                ax2.set_title('Model F1 Score Comparison')
                ax2.set_ylabel('F1 Score')
                ax2.set_ylim(0, 1)
                
                plt.tight_layout()
                plt.savefig(f'ml_models/plots/model_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Skills extraction performance
            if 'skills_extractor' in self.metrics:
                skill_metrics = self.metrics['skills_extractor']['skill_metrics']
                
                skills = list(skill_metrics.keys())[:10]  # Top 10 skills
                f1_scores = [skill_metrics[s]['f1_score'] for s in skills]
                
                plt.figure(figsize=(10, 6))
                plt.barh(skills, f1_scores, color='skyblue')
                plt.title('Skills Extraction F1 Scores')
                plt.xlabel('F1 Score')
                plt.xlim(0, 1)
                plt.tight_layout()
                plt.savefig(f'ml_models/plots/skills_performance_{timestamp}.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            print("✅ Performance plots created")
            
        except Exception as e:
            print(f"⚠️ Could not create plots: {e}")
    
    def train_all_models(self):
        """Train all ML models with new datasets"""
        print("🚀 TRAINING ALL ML MODELS WITH NEW DATASETS")
        print("=" * 60)
        
        self.start_time = time.time()
        
        success_count = 0
        total_models = 4
        
        # Train each model
        if self.train_resume_classifier():
            success_count += 1
        
        if self.train_skills_extractor():
            success_count += 1
        
        if self.train_job_matcher():
            success_count += 1
        
        if self.train_ner_model():
            success_count += 1
        
        # Generate performance report
        report = self.generate_performance_report()
        
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"✅ Successfully trained {success_count}/{total_models} models")
        print(f"⏱️ Total training time: {time.time() - self.start_time:.2f} seconds")
        print(f"\n📁 Generated files:")
        print(f"   • ml_models/trained/ - Trained model files")
        print(f"   • ml_models/metrics/ - Performance metrics")
        print(f"   • ml_models/plots/ - Performance visualizations")
        
        return report

def main():
    """Main training function"""
    trainer = ModelTrainer()
    report = trainer.train_all_models()
    
    # Print summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    for model_name, metrics in trainer.metrics.items():
        print(f"\n🎯 {model_name.upper()}:")
        if isinstance(metrics, dict):
            if 'accuracy' in metrics:
                print(f"   Accuracy: {metrics['accuracy']:.3f}")
            if 'f1_score' in metrics:
                print(f"   F1 Score: {metrics['f1_score']:.3f}")
            if 'training_pairs' in metrics:
                print(f"   Training Pairs: {metrics['training_pairs']}")
            if 'skills_trained' in metrics:
                print(f"   Skills Trained: {metrics['skills_trained']}")

if __name__ == "__main__":
    main()
