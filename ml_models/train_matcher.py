import torch
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_loader import DatasetLoader

class ResumeJobDataset(Dataset):
    def __init__(self, resume_texts, job_texts, scores, tokenizer, max_length=512):
        self.resume_texts = resume_texts
        self.job_texts = job_texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.resume_texts)
    
    def __getitem__(self, idx):
        resume_text = str(self.resume_texts[idx])
        job_text = str(self.job_texts[idx])
        score = float(self.scores[idx])
        
        # Tokenize resume
        resume_encoding = self.tokenizer(
            resume_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize job description
        job_encoding = self.tokenizer(
            job_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'resume_input_ids': resume_encoding['input_ids'].flatten(),
            'resume_attention_mask': resume_encoding['attention_mask'].flatten(),
            'job_input_ids': job_encoding['input_ids'].flatten(),
            'job_attention_mask': job_encoding['attention_mask'].flatten(),
            'score': torch.tensor(score, dtype=torch.float)
        }

class ResumeJobMatcher(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(ResumeJobMatcher, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size * 2, 1)
        
    def forward(self, resume_input_ids, resume_attention_mask, job_input_ids, job_attention_mask):
        # Get BERT embeddings for resume
        resume_outputs = self.bert(
            input_ids=resume_input_ids,
            attention_mask=resume_attention_mask
        )
        resume_pooled = resume_outputs.pooler_output
        
        # Get BERT embeddings for job description
        job_outputs = self.bert(
            input_ids=job_input_ids,
            attention_mask=job_attention_mask
        )
        job_pooled = job_outputs.pooler_output
        
        # Concatenate embeddings
        combined = torch.cat((resume_pooled, job_pooled), dim=1)
        combined = self.dropout(combined)
        
        # Predict match score
        output = self.classifier(combined)
        return torch.sigmoid(output) * 100  # Scale to 0-100

class MatcherTrainer:
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_training_data(self):
        """Prepare training data from datasets"""
        print("Loading datasets...")
        loader = DatasetLoader()
        
        # Load resume and job datasets
        resume_df = loader.load_resume_dataset()
        job_df = loader.load_job_descriptions()
        
        # Create training pairs
        resume_texts = []
        job_texts = []
        scores = []
        
        print("Creating training pairs...")
        
        # Create positive and negative examples
        for _, resume in resume_df.iterrows():
            resume_text = f"{resume.get('name', '')} {resume.get('skills', '')} {resume.get('education', '')} {resume.get('resume_text', '')}"
            resume_skills = set(str(resume.get('skills', '')).lower().split(', '))
            resume_exp = resume.get('experience_years', 0)
            
            for _, job in job_df.iterrows():
                job_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('required_skills', '')}"
                job_skills = set(str(job.get('required_skills', '')).lower().split(', '))
                job_exp = job.get('experience_required', 0)
                
                # Calculate similarity score
                skill_overlap = len(resume_skills.intersection(job_skills))
                skill_total = len(job_skills) if job_skills else 1
                skill_match = (skill_overlap / skill_total) * 100
                
                # Experience match
                exp_match = min(resume_exp / max(job_exp, 1), 1.0) * 100 if job_exp > 0 else 50
                
                # Overall score (weighted average)
                overall_score = (skill_match * 0.7 + exp_match * 0.3)
                
                resume_texts.append(resume_text)
                job_texts.append(job_text)
                scores.append(overall_score)
        
        print(f"Created {len(resume_texts)} training pairs")
        return resume_texts, job_texts, scores
    
    def train_model(self, resume_texts, job_texts, scores, epochs=3, batch_size=8, learning_rate=2e-5):
        """Train the matcher model"""
        print("Preparing model for training...")
        
        # Split data
        train_resume, val_resume, train_job, val_job, train_scores, val_scores = train_test_split(
            resume_texts, job_texts, scores, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = ResumeJobDataset(train_resume, train_job, train_scores, self.tokenizer)
        val_dataset = ResumeJobDataset(val_resume, val_job, val_scores, self.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        self.model = ResumeJobMatcher(self.model_name)
        self.model.to(self.device)
        
        # Setup optimizer and loss
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move to device
                resume_input_ids = batch['resume_input_ids'].to(self.device)
                resume_attention_mask = batch['resume_attention_mask'].to(self.device)
                job_input_ids = batch['job_input_ids'].to(self.device)
                job_attention_mask = batch['job_attention_mask'].to(self.device)
                scores = batch['score'].to(self.device)
                
                # Forward pass
                outputs = self.model(resume_input_ids, resume_attention_mask, job_input_ids, job_attention_mask)
                loss = criterion(outputs.squeeze(), scores)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    resume_input_ids = batch['resume_input_ids'].to(self.device)
                    resume_attention_mask = batch['resume_attention_mask'].to(self.device)
                    job_input_ids = batch['job_input_ids'].to(self.device)
                    job_attention_mask = batch['job_attention_mask'].to(self.device)
                    scores = batch['score'].to(self.device)
                    
                    outputs = self.model(resume_input_ids, resume_attention_mask, job_input_ids, job_attention_mask)
                    loss = criterion(outputs.squeeze(), scores)
                    val_loss += loss.item()
            
            print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    def save_model(self, output_dir):
        """Save the trained model"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), output_path / 'model.pt')
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_path)
        
        # Save model config
        config = {
            'model_name': self.model_name,
            'device': str(self.device)
        }
        with open(output_path / 'config.json', 'w') as f:
            json.dump(config, f)
        
        print(f"Model saved to {output_path}")
    
    def test_model(self, resume_text, job_text):
        """Test the model with sample texts"""
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        self.model.eval()
        
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
        
        # Predict
        with torch.no_grad():
            score = self.model(resume_input_ids, resume_attention_mask, job_input_ids, job_attention_mask)
        
        return score.item()

def main():
    """Main training function"""
    print("Starting Resume-Job Matcher training...")
    
    # Initialize trainer
    trainer = MatcherTrainer()
    
    # Prepare training data
    resume_texts, job_texts, scores = trainer.prepare_training_data()
    
    # Train model
    trainer.train_model(resume_texts, job_texts, scores, epochs=2, batch_size=4)
    
    # Save model
    trainer.save_model("ml_models/trained_matcher")
    
    # Test with sample data
    sample_resume = "John Doe, Software Engineer with 5 years experience in Python, JavaScript, React"
    sample_job = "Senior Software Engineer position requiring Python, React, and 3+ years experience"
    
    print("\nTesting model:")
    score = trainer.test_model(sample_resume, sample_job)
    print(f"Match score: {score:.1f}%")
    
    print("Matcher training completed successfully!")

if __name__ == "__main__":
    main()
