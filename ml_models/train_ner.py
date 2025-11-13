import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import json
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_loader import DatasetLoader

class NERTrainer:
    def __init__(self, model_name="en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self.ner = None
        
    def load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            # Try to load existing model
            self.nlp = spacy.load(self.model_name)
            print(f"Loaded existing model: {self.model_name}")
        except OSError:
            # Create blank model if not found
            self.nlp = spacy.blank("en")
            print("Created blank English model")
        
        # Get or create NER component
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner")
        else:
            self.ner = self.nlp.get_pipe("ner")
    
    def prepare_training_data(self, raw_data):
        """Convert raw data to spaCy training format"""
        training_data = []
        
        for item in raw_data:
            text = item.get('content', '')
            annotations = item.get('annotation', [])
            
            entities = []
            for annotation in annotations:
                for point in annotation.get('points', []):
                    start = point['start']
                    end = point['end']
                    label = annotation['label'][0]
                    entities.append((start, end, label))
            
            training_data.append((text, {"entities": entities}))
        
        return training_data
    
    def add_labels(self, training_data):
        """Add labels to the NER model"""
        labels = set()
        for text, annotations in training_data:
            for start, end, label in annotations["entities"]:
                labels.add(label)
        
        for label in labels:
            self.ner.add_label(label)
        
        print(f"Added labels: {labels}")
        return labels
    
    def train_model(self, training_data, n_iter=30, drop_rate=0.2):
        """Train the NER model"""
        print(f"Training NER model with {len(training_data)} examples...")
        
        # Convert to spaCy Example format
        examples = []
        for text, annotations in training_data:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        
        # Disable other pipes during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            # Initialize the model
            self.nlp.initialize()
            
            # Training loop
            for iteration in range(n_iter):
                print(f"Iteration {iteration + 1}/{n_iter}")
                
                # Shuffle examples
                random.shuffle(examples)
                
                losses = {}
                batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
                
                for batch in batches:
                    self.nlp.update(batch, drop=drop_rate, losses=losses)
                
                print(f"Losses: {losses}")
        
        print("Training completed!")
    
    def evaluate_model(self, test_data):
        """Evaluate the trained model"""
        print("Evaluating model...")
        
        correct = 0
        total = 0
        
        for text, annotations in test_data:
            doc = self.nlp(text)
            predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            actual_entities = annotations["entities"]
            
            for entity in actual_entities:
                total += 1
                if entity in predicted_entities:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        return accuracy
    
    def save_model(self, output_dir):
        """Save the trained model"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.nlp.to_disk(output_path)
        print(f"Model saved to {output_path}")
    
    def test_model(self, text):
        """Test the model with sample text"""
        doc = self.nlp(text)
        
        print(f"Text: {text}")
        print("Entities found:")
        for ent in doc.ents:
            print(f"  {ent.text} -> {ent.label_}")
        
        return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]

def main():
    """Main training function"""
    print("Starting NER model training...")
    
    # Initialize trainer
    trainer = NERTrainer()
    trainer.load_or_create_model()
    
    # Load training data
    loader = DatasetLoader()
    raw_data = loader.load_resume_entities_ner()
    
    # Prepare training data
    training_data = trainer.prepare_training_data(raw_data)
    
    # Split data for training and testing
    split_idx = int(0.8 * len(training_data))
    train_data = training_data[:split_idx]
    test_data = training_data[split_idx:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Add labels
    trainer.add_labels(train_data)
    
    # Train model
    trainer.train_model(train_data, n_iter=20)
    
    # Evaluate model
    if test_data:
        trainer.evaluate_model(test_data)
    
    # Save model
    trainer.save_model("ml_models/trained_ner_model")
    
    # Test with sample text
    sample_text = """
    John Doe
    Software Engineer
    Email: john.doe@email.com
    Phone: +1-555-0123
    Location: San Francisco, CA
    Skills: Python, JavaScript, React, Node.js
    Experience: 5 years
    Education: BS Computer Science, Stanford University
    """
    
    print("\nTesting model with sample text:")
    trainer.test_model(sample_text)
    
    print("NER training completed successfully!")

if __name__ == "__main__":
    main()
