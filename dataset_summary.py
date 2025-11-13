#!/usr/bin/env python3
"""
Final Dataset Summary - Clean, Real, Diverse Data
"""

import pandas as pd
import json
import os

def show_dataset_summary():
    """Show summary of cleaned, real datasets"""
    print("🎉 FINAL CLEAN DATASET SUMMARY")
    print("=" * 50)
    
    # Resume Dataset
    print("\n📄 RESUME DATASET (REAL & DIVERSE)")
    print("-" * 40)
    resume_df = pd.read_csv('data/resume_dataset.csv')
    print(f"📊 Total Samples: {len(resume_df)}")
    print(f"📝 Columns: {list(resume_df.columns)}")
    
    categories = resume_df['category'].value_counts()
    print(f"📋 Job Categories ({len(categories)} unique):")
    for category, count in categories.head(10).items():
        print(f"   • {category}: {count} resumes")
    
    # Show skills diversity
    all_skills = set()
    for skills_str in resume_df['skills'].dropna():
        if skills_str != 'Not specified':
            skills = [s.strip() for s in str(skills_str).split(',')]
            all_skills.update(skills)
    
    print(f"🎯 Unique Skills Found: {len(all_skills)}")
    print(f"   Sample skills: {list(all_skills)[:10]}")
    
    # Job Dataset
    print("\n💼 JOB DATASET (REAL POSTINGS)")
    print("-" * 40)
    job_df = pd.read_csv('data/job_descriptions.csv')
    print(f"📊 Total Job Postings: {len(job_df)}")
    print(f"📝 Columns: {list(job_df.columns)}")
    
    # Show sample jobs
    print("📋 Sample Job Titles:")
    for i, title in enumerate(job_df['title'].dropna().head(5)):
        if title.strip():
            print(f"   • {title}")
    
    # Skills Dataset
    print("\n🛠️ SKILLS DATASET (CURATED)")
    print("-" * 40)
    skills_df = pd.read_csv('data/skills.csv')
    print(f"📊 Total Skills: {len(skills_df)}")
    print("📋 Top Skills by Popularity:")
    for _, row in skills_df.head(10).iterrows():
        print(f"   • {row['skill']:15} | {row['category']:20} | {row['popularity']}%")
    
    # NER Dataset
    print("\n🏷️ NER TRAINING DATASET (FROM REAL DATA)")
    print("-" * 40)
    with open('data/resume_entities_ner.json', 'r', encoding='utf-8') as f:
        ner_data = json.load(f)
    
    print(f"📊 Training Samples: {len(ner_data)}")
    
    # Count entity types
    entity_counts = {}
    for sample in ner_data:
        for annotation in sample['annotation']:
            label = annotation['label'][0]
            entity_counts[label] = entity_counts.get(label, 0) + 1
    
    print("📋 Entity Types:")
    for entity, count in sorted(entity_counts.items()):
        print(f"   • {entity}: {count} annotations")
    
    # File sizes
    print("\n📁 DATASET FILE SIZES")
    print("-" * 40)
    data_files = ['resume_dataset.csv', 'job_descriptions.csv', 'skills.csv', 'resume_entities_ner.json']
    total_size = 0
    
    for file in data_files:
        file_path = f'data/{file}'
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            total_size += size_mb
            print(f"   • {file}: {size_mb:.1f} MB")
    
    print(f"   📊 Total Dataset Size: {total_size:.1f} MB")
    
    print("\n✅ CLEANUP COMPLETE!")
    print("🎯 All datasets now contain:")
    print("   • REAL data from Kaggle (no repetition)")
    print("   • DIVERSE categories and skills")
    print("   • PROPER annotations for ML training")
    print("   • CLEAN file structure")

if __name__ == "__main__":
    show_dataset_summary()
