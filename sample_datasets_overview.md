# 📊 Dataset Overview for AI Resume Parser

## 🎯 Available Datasets

### 1. **Resume Entities NER Dataset** (`/data/resume_entities_ner.json`)
**Purpose**: Train Named Entity Recognition model to extract information from resumes
**Size**: 50 samples (repeated from base template)
**Format**: JSON with annotated text

**Sample Entry**:
```json
{
  "content": "John Doe\nSoftware Engineer\nEmail: john.doe@email.com\nPhone: +1-555-0123\nLocation: San Francisco, CA\nSkills: Python, JavaScript, React, Node.js\nExperience: 5 years\nEducation: BS Computer Science, Stanford University\nCertification: AWS Certified Developer",
  "annotation": [
    {"label": ["PERSON"], "points": [{"start": 0, "end": 8, "text": "John Doe"}]},
    {"label": ["EMAIL"], "points": [{"start": 32, "end": 51, "text": "john.doe@email.com"}]},
    {"label": ["PHONE"], "points": [{"start": 59, "end": 71, "text": "+1-555-0123"}]},
    {"label": ["LOCATION"], "points": [{"start": 82, "end": 98, "text": "San Francisco, CA"}]},
    {"label": ["SKILLS"], "points": [{"start": 107, "end": 140, "text": "Python, JavaScript, React, Node.js"}]},
    {"label": ["EXPERIENCE_YEARS"], "points": [{"start": 153, "end": 160, "text": "5 years"}]},
    {"label": ["DEGREE"], "points": [{"start": 172, "end": 190, "text": "BS Computer Science"}]},
    {"label": ["COLLEGE"], "points": [{"start": 192, "end": 210, "text": "Stanford University"}]},
    {"label": ["CERTIFICATION"], "points": [{"start": 225, "end": 248, "text": "AWS Certified Developer"}]}
  ]
}
```

**Entity Labels**:
- `PERSON`: Candidate name
- `EMAIL`: Email address
- `PHONE`: Phone number
- `LOCATION`: Geographic location
- `SKILLS`: Technical skills
- `EXPERIENCE_YEARS`: Years of experience
- `DEGREE`: Educational degree
- `COLLEGE`: Educational institution
- `CERTIFICATION`: Professional certifications

---

### 2. **Resume Dataset** (Generated - 200 samples)
**Purpose**: Resume classification and structured data extraction
**Format**: CSV with structured resume information

**Columns**:
- `name`: Candidate name
- `email`: Email address
- `phone`: Phone number
- `location`: Location
- `skills`: Comma-separated skills
- `experience_years`: Years of experience
- `education`: Educational background
- `category`: Job category (IT, Marketing, etc.)
- `resume_text`: Full resume text

**Sample Data**:
```csv
name,email,phone,location,skills,experience_years,education,category,resume_text
John Doe,john.doe@email.com,+1-555-0123,San Francisco CA,"Python, JavaScript, React, Node.js, AWS",5,BS Computer Science Stanford University,IT,Experienced software engineer with 5 years of experience in full-stack development...
Jane Smith,jane.smith@email.com,+1-555-0124,New York NY,"Java, Spring Boot, MySQL, Docker, Kubernetes",7,MS Computer Science MIT,IT,Senior backend developer with expertise in microservices architecture...
```

---

### 3. **Job Descriptions Dataset** (Generated - 100 samples)
**Purpose**: Job posting data for resume-job matching
**Format**: CSV with job requirements and descriptions

**Columns**:
- `title`: Job title
- `company`: Company name
- `location`: Job location
- `description`: Job description
- `required_skills`: Required skills
- `experience_required`: Required years of experience
- `salary_range`: Salary range
- `employment_type`: Employment type

**Sample Data**:
```csv
title,company,location,description,required_skills,experience_required,salary_range,employment_type
Senior Software Engineer,Tech Corp,San Francisco CA,We are looking for a senior software engineer with experience in Python React and AWS...,"Python, React, AWS, Docker",5,$120000 - $160000,Full-time
Full Stack Developer,StartupXYZ,Austin TX,Join our growing team as a full stack developer. Experience with Node.js and React required...,"JavaScript, Node.js, React, MongoDB",3,$80000 - $120000,Full-time
```

---

### 4. **Skills Dataset** (Generated - 10 core skills)
**Purpose**: Skills categorization and popularity mapping
**Format**: CSV with skill information

**Columns**:
- `skill`: Skill name
- `category`: Skill category
- `popularity`: Popularity score (0-100)

**Sample Data**:
```csv
skill,category,popularity
Python,Programming Language,95
JavaScript,Programming Language,92
React,Frontend Framework,88
Node.js,Backend Framework,85
AWS,Cloud Platform,90
Docker,DevOps Tool,82
Kubernetes,DevOps Tool,78
Java,Programming Language,89
Spring Boot,Backend Framework,75
MySQL,Database,80
```

---

### 5. **Simple Training Data** (Hardcoded - 50 samples)
**Purpose**: Fallback NER training data when other sources unavailable
**Format**: Python tuples with text and entity annotations

**Sample Training Entry**:
```python
(
    "John Doe is a Software Engineer. Email: john.doe@email.com Phone: +1-555-0123 Location: San Francisco, CA Skills: Python, JavaScript, React Experience: 5 years Education: BS Computer Science, Stanford University",
    {
        "entities": [
            (0, 8, "PERSON"),           # "John Doe"
            (32, 51, "EMAIL"),          # "john.doe@email.com"
            (59, 71, "PHONE"),          # "+1-555-0123"
            (82, 98, "LOCATION"),       # "San Francisco, CA"
            (107, 140, "SKILLS"),       # "Python, JavaScript, React"
            (153, 160, "EXPERIENCE_YEARS"), # "5 years"
            (172, 210, "EDUCATION")     # "BS Computer Science, Stanford University"
        ]
    }
)
```

---

## 🔄 Dataset Loading Strategy

### Primary Source: Kaggle Datasets
1. **`dataturks/resume-entities-for-ner`** - Real annotated resume data
2. **`jillianseed/resume-data`** - Resume classification dataset
3. **`andrewmvd/adzuna-job-salary-predictions`** - Job descriptions
4. **`nigelsmithdigital/global-skill-mapping`** - Skills mapping

### Fallback: Generated Data
- When Kaggle API unavailable or datasets restricted
- Programmatically generated realistic training data
- Ensures models can be trained offline

---

## 🎯 Usage in ML Models

### NER Model Training:
- **Input**: Resume text with entity annotations
- **Output**: Trained spaCy NER model
- **Purpose**: Extract structured data from raw resume text

### Job Matching Model:
- **Input**: Resume data + Job descriptions
- **Output**: BERT-based similarity model
- **Purpose**: Calculate compatibility scores between resumes and jobs

### AI Resume Analyzer:
- **Input**: All datasets combined
- **Output**: Comprehensive candidate analysis
- **Purpose**: Intelligent resume evaluation and job matching

---

## 📈 Dataset Statistics

| Dataset | Size | Format | Purpose |
|---------|------|--------|---------|
| NER Training | 50 samples | JSON | Entity extraction |
| Resume Data | 200 samples | CSV | Classification |
| Job Data | 100 samples | CSV | Job matching |
| Skills Data | 10 skills | CSV | Skills mapping |
| **Total** | **360 samples** | **Mixed** | **Complete AI training** |

The system uses a combination of **real-world data** (when available) and **synthetic data** (as fallback) to ensure robust AI model training regardless of external dependencies! 🚀
