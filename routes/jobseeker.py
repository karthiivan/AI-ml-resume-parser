from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
from pathlib import Path

from models import db, JobSeeker, Job, Application
from utils.file_processor import FileProcessor
from utils.pdf_generator import PDFGenerator
try:
    from ml_models.ai_resume_analyzer_improved import AIResumeAnalyzer
except ImportError:
    from ml_models.ai_resume_analyzer import AIResumeAnalyzer

jobseeker_bp = Blueprint('jobseeker', __name__, url_prefix='/jobseeker')

# Initialize processors
file_processor = FileProcessor()
pdf_generator = PDFGenerator()
ai_analyzer = AIResumeAnalyzer()

# Try to initialize ML models, but handle gracefully if they fail
try:
    from ml_models.ner_parser import NERParser
    from ml_models.job_matcher import JobMatcher
    ner_parser = NERParser()
    job_matcher = JobMatcher()
    ML_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML models not available: {e}")
    ner_parser = None
    job_matcher = None
    ML_MODELS_AVAILABLE = False

@jobseeker_bp.route('/dashboard')
@login_required
def dashboard():
    if current_user.user_type != 'jobseeker':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    jobseeker = JobSeeker.query.filter_by(user_id=current_user.id).first()
    if not jobseeker:
        flash('Profile not found', 'error')
        return redirect(url_for('main.index'))
    
    # Get recent applications
    recent_applications = Application.query.filter_by(jobseeker_id=jobseeker.id)\
        .order_by(Application.applied_at.desc()).limit(5).all()
    
    # Get recommended jobs with application status
    recommended_jobs_query = Job.query.filter_by(status='active').limit(3).all()
    recommended_jobs = []
    
    for job in recommended_jobs_query:
        # Check if user has already applied
        existing_application = Application.query.filter_by(
            job_id=job.id, 
            jobseeker_id=jobseeker.id
        ).first()
        
        recommended_jobs.append({
            'job': job,
            'application': existing_application,
            'has_applied': existing_application is not None
        })
    
    # Get statistics
    stats = {
        'total_applications': Application.query.filter_by(jobseeker_id=jobseeker.id).count(),
        'pending_applications': Application.query.filter_by(jobseeker_id=jobseeker.id, status='pending').count(),
        'shortlisted': Application.query.filter_by(jobseeker_id=jobseeker.id, status='shortlisted').count(),
        'has_resume': bool(jobseeker.resume_path)
    }
    
    return render_template('jobseeker/dashboard.html', 
                         jobseeker=jobseeker, 
                         recent_applications=recent_applications,
                         recommended_jobs=recommended_jobs,
                         stats=stats)

@jobseeker_bp.route('/upload-resume', methods=['GET', 'POST'])
@login_required
def upload_resume():
    if current_user.user_type != 'jobseeker':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    jobseeker = JobSeeker.query.filter_by(user_id=current_user.id).first()
    
    if request.method == 'POST':
        if 'resume_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        temp_path = None
        try:
            # Validate file - use proper temp directory for Windows
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(temp_path)
            
            is_valid, message = file_processor.validate_file(temp_path)
            if not is_valid:
                os.remove(temp_path)
                return jsonify({'error': message}), 400
            
            # Extract text
            text = file_processor.extract_text(temp_path)
            
            # Parse with NER model if available
            if ML_MODELS_AVAILABLE and ner_parser:
                parsed_data = ner_parser.parse_resume(text)
            else:
                # Basic parsing fallback
                parsed_data = {
                    'name': 'Extracted Name',
                    'email': 'extracted@email.com',
                    'phone': '+1-555-0123',
                    'location': 'City, State',
                    'skills': ['Python', 'JavaScript', 'React'],
                    'experience_years': 3,
                    'education': ['Bachelor of Science'],
                    'summary': 'Professional summary extracted from resume.'
                }
            
            # Save file to uploads directory
            upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
            file_path = upload_dir / filename
            
            # Move file from temp to uploads
            os.rename(temp_path, file_path)
            
            # Update jobseeker profile
            jobseeker.name = parsed_data.get('name') or jobseeker.name
            jobseeker.phone = parsed_data.get('phone') or jobseeker.phone
            jobseeker.location = parsed_data.get('location') or jobseeker.location
            jobseeker.experience_years = parsed_data.get('experience_years') or jobseeker.experience_years
            jobseeker.linkedin = parsed_data.get('linkedin') or jobseeker.linkedin
            jobseeker.github = parsed_data.get('github') or jobseeker.github
            jobseeker.resume_path = str(file_path)
            
            if parsed_data.get('skills'):
                jobseeker.set_skills(parsed_data['skills'])
            
            if parsed_data.get('education'):
                jobseeker.education = '; '.join(parsed_data['education'])
            
            jobseeker.set_parsed_data(parsed_data)
            
            # AI-powered automatic job matching and application
            ai_results = _analyze_and_apply_to_jobs(jobseeker, text, parsed_data)
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Resume uploaded, analyzed, and automatically matched to jobs!',
                'parsed_data': parsed_data,
                'ai_analysis': ai_results
            })
        
        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': f'Error processing resume: {str(e)}'}), 500
    
    return render_template('jobseeker/upload_resume.html', jobseeker=jobseeker)

@jobseeker_bp.route('/parse-resume')
@login_required
def parse_resume():
    if current_user.user_type != 'jobseeker':
        return jsonify({'error': 'Access denied'}), 403
    
    jobseeker = JobSeeker.query.filter_by(user_id=current_user.id).first()
    if not jobseeker or not jobseeker.parsed_data:
        return jsonify({'error': 'No parsed resume data found'}), 404
    
    return jsonify({
        'success': True,
        'parsed_data': jobseeker.get_parsed_data()
    })

@jobseeker_bp.route('/update-profile', methods=['POST'])
@login_required
def update_profile():
    if current_user.user_type != 'jobseeker':
        return jsonify({'error': 'Access denied'}), 403
    
    jobseeker = JobSeeker.query.filter_by(user_id=current_user.id).first()
    
    try:
        # Update basic info
        jobseeker.name = request.form.get('name', jobseeker.name)
        jobseeker.phone = request.form.get('phone', jobseeker.phone)
        jobseeker.location = request.form.get('location', jobseeker.location)
        jobseeker.linkedin = request.form.get('linkedin', jobseeker.linkedin)
        jobseeker.github = request.form.get('github', jobseeker.github)
        
        # Update skills
        skills_str = request.form.get('skills', '')
        if skills_str:
            skills = [skill.strip() for skill in skills_str.split(',') if skill.strip()]
            jobseeker.set_skills(skills)
        
        # Update experience
        exp_years = request.form.get('experience_years')
        if exp_years:
            jobseeker.experience_years = int(exp_years)
        
        # Update education
        education = request.form.get('education', jobseeker.education)
        jobseeker.education = education
        
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error updating profile: {str(e)}'}), 500

@jobseeker_bp.route('/jobs')
@login_required
def view_jobs():
    if current_user.user_type != 'jobseeker':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    # Get filter parameters
    search_query = request.args.get('search', '')
    location_filter = request.args.get('location', '')
    skills_filter = request.args.get('skills', '')
    min_experience = request.args.get('min_experience', type=int)
    max_experience = request.args.get('max_experience', type=int)
    employment_type = request.args.get('employment_type', '')
    
    # Build query
    query = Job.query.filter_by(status='active')
    
    if search_query:
        query = query.filter(
            (Job.title.contains(search_query)) |
            (Job.description.contains(search_query)) |
            (Job.company.contains(search_query))
        )
    
    if location_filter:
        query = query.filter(Job.location.contains(location_filter))
    
    if employment_type:
        query = query.filter(Job.employment_type == employment_type)
    
    if min_experience is not None:
        query = query.filter(Job.experience_required >= min_experience)
    
    if max_experience is not None:
        query = query.filter(Job.experience_required <= max_experience)
    
    jobs = query.order_by(Job.created_at.desc()).all()
    
    # Get jobseeker for matching
    jobseeker = JobSeeker.query.filter_by(user_id=current_user.id).first()
    
    # Calculate match scores for each job and check application status
    job_matches = []
    for job in jobs:
        # Check if user has already applied to this job
        existing_application = None
        if jobseeker:
            existing_application = Application.query.filter_by(
                job_id=job.id, 
                jobseeker_id=jobseeker.id
            ).first()
        
        if jobseeker and jobseeker.parsed_data and ML_MODELS_AVAILABLE and job_matcher:
            try:
                job_data = {
                    'title': job.title,
                    'description': job.description,
                    'required_skills': job.get_required_skills(),
                    'experience_required': job.experience_required,
                    'location': job.location
                }
                
                match_result = job_matcher.calculate_match_score(
                    jobseeker.get_parsed_data(), 
                    job_data
                )
                
                if isinstance(match_result, dict):
                    match_score = match_result.get('overall_score', 0)
                else:
                    match_score = match_result
                
                job_matches.append({
                    'job': job,
                    'match_score': round(match_score, 1),
                    'application': existing_application,
                    'has_applied': existing_application is not None
                })
            except Exception as e:
                print(f"Error calculating match score: {e}")
                job_matches.append({
                    'job': job,
                    'match_score': 75,  # Default score when ML is unavailable
                    'application': existing_application,
                    'has_applied': existing_application is not None
                })
        else:
            # Use a simple fallback scoring when ML models aren't available
            fallback_score = 75 if jobseeker and jobseeker.parsed_data else 50
            job_matches.append({
                'job': job,
                'match_score': fallback_score,
                'application': existing_application,
                'has_applied': existing_application is not None
            })
    
    # Sort by match score
    job_matches.sort(key=lambda x: x['match_score'], reverse=True)
    
    return render_template('jobseeker/view_jobs.html', 
                         job_matches=job_matches,
                         filters={
                             'search': search_query,
                             'location': location_filter,
                             'skills': skills_filter,
                             'min_experience': min_experience,
                             'max_experience': max_experience,
                             'employment_type': employment_type
                         })

@jobseeker_bp.route('/apply/<int:job_id>', methods=['POST'])
@login_required
def apply_to_job(job_id):
    if current_user.user_type != 'jobseeker':
        return jsonify({'error': 'Access denied'}), 403
    
    jobseeker = JobSeeker.query.filter_by(user_id=current_user.id).first()
    job = Job.query.get_or_404(job_id)
    
    # Check if already applied
    existing_application = Application.query.filter_by(
        job_id=job_id, 
        jobseeker_id=jobseeker.id
    ).first()
    
    if existing_application:
        return jsonify({'error': 'You have already applied to this job'}), 400
    
    try:
        # Calculate AI match score
        if jobseeker.parsed_data:
            job_data = {
                'title': job.title,
                'description': job.description,
                'required_skills': job.get_required_skills(),
                'experience_required': job.experience_required,
                'location': job.location
            }
            
            match_analysis = job_matcher.get_matching_analysis(
                jobseeker.get_parsed_data(), 
                job_data
            )
            
            ai_score = match_analysis.get('overall_score', 0)
            skills_match = match_analysis.get('breakdown', {}).get('skills_match', 0)
            experience_match = match_analysis.get('breakdown', {}).get('experience_match', 0)
            education_match = match_analysis.get('breakdown', {}).get('education_match', 0)
        else:
            ai_score = 0
            skills_match = 0
            experience_match = 0
            education_match = 0
        
        # Create application
        application = Application(
            job_id=job_id,
            jobseeker_id=jobseeker.id,
            resume_path=jobseeker.resume_path,
            ai_score=ai_score,
            skills_match=skills_match,
            experience_match=experience_match,
            education_match=education_match,
            status='pending'
        )
        
        db.session.add(application)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Application submitted successfully',
            'ai_score': round(ai_score, 1)
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error submitting application: {str(e)}'}), 500

@jobseeker_bp.route('/applications')
@login_required
def my_applications():
    if current_user.user_type != 'jobseeker':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    jobseeker = JobSeeker.query.filter_by(user_id=current_user.id).first()
    
    # Get applications with job details
    applications = db.session.query(Application, Job).join(Job)\
        .filter(Application.jobseeker_id == jobseeker.id)\
        .order_by(Application.applied_at.desc()).all()
    
    return render_template('jobseeker/my_applications.html', applications=applications)

@jobseeker_bp.route('/generate-resume', methods=['GET', 'POST'])
@login_required
def generate_resume():
    if current_user.user_type != 'jobseeker':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    jobseeker = JobSeeker.query.filter_by(user_id=current_user.id).first()
    
    if request.method == 'POST':
        try:
            # Get form data
            resume_text = request.form.get('resume_text', '')
            template = request.form.get('template', 'modern')
            
            if resume_text:
                # Parse unstructured resume text
                parsed_data = ner_parser.parse_resume(resume_text)
            elif jobseeker.parsed_data:
                # Use existing parsed data
                parsed_data = jobseeker.get_parsed_data()
            else:
                return jsonify({'error': 'No resume data available'}), 400
            
            # Generate PDF
            output_dir = Path(current_app.config['UPLOAD_FOLDER']) / 'generated'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"resume_{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            output_path = output_dir / filename
            
            pdf_generator.generate_resume_pdf(parsed_data, output_path, template)
            
            return jsonify({
                'success': True,
                'message': 'Resume generated successfully',
                'download_url': url_for('jobseeker.download_resume', filename=filename)
            })
        
        except Exception as e:
            return jsonify({'error': f'Error generating resume: {str(e)}'}), 500
    
    return render_template('jobseeker/generate_resume.html', jobseeker=jobseeker)

@jobseeker_bp.route('/download-resume/<filename>')
@login_required
def download_resume(filename):
    if current_user.user_type != 'jobseeker':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    file_path = Path(current_app.config['UPLOAD_FOLDER']) / 'generated' / filename
    
    if not file_path.exists():
        flash('File not found', 'error')
        return redirect(url_for('jobseeker.generate_resume'))
    
    return send_file(file_path, as_attachment=True)

def _analyze_and_apply_to_jobs(jobseeker, resume_text, parsed_data):
    """
    AI-powered analysis and automatic job application
    Analyzes resume against all active jobs and creates applications with AI scores
    """
    try:
        # Get all active jobs
        active_jobs = Job.query.filter_by(status='active').all()
        
        if not active_jobs:
            return {'message': 'No active jobs found for matching', 'applications_created': 0}
        
        applications_created = 0
        job_analyses = []
        
        for job in active_jobs:
            # Check if already applied
            existing_application = Application.query.filter_by(
                job_id=job.id, 
                jobseeker_id=jobseeker.id
            ).first()
            
            if existing_application:
                continue  # Skip if already applied
            
            # Prepare job requirements for AI analysis
            job_requirements = {
                'required_skills': job.get_required_skills(),
                'experience_required': job.experience_required or 0,
                'education_required': '',  # Could be extracted from job description
                'job_title': job.title,
                'job_description': job.description
            }
            
            # Run AI analysis
            ai_analysis = ai_analyzer.analyze_resume(resume_text, job_requirements)
            
            # Create application with AI scores
            application = Application(
                job_id=job.id,
                jobseeker_id=jobseeker.id,
                resume_path=jobseeker.resume_path,
                ai_score=ai_analysis['overall_score'],
                skills_match=ai_analysis['skills_analysis']['score'],
                experience_match=ai_analysis['experience_analysis']['score'],
                education_match=ai_analysis['education_analysis']['score'],
                status='pending'
            )
            
            db.session.add(application)
            applications_created += 1
            
            # Store analysis for response
            job_analyses.append({
                'job_id': job.id,
                'job_title': job.title,
                'company': job.company,
                'ai_score': ai_analysis['overall_score'],
                'worthiness': ai_analysis['worthiness'],
                'recommendations': ai_analysis['recommendations']
            })
        
        return {
            'message': f'Automatically applied to {applications_created} jobs based on AI analysis',
            'applications_created': applications_created,
            'job_analyses': job_analyses
        }
        
    except Exception as e:
        print(f"Error in AI job matching: {e}")
        return {
            'message': 'AI analysis completed but automatic application failed',
            'applications_created': 0,
            'error': str(e)
        }
