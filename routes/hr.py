from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file, current_app
from flask_login import login_required, current_user
from datetime import datetime
from pathlib import Path
import json

from models import db, HRUser, Job, Application, JobSeeker, User
from utils.pdf_generator import PDFGenerator
from sqlalchemy.orm import joinedload

hr_bp = Blueprint('hr', __name__, url_prefix='/hr')

# Initialize processors
pdf_generator = PDFGenerator()

# Try to initialize ML models, but handle gracefully if they fail
try:
    from ml_models.job_matcher import JobMatcher
    job_matcher = JobMatcher()
    ML_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML models not available: {e}")
    job_matcher = None
    ML_MODELS_AVAILABLE = False

@hr_bp.route('/dashboard')
@login_required
def dashboard():
    if current_user.user_type != 'hr':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
    if not hr_user:
        flash('HR profile not found', 'error')
        return redirect(url_for('main.index'))
    
    # Get statistics
    total_jobs = Job.query.filter_by(hr_id=hr_user.id).count()
    active_jobs = Job.query.filter_by(hr_id=hr_user.id, status='active').count()
    
    # Get total applications for HR's jobs
    total_applications = db.session.query(Application).join(Job)\
        .filter(Job.hr_id == hr_user.id).count()
    
    # Get shortlisted candidates
    shortlisted = db.session.query(Application).join(Job)\
        .filter(Job.hr_id == hr_user.id, Application.status == 'shortlisted').count()
    
    # Get recent applications with proper relationships
    recent_applications = Application.query\
        .options(joinedload(Application.job), joinedload(Application.jobseeker).joinedload(JobSeeker.user))\
        .join(Job).join(JobSeeker).join(User, JobSeeker.user_id == User.id)\
        .filter(Job.hr_id == hr_user.id)\
        .order_by(Application.applied_at.desc()).limit(5).all()
    
    # Get active jobs for display with application counts
    active_jobs_list = Job.query.filter_by(hr_id=hr_user.id, status='active')\
        .options(joinedload(Job.applications))\
        .order_by(Job.created_at.desc()).limit(5).all()
    
    stats = {
        'total_jobs': total_jobs,
        'total_applications': total_applications,
        'pending_review': total_applications - shortlisted,
        'shortlisted': shortlisted
    }
    
    return render_template('hr/dashboard.html', 
                         hr_user=hr_user,
                         stats=stats,
                         recent_applications=recent_applications,
                         active_jobs=active_jobs_list)

@hr_bp.route('/post-job', methods=['GET', 'POST'])
@login_required
def post_job():
    if current_user.user_type != 'hr':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
    
    if request.method == 'POST':
        try:
            # Get form data
            title = request.form.get('title')
            description = request.form.get('description')
            location = request.form.get('location')
            experience_required = request.form.get('experience_required', type=int)
            salary_range = request.form.get('salary_range')
            employment_type = request.form.get('employment_type')
            required_skills = request.form.get('required_skills')
            
            # Validation
            if not all([title, description, location]):
                flash('Please fill in all required fields', 'error')
                return render_template('hr/post_job.html', hr_user=hr_user)
            
            # Process skills
            skills_list = []
            if required_skills:
                skills_list = [skill.strip() for skill in required_skills.split(',') if skill.strip()]
            
            # Create job
            job = Job(
                hr_id=hr_user.id,
                title=title,
                company=hr_user.company_name,
                location=location,
                description=description,
                experience_required=experience_required or 0,
                salary_range=salary_range,
                employment_type=employment_type,
                status='active'
            )
            job.set_required_skills(skills_list)
            
            db.session.add(job)
            db.session.commit()
            
            flash('Job posted successfully!', 'success')
            return redirect(url_for('hr.view_jobs'))
        
        except Exception as e:
            db.session.rollback()
            flash(f'Error posting job: {str(e)}', 'error')
    
    return render_template('hr/post_job.html', hr_user=hr_user)

@hr_bp.route('/jobs')
@login_required
def view_jobs():
    if current_user.user_type != 'hr':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
    
    # Get jobs with application counts
    jobs_with_counts = db.session.query(
        Job,
        db.func.count(Application.id).label('application_count')
    ).outerjoin(Application)\
     .filter(Job.hr_id == hr_user.id)\
     .group_by(Job.id)\
     .order_by(Job.created_at.desc()).all()
    
    # Extract jobs from the query result
    jobs = [job for job, count in jobs_with_counts]
    
    return render_template('hr/view_jobs.html', jobs=jobs)

@hr_bp.route('/job/<int:job_id>/toggle-status', methods=['POST'])
@login_required
def toggle_job_status(job_id):
    if current_user.user_type != 'hr':
        return jsonify({'error': 'Access denied'}), 403
    
    hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
    job = Job.query.filter_by(id=job_id, hr_id=hr_user.id).first_or_404()
    
    try:
        job.status = 'inactive' if job.status == 'active' else 'active'
        db.session.commit()
        
        return jsonify({
            'success': True,
            'new_status': job.status,
            'message': f'Job {job.status}'
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error updating job status: {str(e)}'}), 500

@hr_bp.route('/applications')
@login_required
def view_applications():
    if current_user.user_type != 'hr':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
    
    # Get filter parameters
    job_filter = request.args.get('job_id', type=int)
    status_filter = request.args.get('status', '')
    sort_by = request.args.get('sort_by', 'date')
    
    # Build query - use proper joins to get Application objects with relationships
    query = Application.query\
        .options(joinedload(Application.job), joinedload(Application.jobseeker).joinedload(JobSeeker.user))\
        .join(Job).join(JobSeeker).join(User, JobSeeker.user_id == User.id)\
        .filter(Job.hr_id == hr_user.id)
    
    if job_filter:
        query = query.filter(Job.id == job_filter)
    
    if status_filter:
        query = query.filter(Application.status == status_filter)
    
    # Apply sorting
    if sort_by == 'score':
        query = query.order_by(Application.ai_score.desc())
    elif sort_by == 'name':
        query = query.order_by(JobSeeker.name)
    else:  # date
        query = query.order_by(Application.applied_at.desc())
    
    applications = query.all()
    
    # Get jobs for filter dropdown
    jobs = Job.query.filter_by(hr_id=hr_user.id).all()
    
    return render_template('hr/view_applications.html', 
                         applications=applications,
                         jobs=jobs,
                         filters={
                             'job_id': job_filter,
                             'status': status_filter,
                             'sort_by': sort_by
                         })

@hr_bp.route('/candidate/<int:application_id>')
@login_required
def candidate_details(application_id):
    if current_user.user_type != 'hr':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
    
    # Get application with all related data
    application_data = db.session.query(Application, Job, JobSeeker, User)\
        .join(Job).join(JobSeeker).join(User, JobSeeker.user_id == User.id)\
        .filter(Application.id == application_id, Job.hr_id == hr_user.id)\
        .first_or_404()
    
    application, job, jobseeker, user = application_data
    
    # Get detailed AI analysis
    if jobseeker.parsed_data:
        job_data = {
            'title': job.title,
            'description': job.description,
            'required_skills': job.get_required_skills(),
            'experience_required': job.experience_required,
            'location': job.location
        }
        
        try:
            analysis = job_matcher.get_matching_analysis(
                jobseeker.get_parsed_data(),
                job_data
            )
        except Exception as e:
            print(f"Error getting analysis: {e}")
            analysis = {
                'overall_score': application.ai_score or 0,
                'breakdown': {
                    'skills_match': application.skills_match or 0,
                    'experience_match': application.experience_match or 0,
                    'education_match': application.education_match or 0,
                    'location_match': 50
                },
                'matching_skills': [],
                'missing_skills': [],
                'recommendation': 'Unable to generate detailed analysis'
            }
    else:
        analysis = {
            'overall_score': 0,
            'breakdown': {
                'skills_match': 0,
                'experience_match': 0,
                'education_match': 0,
                'location_match': 0
            },
            'matching_skills': [],
            'missing_skills': [],
            'recommendation': 'No resume data available'
        }
    
    return render_template('hr/candidate_details.html',
                         application=application,
                         job=job,
                         jobseeker=jobseeker,
                         candidate=jobseeker,  # Template expects 'candidate'
                         user=user,
                         analysis=analysis)

@hr_bp.route('/application/<int:application_id>/update-status', methods=['POST'])
@login_required
def update_application_status(application_id):
    if current_user.user_type != 'hr':
        return jsonify({'error': 'Access denied'}), 403
    
    hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
    
    # Verify application belongs to HR user's job
    application = db.session.query(Application).join(Job)\
        .filter(Application.id == application_id, Job.hr_id == hr_user.id)\
        .first_or_404()
    
    new_status = request.json.get('status')
    
    if new_status not in ['pending', 'shortlisted', 'rejected', 'selected']:
        return jsonify({'error': 'Invalid status'}), 400
    
    try:
        application.status = new_status
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Application status updated to {new_status}',
            'new_status': new_status
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error updating status: {str(e)}'}), 500

@hr_bp.route('/application/<int:application_id>/generate-report')
@login_required
def generate_application_report(application_id):
    if current_user.user_type != 'hr':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
    
    # Get application data
    application_data = db.session.query(Application, Job, JobSeeker, User)\
        .join(Job).join(JobSeeker).join(User, JobSeeker.user_id == User.id)\
        .filter(Application.id == application_id, Job.hr_id == hr_user.id)\
        .first_or_404()
    
    application, job, jobseeker, user = application_data
    
    try:
        # Prepare report data
        report_data = {
            'candidate_name': jobseeker.name,
            'candidate_email': user.email,
            'job_title': job.title,
            'company': job.company,
            'applied_date': application.applied_at.strftime('%Y-%m-%d'),
            'ai_score': application.ai_score or 0,
            'skills_match': application.skills_match or 0,
            'experience_match': application.experience_match or 0,
            'education_match': application.education_match or 0
        }
        
        # Generate PDF report
        output_dir = Path(current_app.config['UPLOAD_FOLDER']) / 'reports'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"application_report_{application_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = output_dir / filename
        
        pdf_generator.generate_application_report(report_data, output_path)
        
        return send_file(output_path, as_attachment=True)
    
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'error')
        return redirect(url_for('hr.candidate_details', application_id=application_id))

@hr_bp.route('/bulk-update-applications', methods=['POST'])
@login_required
def bulk_update_applications():
    if current_user.user_type != 'hr':
        return jsonify({'error': 'Access denied'}), 403
    
    hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
    
    application_ids = request.json.get('application_ids', [])
    new_status = request.json.get('status')
    
    if not application_ids or new_status not in ['pending', 'shortlisted', 'rejected', 'selected']:
        return jsonify({'error': 'Invalid data'}), 400
    
    try:
        # Update applications that belong to HR user's jobs
        updated_count = db.session.query(Application)\
            .join(Job)\
            .filter(Application.id.in_(application_ids), Job.hr_id == hr_user.id)\
            .update({'status': new_status}, synchronize_session=False)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Updated {updated_count} applications to {new_status}',
            'updated_count': updated_count
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error updating applications: {str(e)}'}), 500

@hr_bp.route('/analytics')
@login_required
def analytics():
    if current_user.user_type != 'hr':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
    
    # Get analytics data
    # Applications by status
    status_counts = db.session.query(
        Application.status,
        db.func.count(Application.id).label('count')
    ).join(Job)\
     .filter(Job.hr_id == hr_user.id)\
     .group_by(Application.status).all()
    
    # Average AI scores by job
    job_scores = db.session.query(
        Job.title,
        db.func.avg(Application.ai_score).label('avg_score'),
        db.func.count(Application.id).label('app_count')
    ).outerjoin(Application)\
     .filter(Job.hr_id == hr_user.id)\
     .group_by(Job.id, Job.title).all()
    
    # Applications over time (last 30 days)
    from datetime import datetime, timedelta
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    
    daily_applications = db.session.query(
        db.func.date(Application.applied_at).label('date'),
        db.func.count(Application.id).label('count')
    ).join(Job)\
     .filter(Job.hr_id == hr_user.id, Application.applied_at >= thirty_days_ago)\
     .group_by(db.func.date(Application.applied_at))\
     .order_by('date').all()
    
    analytics_data = {
        'status_counts': [{'status': s.status, 'count': s.count} for s in status_counts],
        'job_scores': [{'job': j.title, 'avg_score': round(j.avg_score or 0, 1), 'app_count': j.app_count} for j in job_scores],
        'daily_applications': [{'date': str(d.date), 'count': d.count} for d in daily_applications]
    }
    
    return render_template('hr/analytics.html', analytics_data=analytics_data)

@hr_bp.route('/job/<int:job_id>/applications')
@login_required
def view_job_applications(job_id):
    if current_user.user_type != 'hr':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
    job = Job.query.filter_by(id=job_id, hr_id=hr_user.id).first_or_404()
    
    # Get applications for this specific job with proper relationships
    applications = Application.query\
        .options(joinedload(Application.job), joinedload(Application.jobseeker).joinedload(JobSeeker.user))\
        .join(Job).join(JobSeeker).join(User, JobSeeker.user_id == User.id)\
        .filter(Job.id == job_id, Job.hr_id == hr_user.id)\
        .order_by(Application.applied_at.desc()).all()
    
    return render_template('hr/view_applications.html', 
                         applications=applications,
                         job=job,
                         filters={'job_id': job_id})


@hr_bp.route('/job/<int:job_id>/status', methods=['POST'])
@login_required
def update_job_status(job_id):
    if current_user.user_type != 'hr':
        return jsonify({'error': 'Access denied'}), 403
    
    hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
    job = Job.query.filter_by(id=job_id, hr_id=hr_user.id).first_or_404()
    
    try:
        data = request.get_json()
        new_status = data.get('status')
        
        if new_status not in ['active', 'inactive', 'closed']:
            return jsonify({'error': 'Invalid status'}), 400
        
        job.status = new_status
        db.session.commit()
        
        return jsonify({'success': True, 'message': f'Job status updated to {new_status}'})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error updating job status: {str(e)}'}), 500

@hr_bp.route('/job/<int:job_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_job(job_id):
    if current_user.user_type != 'hr':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
    job = Job.query.filter_by(id=job_id, hr_id=hr_user.id).first_or_404()
    
    if request.method == 'POST':
        try:
            # Update job data
            job.title = request.form.get('title', job.title)
            job.description = request.form.get('description', job.description)
            job.location = request.form.get('location', job.location)
            job.experience_required = request.form.get('experience_required', type=int) or job.experience_required
            job.salary_range = request.form.get('salary_range', job.salary_range)
            job.employment_type = request.form.get('employment_type', job.employment_type)
            
            # Update skills
            required_skills = request.form.get('required_skills', '')
            if required_skills:
                skills_list = [skill.strip() for skill in required_skills.split(',') if skill.strip()]
                job.set_required_skills(skills_list)
            
            db.session.commit()
            flash('Job updated successfully!', 'success')
            return redirect(url_for('hr.view_jobs'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating job: {str(e)}', 'error')
    
    return render_template('hr/edit_job.html', job=job, hr_user=hr_user)

@hr_bp.route('/job/<int:job_id>/delete', methods=['POST'])
@login_required
def delete_job(job_id):
    if current_user.user_type != 'hr':
        return jsonify({'error': 'Access denied'}), 403
    
    hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
    job = Job.query.filter_by(id=job_id, hr_id=hr_user.id).first_or_404()
    
    try:
        # Delete associated applications first
        Application.query.filter_by(job_id=job_id).delete()
        
        # Delete the job
        db.session.delete(job)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Job deleted successfully'})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error deleting job: {str(e)}'}), 500

@hr_bp.route('/download-resume/<int:jobseeker_id>')
@login_required
def download_resume(jobseeker_id):
    if current_user.user_type != 'hr':
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    jobseeker = JobSeeker.query.get_or_404(jobseeker_id)
    
    # For now, just show a message (would need actual file handling)
    flash('Resume download feature - coming soon!', 'info')
    return redirect(request.referrer or url_for('hr.view_applications'))

@hr_bp.route('/recalculate-ai-scores', methods=['POST'])
@login_required
def recalculate_ai_scores():
    """Recalculate AI scores for all applications"""
    if current_user.user_type != 'hr':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        from ml_models.ai_resume_analyzer import AIResumeAnalyzer
        ai_analyzer = AIResumeAnalyzer()
        
        hr_user = HRUser.query.filter_by(user_id=current_user.id).first()
        
        # Get all applications for this HR's jobs
        applications = Application.query\
            .join(Job).join(JobSeeker)\
            .filter(Job.hr_id == hr_user.id)\
            .all()
        
        updated_count = 0
        
        for application in applications:
            jobseeker = application.jobseeker
            job = application.job
            
            # Skip if no resume data
            if not jobseeker.parsed_data:
                continue
            
            # Get resume text (simplified - in real scenario you'd extract from file)
            resume_data = jobseeker.get_parsed_data()
            resume_text = f"""
            Name: {jobseeker.name or 'Unknown'}
            Skills: {', '.join(jobseeker.get_skills())}
            Experience: {jobseeker.experience_years or 0} years
            Education: {jobseeker.education or 'Not specified'}
            Location: {jobseeker.location or 'Not specified'}
            """
            
            # Prepare job requirements
            job_requirements = {
                'required_skills': job.get_required_skills(),
                'experience_required': job.experience_required or 0,
                'education_required': '',
                'job_title': job.title,
                'job_description': job.description
            }
            
            # Run AI analysis
            ai_analysis = ai_analyzer.analyze_resume(resume_text, job_requirements)
            
            # Update application scores
            application.ai_score = ai_analysis['overall_score']
            application.skills_match = ai_analysis['skills_analysis']['score']
            application.experience_match = ai_analysis['experience_analysis']['score']
            application.education_match = ai_analysis['education_analysis']['score']
            
            updated_count += 1
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Successfully recalculated AI scores for {updated_count} applications'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error recalculating scores: {str(e)}'}), 500

