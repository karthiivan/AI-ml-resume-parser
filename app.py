#!/usr/bin/env python3
"""
Main Flask application for AI Resume Parser
"""

import os
from pathlib import Path
from flask import Flask, render_template, redirect, url_for
from flask_login import LoginManager, current_user
from flask_bcrypt import Bcrypt

# Import models and config
from models import db, User
from config import config

# Import blueprints
from routes.main import main_bp
from routes.auth import auth_bp
from routes.jobseeker import jobseeker_bp
from routes.hr import hr_bp

def create_app(config_name='development'):
    """Create and configure Flask application"""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    bcrypt = Bcrypt(app)
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Create upload directories
    upload_dir = Path(app.config['UPLOAD_FOLDER'])
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / 'generated').mkdir(exist_ok=True)
    (upload_dir / 'reports').mkdir(exist_ok=True)
    
    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(jobseeker_bp)
    app.register_blueprint(hr_bp)
    
    # Routes are now handled by blueprints
    
    # Template filters
    @app.template_filter('score_class')
    def score_class_filter(score):
        if score >= 80:
            return 'high'
        elif score >= 60:
            return 'medium'
        else:
            return 'low'
    
    @app.template_filter('status_class')
    def status_class_filter(status):
        status_map = {
            'pending': 'warning',
            'shortlisted': 'success',
            'rejected': 'danger',
            'hired': 'success'
        }
        return status_map.get(status, 'secondary')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(403)
    def forbidden_error(error):
        return render_template('errors/403.html'), 403
    
    # Template filters
    @app.template_filter('datetime')
    def datetime_filter(value, format='%Y-%m-%d %H:%M'):
        """Format datetime for templates"""
        if value is None:
            return ""
        return value.strftime(format)
    
    @app.template_filter('score_color')
    def score_color_filter(score):
        """Get color class for AI score"""
        if score >= 80:
            return 'success'
        elif score >= 60:
            return 'warning'
        elif score >= 40:
            return 'info'
        else:
            return 'danger'
    
    @app.template_filter('status_color')
    def status_color_filter(status):
        """Get color class for application status"""
        colors = {
            'pending': 'warning',
            'shortlisted': 'success',
            'rejected': 'danger',
            'selected': 'primary'
        }
        return colors.get(status, 'secondary')
    
    # Context processors
    @app.context_processor
    def inject_user():
        """Inject current user into all templates"""
        return dict(current_user=current_user)
    
    return app

def init_ml_models():
    """Initialize ML models on startup"""
    try:
        print("Initializing ML models...")
        
        # Import ML modules to trigger model loading
        from ml_models.ner_parser import NERParser
        from ml_models.job_matcher import JobMatcher
        
        # Initialize models
        ner_parser = NERParser()
        job_matcher = JobMatcher()
        
        print("ML models initialized successfully!")
        return True
    
    except Exception as e:
        print(f"Warning: Could not initialize ML models: {e}")
        print("The application will still work with basic functionality.")
        return False

if __name__ == '__main__':
    # Create application
    app = create_app()
    
    # Initialize ML models
    init_ml_models()
    
    # Run application
    print("Starting AI Resume Parser application...")
    print("Visit http://localhost:5000 to access the application")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
