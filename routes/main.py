#!/usr/bin/env python3
"""
Main routes blueprint for AI Resume Parser
"""

from flask import Blueprint, render_template, redirect, url_for
from flask_login import current_user

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Landing page"""
    if current_user.is_authenticated:
        if current_user.user_type == 'jobseeker':
            return redirect(url_for('jobseeker.dashboard'))
        elif current_user.user_type == 'hr':
            return redirect(url_for('hr.dashboard'))
    
    return render_template('index.html')

@main_bp.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@main_bp.route('/features')
def features():
    """Features page"""
    return render_template('features.html')

@main_bp.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

@main_bp.route('/favicon.ico')
def favicon():
    return '', 204  # Return empty response with "No Content" status
