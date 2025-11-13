from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app
from flask_login import login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from models import db, User, JobSeeker, HRUser

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user_type = request.form.get('user_type')
        
        if not email or not password or not user_type:
            flash('Please fill in all fields', 'error')
            return render_template('auth/login.html')
        
        # Find user
        user = User.query.filter_by(email=email, user_type=user_type).first()
        
        bcrypt = current_app.extensions.get('bcrypt')
        if user and bcrypt and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            session['user_type'] = user.user_type
            
            # Redirect based on user type
            if user.user_type == 'jobseeker':
                return redirect(url_for('jobseeker.dashboard'))
            else:
                return redirect(url_for('hr.dashboard'))
        else:
            flash('Invalid email, password, or user type', 'error')
    
    return render_template('auth/login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        user_type = request.form.get('user_type')
        
        # Validation
        if not all([email, password, confirm_password, user_type]):
            flash('Please fill in all fields', 'error')
            return render_template('auth/register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('auth/register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('auth/register.html')
        
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered', 'error')
            return render_template('auth/register.html')
        
        try:
            # Create user
            bcrypt = current_app.extensions.get('bcrypt')
            if not bcrypt:
                flash('Authentication system error', 'error')
                return render_template('auth/register.html')
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(
                email=email,
                password=hashed_password,
                user_type=user_type
            )
            db.session.add(user)
            db.session.flush()  # Get user ID
            
            # Create profile based on user type
            if user_type == 'jobseeker':
                name = request.form.get('name')
                phone = request.form.get('phone')
                location = request.form.get('location')
                
                profile = JobSeeker(
                    user_id=user.id,
                    name=name,
                    phone=phone,
                    location=location
                )
                db.session.add(profile)
            
            elif user_type == 'hr':
                company_name = request.form.get('company_name')
                contact_name = request.form.get('contact_name')
                phone = request.form.get('phone')
                
                profile = HRUser(
                    user_id=user.id,
                    company_name=company_name,
                    contact_name=contact_name,
                    phone=phone
                )
                db.session.add(profile)
            
            db.session.commit()
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))
        
        except Exception as e:
            db.session.rollback()
            flash('Registration failed. Please try again.', 'error')
            print(f"Registration error: {e}")
    
    return render_template('auth/register.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('index'))
