/* AI Resume Parser - Form Handling JavaScript */

// Form validation and enhancement
document.addEventListener('DOMContentLoaded', function() {
    initializeForms();
});

function initializeForms() {
    // Initialize all form enhancements
    initializeFormValidation();
    initializeSkillsInput();
    initializeFileUpload();
    initializeFormSubmission();
    initializePasswordStrength();
    initializeAutoComplete();
}

// Form Validation
function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        // Real-time validation
        const inputs = form.querySelectorAll('input, textarea, select');
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                validateField(this);
            });
            
            input.addEventListener('input', function() {
                if (this.classList.contains('is-invalid')) {
                    validateField(this);
                }
            });
        });
        
        // Form submission validation
        form.addEventListener('submit', function(e) {
            if (!validateForm(this)) {
                e.preventDefault();
                e.stopPropagation();
            }
        });
    });
}

function validateField(field) {
    const value = field.value.trim();
    const type = field.type;
    const required = field.hasAttribute('required');
    let isValid = true;
    let message = '';
    
    // Clear previous validation
    clearFieldValidation(field);
    
    // Required field validation
    if (required && !value) {
        isValid = false;
        message = 'This field is required';
    }
    
    // Email validation
    else if (type === 'email' && value) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(value)) {
            isValid = false;
            message = 'Please enter a valid email address';
        }
    }
    
    // Phone validation
    else if (field.name === 'phone' && value) {
        const phoneRegex = /^[\+]?[1-9][\d]{0,15}$/;
        if (!phoneRegex.test(value.replace(/[\s\-\(\)]/g, ''))) {
            isValid = false;
            message = 'Please enter a valid phone number';
        }
    }
    
    // Password validation
    else if (type === 'password' && value) {
        if (value.length < 6) {
            isValid = false;
            message = 'Password must be at least 6 characters long';
        }
    }
    
    // URL validation
    else if (field.name === 'linkedin' || field.name === 'github') {
        if (value && !isValidURL(value)) {
            isValid = false;
            message = 'Please enter a valid URL';
        }
    }
    
    // Apply validation result
    if (isValid) {
        field.classList.remove('is-invalid');
        field.classList.add('is-valid');
    } else {
        field.classList.remove('is-valid');
        field.classList.add('is-invalid');
        showFieldError(field, message);
    }
    
    return isValid;
}

function validateForm(form) {
    const inputs = form.querySelectorAll('input, textarea, select');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!validateField(input)) {
            isValid = false;
        }
    });
    
    // Custom form validations
    const formId = form.id;
    
    if (formId === 'registerForm') {
        const password = form.querySelector('#password');
        const confirmPassword = form.querySelector('#confirm_password');
        
        if (password && confirmPassword && password.value !== confirmPassword.value) {
            showFieldError(confirmPassword, 'Passwords do not match');
            confirmPassword.classList.add('is-invalid');
            isValid = false;
        }
    }
    
    if (!isValid) {
        showToast('Please correct the errors in the form', 'error');
    }
    
    return isValid;
}

function clearFieldValidation(field) {
    field.classList.remove('is-valid', 'is-invalid');
    const feedback = field.parentNode.querySelector('.invalid-feedback');
    if (feedback) {
        feedback.remove();
    }
}

function showFieldError(field, message) {
    let feedback = field.parentNode.querySelector('.invalid-feedback');
    if (!feedback) {
        feedback = document.createElement('div');
        feedback.className = 'invalid-feedback';
        field.parentNode.appendChild(feedback);
    }
    feedback.textContent = message;
}

function isValidURL(string) {
    try {
        new URL(string);
        return true;
    } catch (_) {
        return false;
    }
}

// Skills Input Enhancement
function initializeSkillsInput() {
    const skillsInputs = document.querySelectorAll('input[name="skills"], textarea[name="skills"]');
    
    skillsInputs.forEach(input => {
        const container = createSkillsContainer(input);
        input.style.display = 'none';
        
        // Initialize with existing skills
        const existingSkills = input.value.split(',').filter(skill => skill.trim());
        existingSkills.forEach(skill => {
            addSkillTag(container, skill.trim());
        });
        
        setupSkillsInput(container, input);
    });
}

function createSkillsContainer(input) {
    const container = document.createElement('div');
    container.className = 'skills-input-container';
    container.innerHTML = `
        <div class="skills-tags"></div>
        <input type="text" class="form-control skills-input" placeholder="Type a skill and press Enter">
        <div class="skills-suggestions"></div>
    `;
    
    input.parentNode.insertBefore(container, input);
    return container;
}

function setupSkillsInput(container, hiddenInput) {
    const skillsInput = container.querySelector('.skills-input');
    const tagsContainer = container.querySelector('.skills-tags');
    const suggestionsContainer = container.querySelector('.skills-suggestions');
    
    // Common skills for suggestions
    const commonSkills = [
        'JavaScript', 'Python', 'Java', 'React', 'Node.js', 'HTML', 'CSS', 'SQL',
        'Git', 'AWS', 'Docker', 'Kubernetes', 'MongoDB', 'PostgreSQL', 'Redis',
        'Angular', 'Vue.js', 'TypeScript', 'C++', 'C#', 'PHP', 'Ruby', 'Go',
        'Machine Learning', 'Data Analysis', 'Project Management', 'Agile', 'Scrum'
    ];
    
    skillsInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ',') {
            e.preventDefault();
            const skill = this.value.trim();
            if (skill) {
                addSkillTag(container, skill);
                this.value = '';
                updateHiddenInput(container, hiddenInput);
            }
        }
    });
    
    skillsInput.addEventListener('input', function() {
        const query = this.value.toLowerCase();
        if (query.length > 0) {
            const suggestions = commonSkills.filter(skill => 
                skill.toLowerCase().includes(query) && 
                !isSkillAlreadyAdded(container, skill)
            );
            showSuggestions(suggestionsContainer, suggestions, container, hiddenInput);
        } else {
            suggestionsContainer.innerHTML = '';
        }
    });
    
    // Hide suggestions when clicking outside
    document.addEventListener('click', function(e) {
        if (!container.contains(e.target)) {
            suggestionsContainer.innerHTML = '';
        }
    });
}

function addSkillTag(container, skill) {
    if (isSkillAlreadyAdded(container, skill)) return;
    
    const tagsContainer = container.querySelector('.skills-tags');
    const tag = document.createElement('span');
    tag.className = 'skill-tag';
    tag.innerHTML = `
        ${skill}
        <button type="button" class="skill-remove" onclick="removeSkillTag(this)">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    tagsContainer.appendChild(tag);
}

function removeSkillTag(button) {
    const tag = button.parentElement;
    const container = tag.closest('.skills-input-container');
    const hiddenInput = container.nextElementSibling;
    
    tag.remove();
    updateHiddenInput(container, hiddenInput);
}

function isSkillAlreadyAdded(container, skill) {
    const existingTags = container.querySelectorAll('.skill-tag');
    return Array.from(existingTags).some(tag => 
        tag.textContent.trim().toLowerCase() === skill.toLowerCase()
    );
}

function showSuggestions(container, suggestions, skillsContainer, hiddenInput) {
    container.innerHTML = '';
    
    if (suggestions.length === 0) return;
    
    suggestions.slice(0, 5).forEach(skill => {
        const suggestion = document.createElement('div');
        suggestion.className = 'skill-suggestion';
        suggestion.textContent = skill;
        suggestion.addEventListener('click', function() {
            addSkillTag(skillsContainer, skill);
            skillsContainer.querySelector('.skills-input').value = '';
            container.innerHTML = '';
            updateHiddenInput(skillsContainer, hiddenInput);
        });
        container.appendChild(suggestion);
    });
}

function updateHiddenInput(container, hiddenInput) {
    const tags = container.querySelectorAll('.skill-tag');
    const skills = Array.from(tags).map(tag => tag.textContent.trim());
    hiddenInput.value = skills.join(', ');
}

// File Upload Enhancement
function initializeFileUpload() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        enhanceFileInput(input);
    });
}

function enhanceFileInput(input) {
    const wrapper = document.createElement('div');
    wrapper.className = 'file-upload-wrapper';
    
    const dropZone = document.createElement('div');
    dropZone.className = 'drop-zone';
    dropZone.innerHTML = `
        <div class="drop-zone-content">
            <i class="fas fa-cloud-upload-alt drop-icon"></i>
            <h5>Drop your file here</h5>
            <p class="text-muted">or click to browse</p>
            <div class="file-info"></div>
        </div>
    `;
    
    input.parentNode.insertBefore(wrapper, input);
    wrapper.appendChild(dropZone);
    wrapper.appendChild(input);
    
    input.style.display = 'none';
    
    // Setup drag and drop
    setupDropZone(dropZone, input);
}

// Form Submission Enhancement
function initializeFormSubmission() {
    const forms = document.querySelectorAll('form[data-ajax]');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            submitFormAjax(this);
        });
    });
}

function submitFormAjax(form) {
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    const formData = new FormData(form);
    
    // Show loading state
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
    submitBtn.disabled = true;
    
    fetch(form.action || window.location.href, {
        method: form.method || 'POST',
        body: formData
    })
    .then(response => {
        if (response.redirected) {
            window.location.href = response.url;
            return;
        }
        return response.json();
    })
    .then(data => {
        if (data && data.success) {
            showToast(data.message || 'Success!', 'success');
            if (data.redirect) {
                setTimeout(() => {
                    window.location.href = data.redirect;
                }, 1500);
            }
        } else if (data && data.error) {
            showToast(data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Form submission error:', error);
        showToast('An error occurred. Please try again.', 'error');
    })
    .finally(() => {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    });
}

// Password Strength Indicator
function initializePasswordStrength() {
    const passwordInputs = document.querySelectorAll('input[type="password"][data-strength]');
    
    passwordInputs.forEach(input => {
        const indicator = createPasswordStrengthIndicator();
        input.parentNode.appendChild(indicator);
        
        input.addEventListener('input', function() {
            updatePasswordStrength(this.value, indicator);
        });
    });
}

function createPasswordStrengthIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'password-strength-indicator';
    indicator.innerHTML = `
        <div class="strength-bar">
            <div class="strength-fill"></div>
        </div>
        <small class="strength-text">Password strength</small>
    `;
    return indicator;
}

function updatePasswordStrength(password, indicator) {
    const fill = indicator.querySelector('.strength-fill');
    const text = indicator.querySelector('.strength-text');
    
    let strength = 0;
    let label = 'Weak';
    let color = '#ef4444';
    
    // Calculate strength
    if (password.length >= 8) strength += 25;
    if (/[a-z]/.test(password)) strength += 25;
    if (/[A-Z]/.test(password)) strength += 25;
    if (/[0-9]/.test(password)) strength += 25;
    
    // Determine label and color
    if (strength >= 75) {
        label = 'Strong';
        color = '#10b981';
    } else if (strength >= 50) {
        label = 'Medium';
        color = '#f59e0b';
    } else if (strength >= 25) {
        label = 'Fair';
        color = '#f97316';
    }
    
    // Update UI
    fill.style.width = strength + '%';
    fill.style.backgroundColor = color;
    text.textContent = `Password strength: ${label}`;
    text.style.color = color;
}

// Auto-complete Enhancement
function initializeAutoComplete() {
    const autoCompleteInputs = document.querySelectorAll('input[data-autocomplete]');
    
    autoCompleteInputs.forEach(input => {
        const source = input.dataset.autocomplete;
        setupAutoComplete(input, source);
    });
}

function setupAutoComplete(input, source) {
    const container = document.createElement('div');
    container.className = 'autocomplete-container';
    
    const suggestions = document.createElement('div');
    suggestions.className = 'autocomplete-suggestions';
    
    input.parentNode.insertBefore(container, input);
    container.appendChild(input);
    container.appendChild(suggestions);
    
    let debounceTimer;
    
    input.addEventListener('input', function() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            fetchSuggestions(this.value, source, suggestions, input);
        }, 300);
    });
    
    // Hide suggestions when clicking outside
    document.addEventListener('click', function(e) {
        if (!container.contains(e.target)) {
            suggestions.innerHTML = '';
        }
    });
}

function fetchSuggestions(query, source, container, input) {
    if (query.length < 2) {
        container.innerHTML = '';
        return;
    }
    
    // This would typically fetch from an API
    // For now, we'll use static data based on the source
    const data = getStaticSuggestions(source, query);
    displaySuggestions(data, container, input);
}

function getStaticSuggestions(source, query) {
    const data = {
        locations: [
            'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX',
            'Phoenix, AZ', 'Philadelphia, PA', 'San Antonio, TX', 'San Diego, CA',
            'Dallas, TX', 'San Jose, CA', 'Austin, TX', 'Jacksonville, FL'
        ],
        companies: [
            'Google', 'Microsoft', 'Apple', 'Amazon', 'Facebook', 'Netflix',
            'Tesla', 'Uber', 'Airbnb', 'Spotify', 'Twitter', 'LinkedIn'
        ]
    };
    
    const sourceData = data[source] || [];
    return sourceData.filter(item => 
        item.toLowerCase().includes(query.toLowerCase())
    ).slice(0, 5);
}

function displaySuggestions(suggestions, container, input) {
    container.innerHTML = '';
    
    suggestions.forEach(suggestion => {
        const item = document.createElement('div');
        item.className = 'autocomplete-item';
        item.textContent = suggestion;
        item.addEventListener('click', function() {
            input.value = suggestion;
            container.innerHTML = '';
            input.focus();
        });
        container.appendChild(item);
    });
}

// Export functions for global use
window.validateForm = validateForm;
window.removeSkillTag = removeSkillTag;
window.submitFormAjax = submitFormAjax;
