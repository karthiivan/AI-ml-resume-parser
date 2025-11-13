from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from datetime import datetime
import json

class PDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for the PDF"""
        # Header style
        if 'CustomHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomHeader',
                parent=self.styles['Heading1'],
                fontSize=18,
                spaceAfter=12,
                textColor=colors.HexColor('#2563eb'),
                alignment=TA_CENTER
            ))
        
        # Section header style
        if 'SectionHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceAfter=6,
                spaceBefore=12,
                textColor=colors.HexColor('#1f2937'),
                borderWidth=1,
                borderColor=colors.HexColor('#e5e7eb'),
                borderPadding=4
            ))
        
        # Contact info style
        if 'ContactInfo' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='ContactInfo',
                parent=self.styles['Normal'],
                fontSize=10,
                alignment=TA_CENTER,
                spaceAfter=12
            ))
        
        # Body text style
        if 'BodyText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontSize=11,
                alignment=TA_JUSTIFY,
                spaceAfter=6
            ))
    
    def generate_resume_pdf(self, parsed_data, output_path, template='modern'):
        """Generate a structured resume PDF from parsed data"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Header with name
        if 'name' in parsed_data:
            story.append(Paragraph(parsed_data['name'], self.styles['CustomHeader']))
        
        # Contact information
        contact_info = self._build_contact_info(parsed_data)
        if contact_info:
            story.append(Paragraph(contact_info, self.styles['ContactInfo']))
        
        story.append(Spacer(1, 12))
        
        # Professional Summary
        if 'summary' in parsed_data:
            story.append(Paragraph("Professional Summary", self.styles['SectionHeader']))
            story.append(Paragraph(parsed_data['summary'], self.styles['BodyText']))
            story.append(Spacer(1, 12))
        
        # Skills
        if 'skills' in parsed_data and parsed_data['skills']:
            story.append(Paragraph("Skills", self.styles['SectionHeader']))
            skills_text = " • ".join(parsed_data['skills'])
            story.append(Paragraph(skills_text, self.styles['BodyText']))
            story.append(Spacer(1, 12))
        
        # Experience
        if 'experience' in parsed_data and parsed_data['experience']:
            story.append(Paragraph("Professional Experience", self.styles['SectionHeader']))
            for exp in parsed_data['experience']:
                story.extend(self._build_experience_section(exp))
            story.append(Spacer(1, 12))
        
        # Education
        if 'education' in parsed_data and parsed_data['education']:
            story.append(Paragraph("Education", self.styles['SectionHeader']))
            for edu in parsed_data['education']:
                story.extend(self._build_education_section(edu))
            story.append(Spacer(1, 12))
        
        # Certifications
        if 'certifications' in parsed_data and parsed_data['certifications']:
            story.append(Paragraph("Certifications", self.styles['SectionHeader']))
            for cert in parsed_data['certifications']:
                story.append(Paragraph(f"• {cert}", self.styles['BodyText']))
            story.append(Spacer(1, 12))
        
        # Projects
        if 'projects' in parsed_data and parsed_data['projects']:
            story.append(Paragraph("Projects", self.styles['SectionHeader']))
            for project in parsed_data['projects']:
                story.extend(self._build_project_section(project))
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def _build_contact_info(self, parsed_data):
        """Build contact information string"""
        contact_parts = []
        
        if 'email' in parsed_data:
            contact_parts.append(parsed_data['email'])
        
        if 'phone' in parsed_data:
            contact_parts.append(parsed_data['phone'])
        
        if 'location' in parsed_data:
            contact_parts.append(parsed_data['location'])
        
        if 'linkedin' in parsed_data:
            contact_parts.append(f"LinkedIn: {parsed_data['linkedin']}")
        
        if 'github' in parsed_data:
            contact_parts.append(f"GitHub: {parsed_data['github']}")
        
        return " | ".join(contact_parts)
    
    def _build_experience_section(self, experience):
        """Build experience section"""
        elements = []
        
        # Job title and company
        if isinstance(experience, dict):
            title = experience.get('title', 'Position')
            company = experience.get('company', 'Company')
            duration = experience.get('duration', '')
            
            header = f"<b>{title}</b> - {company}"
            if duration:
                header += f" ({duration})"
            
            elements.append(Paragraph(header, self.styles['BodyText']))
            
            if 'description' in experience:
                elements.append(Paragraph(experience['description'], self.styles['BodyText']))
        else:
            # Handle string format
            elements.append(Paragraph(str(experience), self.styles['BodyText']))
        
        elements.append(Spacer(1, 6))
        return elements
    
    def _build_education_section(self, education):
        """Build education section"""
        elements = []
        
        if isinstance(education, dict):
            degree = education.get('degree', 'Degree')
            institution = education.get('institution', 'Institution')
            year = education.get('year', '')
            
            header = f"<b>{degree}</b> - {institution}"
            if year:
                header += f" ({year})"
            
            elements.append(Paragraph(header, self.styles['BodyText']))
        else:
            elements.append(Paragraph(str(education), self.styles['BodyText']))
        
        elements.append(Spacer(1, 6))
        return elements
    
    def _build_project_section(self, project):
        """Build project section"""
        elements = []
        
        if isinstance(project, dict):
            name = project.get('name', 'Project')
            description = project.get('description', '')
            technologies = project.get('technologies', [])
            
            header = f"<b>{name}</b>"
            elements.append(Paragraph(header, self.styles['BodyText']))
            
            if description:
                elements.append(Paragraph(description, self.styles['BodyText']))
            
            if technologies:
                tech_text = f"Technologies: {', '.join(technologies)}"
                elements.append(Paragraph(tech_text, self.styles['BodyText']))
        else:
            elements.append(Paragraph(str(project), self.styles['BodyText']))
        
        elements.append(Spacer(1, 6))
        return elements
    
    def generate_application_report(self, application_data, output_path):
        """Generate application analysis report PDF"""
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Title
        story.append(Paragraph("Application Analysis Report", self.styles['CustomHeader']))
        story.append(Spacer(1, 12))
        
        # Candidate info
        story.append(Paragraph("Candidate Information", self.styles['SectionHeader']))
        candidate_info = [
            ['Name:', application_data.get('candidate_name', 'N/A')],
            ['Email:', application_data.get('candidate_email', 'N/A')],
            ['Position:', application_data.get('job_title', 'N/A')],
            ['Company:', application_data.get('company', 'N/A')],
            ['Application Date:', application_data.get('applied_date', 'N/A')]
        ]
        
        table = Table(candidate_info, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        
        # AI Analysis
        story.append(Paragraph("AI Analysis Results", self.styles['SectionHeader']))
        
        ai_score = application_data.get('ai_score', 0)
        skills_match = application_data.get('skills_match', 0)
        experience_match = application_data.get('experience_match', 0)
        
        analysis_info = [
            ['Overall AI Score:', f"{ai_score:.1f}%"],
            ['Skills Match:', f"{skills_match:.1f}%"],
            ['Experience Match:', f"{experience_match:.1f}%"],
            ['Recommendation:', self._get_recommendation(ai_score)]
        ]
        
        table = Table(analysis_info, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(table)
        
        doc.build(story)
        return output_path
    
    def _get_recommendation(self, score):
        """Get recommendation based on AI score"""
        if score >= 80:
            return "Highly Recommended - Shortlist"
        elif score >= 50:
            return "Good Fit - Consider Interview"
        else:
            return "Below Requirements - Consider Rejecting"
