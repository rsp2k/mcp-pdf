#!/usr/bin/env python3
"""Create a test PDF for testing the MCP PDF Tools"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def create_test_pdf(filename="test_document.pdf"):
    """Create a test PDF with text, tables, and metadata"""
    
    # Create the PDF
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph("MCP PDF Tools Test Document", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.3*inch))
    
    # Introduction
    intro = Paragraph(
        "This is a test document created to demonstrate the capabilities of the MCP PDF Tools server. "
        "It contains various elements including text, tables, and metadata to test different extraction features.",
        styles['Normal']
    )
    story.append(intro)
    story.append(Spacer(1, 0.2*inch))
    
    # Section 1
    section1 = Paragraph("1. Text Extraction Test", styles['Heading2'])
    story.append(section1)
    story.append(Spacer(1, 0.1*inch))
    
    text1 = Paragraph(
        "This section contains regular paragraph text that should be easily extractable using any of the "
        "text extraction methods (PyMuPDF, pdfplumber, or pypdf). The text includes various formatting "
        "and should maintain its structure when extracted with layout preservation enabled.",
        styles['Normal']
    )
    story.append(text1)
    story.append(Spacer(1, 0.2*inch))
    
    # Section 2 - Table
    section2 = Paragraph("2. Table Extraction Test", styles['Heading2'])
    story.append(section2)
    story.append(Spacer(1, 0.1*inch))
    
    # Create a table
    data = [
        ['Product', 'Price', 'Quantity', 'Total'],
        ['Widget A', '$10.00', '5', '$50.00'],
        ['Widget B', '$15.00', '3', '$45.00'],
        ['Widget C', '$20.00', '2', '$40.00'],
        ['Total', '', '', '$135.00']
    ]
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.2*inch))
    
    # Section 3
    section3 = Paragraph("3. Document Structure Test", styles['Heading2'])
    story.append(section3)
    story.append(Spacer(1, 0.1*inch))
    
    text3 = Paragraph(
        "This document has a clear structure with numbered sections and headings. "
        "The document structure extraction should identify these sections and create "
        "an outline or table of contents.",
        styles['Normal']
    )
    story.append(text3)
    story.append(Spacer(1, 0.2*inch))
    
    # Add metadata
    doc.title = "MCP PDF Tools Test Document"
    doc.author = "MCP PDF Tools Tester"
    doc.subject = "Testing PDF Processing"
    doc.keywords = ["test", "pdf", "mcp", "extraction"]
    
    # Build the PDF
    doc.build(story)
    print(f"✅ Created test PDF: {filename}")

if __name__ == "__main__":
    create_test_pdf()
