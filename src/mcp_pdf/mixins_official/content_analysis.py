"""
Content Analysis Mixin - PDF content classification, summarization, and layout analysis
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import re
from collections import Counter

# PDF processing libraries
import fitz  # PyMuPDF

# Official FastMCP mixin
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, sanitize_error_message
from .utils import parse_pages_parameter

logger = logging.getLogger(__name__)


class ContentAnalysisMixin(MCPMixin):
    """
    Handles PDF content analysis including classification, summarization, and layout analysis.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()
        self.max_file_size = 100 * 1024 * 1024  # 100MB

    @mcp_tool(
        name="classify_content",
        description="Classify and analyze PDF content type and structure"
    )
    async def classify_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        Classify PDF content type and analyze document structure.

        Args:
            pdf_path: Path to PDF file or HTTPS URL

        Returns:
            Dictionary containing content classification results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            # Extract text from sample pages for analysis
            sample_size = min(10, len(doc))
            full_text = ""
            total_words = 0
            total_sentences = 0

            for page_num in range(sample_size):
                page_text = doc[page_num].get_text()
                full_text += page_text + " "
                total_words += len(page_text.split())

            # Count sentences (basic estimation)
            sentences = re.split(r'[.!?]+', full_text)
            total_sentences = len([s for s in sentences if s.strip()])

            # Analyze document structure
            toc = doc.get_toc()
            has_bookmarks = len(toc) > 0
            bookmark_levels = max([item[0] for item in toc]) if toc else 0

            # Content type classification
            content_indicators = {
                "academic": ["abstract", "introduction", "methodology", "conclusion", "references", "bibliography"],
                "business": ["executive summary", "proposal", "budget", "quarterly", "revenue", "profit"],
                "legal": ["whereas", "hereby", "pursuant", "plaintiff", "defendant", "contract", "agreement"],
                "technical": ["algorithm", "implementation", "system", "configuration", "specification", "api"],
                "financial": ["financial", "income", "expense", "balance sheet", "cash flow", "investment"],
                "medical": ["patient", "diagnosis", "treatment", "symptoms", "medical", "clinical"],
                "educational": ["course", "curriculum", "lesson", "assignment", "grade", "student"]
            }

            content_scores = {}
            text_lower = full_text.lower()

            for category, keywords in content_indicators.items():
                score = sum(text_lower.count(keyword) for keyword in keywords)
                content_scores[category] = score

            # Determine primary content type
            if content_scores:
                primary_type = max(content_scores, key=content_scores.get)
                confidence = content_scores[primary_type] / max(sum(content_scores.values()), 1)
            else:
                primary_type = "general"
                confidence = 0.5

            # Analyze text characteristics
            avg_words_per_page = total_words / sample_size if sample_size > 0 else 0
            avg_sentences_per_page = total_sentences / sample_size if sample_size > 0 else 0

            # Document complexity analysis
            unique_words = len(set(full_text.lower().split()))
            vocabulary_diversity = unique_words / max(total_words, 1)

            # Reading level estimation (simplified)
            if avg_sentences_per_page > 0:
                avg_words_per_sentence = total_words / total_sentences
                # Simplified readability score
                readability_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * (total_sentences / max(total_words, 1)))
                readability_score = max(0, min(100, readability_score))
            else:
                readability_score = 50

            # Determine reading level
            if readability_score >= 90:
                reading_level = "Elementary"
            elif readability_score >= 70:
                reading_level = "Middle School"
            elif readability_score >= 50:
                reading_level = "High School"
            elif readability_score >= 30:
                reading_level = "College"
            else:
                reading_level = "Graduate"

            # Check for multimedia content
            total_images = sum(len(doc[i].get_images()) for i in range(sample_size))
            total_links = sum(len(doc[i].get_links()) for i in range(sample_size))

            # Estimate for full document
            estimated_total_images = int(total_images * len(doc) / sample_size) if sample_size > 0 else 0
            estimated_total_links = int(total_links * len(doc) / sample_size) if sample_size > 0 else 0

            doc.close()

            return {
                "success": True,
                "classification": {
                    "primary_type": primary_type,
                    "confidence": round(confidence, 2),
                    "secondary_types": sorted(content_scores.items(), key=lambda x: x[1], reverse=True)[1:4]
                },
                "content_analysis": {
                    "total_pages": len(doc),
                    "estimated_word_count": int(total_words * len(doc) / sample_size),
                    "avg_words_per_page": round(avg_words_per_page, 1),
                    "vocabulary_diversity": round(vocabulary_diversity, 2),
                    "reading_level": reading_level,
                    "readability_score": round(readability_score, 1)
                },
                "document_structure": {
                    "has_bookmarks": has_bookmarks,
                    "bookmark_levels": bookmark_levels,
                    "estimated_sections": len([item for item in toc if item[0] <= 2]),
                    "is_structured": has_bookmarks and bookmark_levels > 1
                },
                "multimedia_content": {
                    "estimated_images": estimated_total_images,
                    "estimated_links": estimated_total_links,
                    "is_multimedia_rich": estimated_total_images > 10 or estimated_total_links > 5
                },
                "content_characteristics": {
                    "is_text_heavy": avg_words_per_page > 500,
                    "is_technical": content_scores.get("technical", 0) > 5,
                    "has_formal_language": primary_type in ["legal", "academic", "technical"],
                    "complexity_level": "high" if vocabulary_diversity > 0.7 else "medium" if vocabulary_diversity > 0.4 else "low"
                },
                "file_info": {
                    "path": str(path),
                    "pages_analyzed": sample_size
                },
                "analysis_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Content classification failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "analysis_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="summarize_content",
        description="Generate summary and key insights from PDF content"
    )
    async def summarize_content(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        summary_length: str = "medium"
    ) -> Dict[str, Any]:
        """
        Generate summary and extract key insights from PDF content.

        Args:
            pdf_path: Path to PDF file or HTTPS URL
            pages: Page numbers to summarize (comma-separated, 1-based), None for all
            summary_length: Summary length ("short", "medium", "long")

        Returns:
            Dictionary containing content summary and insights
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            # Parse pages parameter
            parsed_pages = parse_pages_parameter(pages)
            page_numbers = parsed_pages if parsed_pages else list(range(len(doc)))
            page_numbers = [p for p in page_numbers if 0 <= p < len(doc)]

            # If parsing failed but pages was specified, use all pages
            if pages and not page_numbers:
                page_numbers = list(range(len(doc)))

            # Extract text from specified pages
            full_text = ""
            for page_num in page_numbers:
                page_text = doc[page_num].get_text()
                full_text += page_text + "\n"

            # Basic text processing
            paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
            sentences = [s.strip() for s in re.split(r'[.!?]+', full_text) if s.strip()]
            words = full_text.split()

            # Extract key phrases (simple frequency-based approach)
            word_freq = Counter(word.lower().strip('.,!?;:()[]{}') for word in words
                               if len(word) > 3 and word.isalpha())
            common_words = word_freq.most_common(20)

            # Extract potential key topics (capitalized phrases)
            topics = []
            topic_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            topic_matches = re.findall(topic_pattern, full_text)
            topic_freq = Counter(topic_matches)
            topics = [topic for topic, freq in topic_freq.most_common(10) if freq > 1]

            # Extract potential dates and numbers
            date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
            dates = list(set(re.findall(date_pattern, full_text)))

            number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
            numbers = [num for num in re.findall(number_pattern, full_text) if len(num) > 2]

            # Generate summary based on length preference
            summary_sentences = []
            target_sentences = {"short": 3, "medium": 7, "long": 15}.get(summary_length, 7)

            # Simple extractive summarization: select sentences with high keyword overlap
            if sentences:
                sentence_scores = []
                for sentence in sentences[:50]:  # Limit to first 50 sentences
                    score = sum(word_freq.get(word.lower(), 0) for word in sentence.split())
                    sentence_scores.append((score, sentence))

                # Select top sentences
                sentence_scores.sort(reverse=True)
                summary_sentences = [sent for _, sent in sentence_scores[:target_sentences]]

            # Generate insights
            insights = []

            if len(words) > 1000:
                insights.append(f"This is a substantial document with approximately {len(words):,} words")

            if topics:
                insights.append(f"Key topics include: {', '.join(topics[:5])}")

            if dates:
                insights.append(f"Document references {len(dates)} dates, suggesting time-sensitive content")

            if len(paragraphs) > 20:
                insights.append("Document has extensive content with detailed sections")

            # Document metrics
            reading_time = len(words) // 200  # Assuming 200 words per minute

            doc.close()

            return {
                "success": True,
                "summary": {
                    "length": summary_length,
                    "sentences": summary_sentences,
                    "key_insights": insights
                },
                "content_metrics": {
                    "total_words": len(words),
                    "total_sentences": len(sentences),
                    "total_paragraphs": len(paragraphs),
                    "estimated_reading_time_minutes": reading_time,
                    "pages_analyzed": len(page_numbers)
                },
                "key_elements": {
                    "top_keywords": [{"word": word, "frequency": freq} for word, freq in common_words[:10]],
                    "identified_topics": topics,
                    "dates_found": dates[:10],  # Limit for context window
                    "significant_numbers": numbers[:10]
                },
                "document_characteristics": {
                    "content_density": "high" if len(words) / len(page_numbers) > 500 else "medium" if len(words) / len(page_numbers) > 200 else "low",
                    "structure_complexity": "high" if len(paragraphs) / len(page_numbers) > 10 else "medium" if len(paragraphs) / len(page_numbers) > 5 else "low",
                    "topic_diversity": len(topics)
                },
                "file_info": {
                    "path": str(path),
                    "total_pages": len(doc),
                    "pages_processed": pages or "all"
                },
                "analysis_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Content summarization failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "analysis_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="analyze_layout",
        description="Analyze PDF page layout including text blocks, columns, and spacing"
    )
    async def analyze_layout(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        include_coordinates: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze PDF page layout structure including text blocks and spacing.

        Args:
            pdf_path: Path to PDF file or HTTPS URL
            pages: Page numbers to analyze (comma-separated, 1-based), None for all
            include_coordinates: Whether to include detailed coordinate information

        Returns:
            Dictionary containing layout analysis results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            # Parse pages parameter
            parsed_pages = parse_pages_parameter(pages)
            if parsed_pages:
                page_numbers = [p for p in parsed_pages if 0 <= p < len(doc)]
            else:
                page_numbers = list(range(min(5, len(doc))))  # Limit to 5 pages for performance

            # If parsing failed but pages was specified, default to first 5
            if pages and not page_numbers:
                page_numbers = list(range(min(5, len(doc))))

            layout_analysis = []

            for page_num in page_numbers:
                page = doc[page_num]
                page_rect = page.rect

                # Get text blocks
                text_dict = page.get_text("dict")
                blocks = text_dict.get("blocks", [])

                # Analyze text blocks
                text_blocks = []
                total_text_area = 0

                for block in blocks:
                    if "lines" in block:  # Text block
                        block_bbox = block.get("bbox", [0, 0, 0, 0])
                        block_width = block_bbox[2] - block_bbox[0]
                        block_height = block_bbox[3] - block_bbox[1]
                        block_area = block_width * block_height

                        total_text_area += block_area

                        block_info = {
                            "type": "text",
                            "width": round(block_width, 2),
                            "height": round(block_height, 2),
                            "area": round(block_area, 2),
                            "line_count": len(block["lines"])
                        }

                        if include_coordinates:
                            block_info["coordinates"] = {
                                "x1": round(block_bbox[0], 2),
                                "y1": round(block_bbox[1], 2),
                                "x2": round(block_bbox[2], 2),
                                "y2": round(block_bbox[3], 2)
                            }

                        text_blocks.append(block_info)

                # Analyze images
                images = page.get_images()
                image_blocks = []
                total_image_area = 0

                for img in images:
                    try:
                        # Get image position (approximate)
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        img_area = pix.width * pix.height
                        total_image_area += img_area

                        image_blocks.append({
                            "type": "image",
                            "width": pix.width,
                            "height": pix.height,
                            "area": img_area
                        })

                        pix = None
                    except:
                        pass

                # Calculate layout metrics
                page_area = page_rect.width * page_rect.height
                text_coverage = (total_text_area / page_area) if page_area > 0 else 0

                # Detect column layout (simplified)
                if text_blocks:
                    # Group blocks by x-coordinate to detect columns
                    x_positions = [block.get("coordinates", {}).get("x1", 0) for block in text_blocks if include_coordinates]
                    if x_positions:
                        x_positions.sort()
                        column_breaks = []
                        for i in range(1, len(x_positions)):
                            if x_positions[i] - x_positions[i-1] > 50:  # Significant gap
                                column_breaks.append(x_positions[i])

                        estimated_columns = len(column_breaks) + 1 if column_breaks else 1
                    else:
                        estimated_columns = 1
                else:
                    estimated_columns = 1

                # Determine layout type
                if estimated_columns > 2:
                    layout_type = "multi_column"
                elif estimated_columns == 2:
                    layout_type = "two_column"
                elif len(text_blocks) > 10:
                    layout_type = "complex"
                elif len(image_blocks) > 3:
                    layout_type = "image_heavy"
                else:
                    layout_type = "simple"

                page_analysis = {
                    "page": page_num + 1,
                    "page_size": {
                        "width": round(page_rect.width, 2),
                        "height": round(page_rect.height, 2)
                    },
                    "layout_type": layout_type,
                    "content_summary": {
                        "text_blocks": len(text_blocks),
                        "image_blocks": len(image_blocks),
                        "estimated_columns": estimated_columns,
                        "text_coverage_percent": round(text_coverage * 100, 1)
                    },
                    "text_blocks": text_blocks[:10] if len(text_blocks) > 10 else text_blocks,  # Limit for context
                    "image_blocks": image_blocks
                }

                layout_analysis.append(page_analysis)

            doc.close()

            # Overall document layout analysis
            layout_types = [page["layout_type"] for page in layout_analysis]
            most_common_layout = max(set(layout_types), key=layout_types.count) if layout_types else "unknown"

            avg_text_blocks = sum(page["content_summary"]["text_blocks"] for page in layout_analysis) / len(layout_analysis)
            avg_columns = sum(page["content_summary"]["estimated_columns"] for page in layout_analysis) / len(layout_analysis)

            return {
                "success": True,
                "layout_summary": {
                    "pages_analyzed": len(page_numbers),
                    "most_common_layout": most_common_layout,
                    "average_text_blocks_per_page": round(avg_text_blocks, 1),
                    "average_columns_per_page": round(avg_columns, 1),
                    "layout_consistency": "high" if len(set(layout_types)) <= 2 else "medium" if len(set(layout_types)) <= 3 else "low"
                },
                "page_layouts": layout_analysis,
                "layout_insights": [
                    f"Document uses primarily {most_common_layout} layout",
                    f"Average of {avg_text_blocks:.1f} text blocks per page",
                    f"Estimated {avg_columns:.1f} columns per page on average"
                ],
                "analysis_settings": {
                    "include_coordinates": include_coordinates,
                    "pages_processed": pages or f"first_{len(page_numbers)}"
                },
                "file_info": {
                    "path": str(path),
                    "total_pages": len(doc)
                },
                "analysis_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Layout analysis failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "analysis_time": round(time.time() - start_time, 2)
            }