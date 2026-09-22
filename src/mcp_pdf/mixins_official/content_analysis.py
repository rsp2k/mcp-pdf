"""
Content Analysis Mixin - PDF content classification, summarization, and layout analysis
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import time
from typing import Dict, Any, Literal, Optional
import logging
import re
from collections import Counter

# PDF processing libraries
import pymupdf

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

    @mcp_tool(
        name="classify_content",
        description=(
            "Guess what KIND of document this is — academic, business, legal, "
            "technical, financial, medical or educational — and report reading "
            "level, vocabulary diversity and rough word/image/link counts. "
            "Read-only, writes nothing, samples the first 10 pages.\n"
            "\n"
            "classify_content answers \"what is this document?\"; "
            "summarize_content answers \"what does it say?\" and returns actual "
            "sentences and keywords. Ask this one first when routing a pile of "
            "unknown PDFs, then summarize the ones that matter.\n"
            "\n"
            "ALWAYS CHECK confidence BEFORE USING primary_type. Classification "
            "is a raw count of fixed keyword substrings per category, and "
            "confidence is that category's SHARE of all keyword hits, not a "
            "probability of being right. When the document matches no keyword "
            "at all, primary_type is \"general\" with confidence 0.0, which "
            "means \"unclassified\" rather than any positive finding. "
            "Anything under roughly 0.4 is a weak signal; compare against "
            "secondary_types, which is a list of [category, raw_score] pairs "
            "for the next three ranked categories.\n"
            "\n"
            "Other caveats: estimated_word_count, estimated_images and "
            "estimated_links are extrapolated from the 10-page sample to the "
            "full page count, not counted. readability_score is a modified "
            "Flesch formula that substitutes sentence length for syllable "
            "count, so treat reading_level as a coarse band. A HIGHER "
            "readability_score means EASIER text (>=90 Elementary, >=70 Middle "
            "School, >=50 High School, >=30 College, below that Graduate). On a "
            "scanned PDF with no extractable text every figure here is "
            "meaningless — run is_scanned_pdf first."
        ),
        annotations={
            "readOnlyHint": True,        # opens the PDF, writes nothing
            "idempotentHint": True,
            "openWorldHint": True,       # pdf_path may be an http(s) URL
        },
    )
    async def classify_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        Classify PDF content type and analyze document structure.

        Args:
            pdf_path: Path to the PDF, or an http(s) URL to fetch.

        Returns:
            Dict with success plus:
              - classification: primary_type (one of academic, business, legal,
                technical, financial, medical, educational, or "general" when
                nothing matched), confidence (0.0-1.0 share of keyword hits;
                0.0 always means unclassified),
                secondary_types as [category, raw_score] pairs for ranks 2-4
              - content_analysis: total_pages, estimated_word_count
                (extrapolated), avg_words_per_page, vocabulary_diversity
                (unique words over total words in the sample),
                reading_level, readability_score (higher is easier)
              - document_structure: has_bookmarks, bookmark_levels,
                estimated_sections, is_structured
              - multimedia_content: estimated_images, estimated_links (both
                extrapolated), is_multimedia_rich
              - content_characteristics: is_text_heavy (>500 words/page),
                is_technical, has_formal_language, complexity_level
                (high/medium/low from vocabulary_diversity)
              - file_info.pages_analyzed: how many pages were actually sampled
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = pymupdf.open(str(path))
            total_pages = len(doc)

            # Extract text from sample pages for analysis
            sample_size = min(10, total_pages)
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

            # Determine primary content type.
            #
            # Guard on any(...) rather than on the dict being non-empty.
            # content_scores always has one entry per category, so `if
            # content_scores:` was unconditionally true and the "general"
            # branch below was dead code. A document matching no keyword got
            # max() over an all-zero dict, which returns the FIRST key by
            # insertion order, so every unclassifiable document was reported
            # as "academic" purely because academic is declared first.
            # Confidence was 0.0, but a caller reading primary_type without
            # checking confidence saw a confident-looking wrong answer.
            if any(content_scores.values()):
                primary_type = max(content_scores, key=content_scores.get)
                confidence = content_scores[primary_type] / max(sum(content_scores.values()), 1)
            else:
                primary_type = "general"
                confidence = 0.0

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
            estimated_total_images = int(total_images * total_pages / sample_size) if sample_size > 0 else 0
            estimated_total_links = int(total_links * total_pages / sample_size) if sample_size > 0 else 0

            doc.close()

            return {
                "success": True,
                "classification": {
                    "primary_type": primary_type,
                    "confidence": round(confidence, 2),
                    "secondary_types": sorted(content_scores.items(), key=lambda x: x[1], reverse=True)[1:4]
                },
                "content_analysis": {
                    "total_pages": total_pages,
                    "estimated_word_count": int(total_words * total_pages / sample_size),
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
        description=(
            "Pull representative sentences, frequent keywords, capitalised "
            "topics, dates and numbers out of a PDF. Read-only, writes "
            "nothing. Processes ALL pages by default, so pass `pages` on a long "
            "document.\n"
            "\n"
            "This is EXTRACTIVE and purely statistical — no model is called, so "
            "there is no abstractive prose and no paraphrase. If you want an "
            "actual written summary, take the returned sentences and write it "
            "yourself, or use extract_text and read the text. Use "
            "classify_content instead to learn what kind of document this is.\n"
            "\n"
            "Two things about `summary.sentences` that will surprise you: only "
            "the FIRST 50 SENTENCES of the selected pages are ever candidates, "
            "so nothing from later in a long document can appear; and the "
            "sentences are returned in DESCENDING SCORE ORDER, not document "
            "order, so they do not read as a paragraph. Sentence count is 3 for "
            "\"short\", 7 for \"medium\", 15 for \"long\".\n"
            "\n"
            "`pages` is a 1-based string: \"5\", \"1,3,5\", \"1-10\", or mixed "
            "\"1,3-5,12-20\". Omit for the whole document; an unparseable value "
            "silently falls back to all pages. top_keywords is raw frequency "
            "with NO stopword filtering, so expect \"that\", \"with\" and "
            "\"this\" near the top. dates_found only matches numeric forms like "
            "3/14/2026 or 2026-03-14, never \"March 14, 2026\", and is "
            "deduplicated through a set so the order is arbitrary. dates_found "
            "and significant_numbers are each capped at 10 entries."
        ),
        annotations={
            "readOnlyHint": True,        # opens the PDF, writes nothing
            "idempotentHint": True,
            "openWorldHint": True,       # pdf_path may be an http(s) URL
        },
    )
    async def summarize_content(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        summary_length: Literal["short", "medium", "long"] = "medium"
    ) -> Dict[str, Any]:
        """
        Generate summary and extract key insights from PDF content.

        Args:
            pdf_path: Path to the PDF, or an http(s) URL to fetch.
            pages: 1-based page selection, e.g. "5", "1,3,5", "1-10",
                "1,3-5,12-20". None (the default) means every page. An
                unparseable value falls back to every page.
            summary_length: How many sentences to return — "short" gives 3,
                "medium" 7, "long" 15.

        Returns:
            Dict with success plus:
              - summary: length, sentences (score-ordered, drawn only from the
                first 50 sentences), key_insights (canned observations about
                word count, topics, dates and section count)
              - content_metrics: total_words, total_sentences,
                total_paragraphs, estimated_reading_time_minutes (words // 200),
                pages_analyzed
              - key_elements: top_keywords as {word, frequency} objects (top 10,
                no stopword filtering), identified_topics (repeated capitalised
                phrases), dates_found (max 10, numeric formats only),
                significant_numbers (max 10, as strings)
              - document_characteristics: content_density, structure_complexity,
                topic_diversity
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = pymupdf.open(str(path))
            total_pages = len(doc)

            # Parse pages parameter
            parsed_pages = parse_pages_parameter(pages)
            page_numbers = parsed_pages if parsed_pages else list(range(total_pages))
            page_numbers = [p for p in page_numbers if 0 <= p < total_pages]

            # If parsing failed but pages was specified, use all pages
            if pages and not page_numbers:
                page_numbers = list(range(total_pages))

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
                    "total_pages": total_pages,
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
        description=(
            "Describe the geometry WITHIN pages: how many text blocks, their "
            "sizes and bounding boxes, how many columns the text appears to be "
            "in, how much of the page the text covers, and a coarse layout_type "
            "per page. Read-only, writes nothing.\n"
            "\n"
            "Page-level, not document-level. Use get_document_structure for the "
            "bookmark outline and page sizes, and detect_structure to find "
            "chapters and sections. Reach for analyze_layout when you need to "
            "know whether text is in two columns before extracting it, or where "
            "on the page a block sits so you can place an annotation.\n"
            "\n"
            "DEFAULTS TO THE FIRST 5 PAGES ONLY when `pages` is omitted — it "
            "does not analyse the whole document. `pages` is a 1-based string: "
            "\"5\", \"1,3,5\", \"1-10\", or mixed \"1,3-5,12-20\"; an "
            "unparseable value falls back to the first 5 pages.\n"
            "\n"
            "include_coordinates only controls response size: set it False to "
            "drop each block's x1/y1/x2/y2 box from the output. Column "
            "detection is unaffected either way. The heuristic is crude — it sorts "
            "text-block left edges and counts gaps wider than 50 points, so "
            "indented blocks, sidebars and tables all read as extra columns. "
            "text_coverage_percent sums block bounding-box areas and can exceed "
            "100 when blocks overlap. image_blocks report the image's own pixel "
            "dimensions, NOT its placement on the page, and carry no "
            "coordinates. Each page's text_blocks list is truncated to the "
            "first 10 entries even though the counts stay complete."
        ),
        annotations={
            "readOnlyHint": True,        # opens the PDF, writes nothing
            "idempotentHint": True,
            "openWorldHint": True,       # pdf_path may be an http(s) URL
        },
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
            pdf_path: Path to the PDF, or an http(s) URL to fetch.
            pages: 1-based page selection, e.g. "5", "1,3,5", "1-10",
                "1,3-5,12-20". None (the default) analyses the FIRST 5 PAGES,
                not all of them. An unparseable value falls back to the first 5.
            include_coordinates: Include each text block's x1/y1/x2/y2 box in
                the response. Purely a verbosity switch; estimated_columns and
                layout_type are computed the same way with it off.

        Returns:
            Dict with success plus:
              - layout_summary: pages_analyzed, most_common_layout,
                average_text_blocks_per_page, average_columns_per_page,
                layout_consistency (high if at most 2 distinct layout types
                were seen, medium at 3, else low)
              - page_layouts: per page — page, page_size, layout_type
                (multi_column when >2 columns, two_column at 2, complex when
                >10 text blocks, image_heavy when >3 images, else simple),
                content_summary counts, text_blocks (first 10) and image_blocks
              - layout_insights: three prose lines restating the summary
              - analysis_settings: include_coordinates and pages_processed
                (the literal `pages` value, or "first_N")
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = pymupdf.open(str(path))
            total_pages = len(doc)

            # Parse pages parameter
            parsed_pages = parse_pages_parameter(pages)
            if parsed_pages:
                page_numbers = [p for p in parsed_pages if 0 <= p < total_pages]
            else:
                page_numbers = list(range(min(5, total_pages)))  # Limit to 5 pages for performance

            # If parsing failed but pages was specified, default to first 5
            if pages and not page_numbers:
                page_numbers = list(range(min(5, total_pages)))

            layout_analysis = []

            for page_num in page_numbers:
                page = doc[page_num]
                page_rect = page.rect

                # Get text blocks
                text_dict = page.get_text("dict")
                blocks = text_dict.get("blocks", [])

                # Analyze text blocks
                text_blocks = []
                block_x_positions = []   # left edges, for column detection
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

                        # Always record the left edge for column detection.
                        # This used to be read back out of block_info
                        # ["coordinates"], which only exists when
                        # include_coordinates is True, so turning the flag off
                        # left the detector with an empty list and every page
                        # reported estimated_columns=1. The flag is supposed to
                        # control how much we put in the RESPONSE, not whether
                        # the analysis runs.
                        block_x_positions.append(block_bbox[0])

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
                        pix = pymupdf.Pixmap(doc, xref)
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
                    # Group blocks by left edge to detect columns. Sourced from
                    # block_x_positions, gathered during the block walk above,
                    # so this works regardless of include_coordinates.
                    x_positions = list(block_x_positions)
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
                    "total_pages": total_pages
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