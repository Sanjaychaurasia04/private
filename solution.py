import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------------------
# PDF Processor
# ----------------------------
class PDFProcessor:
    """Handles PDF text extraction and structure analysis"""

    def __init__(self):
        self.heading_patterns = [
            (re.compile(r'^(?P<text>[A-Z][A-Z0-9\s]+)$'), 'H1'),  # All caps
            (re.compile(r'^(?P<num>\d+\.\d+)\s+(?P<text>.+)$'), 'H2'),  # Numbered headings
            (re.compile(r'^(?P<num>\d+\.\d+\.\d+)\s+(?P<text>.+)$'), 'H3'),  # Sub-numbered
            (re.compile(r'^(?P<text>Chapter \d+:.+)$'), 'H1'),  # Chapter titles
            (re.compile(r'^[IVX]+\.\s+(?P<text>.+)$'), 'H1')  # Roman numerals
        ]

    def extract_text_with_structure(self, pdf_path: str) -> List[Dict]:
        sections = []
        current_hierarchy = []
        prev_level = None

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue

                for line in text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue

                    heading_result = self._detect_heading(line)
                    if heading_result:
                        heading_level, heading_text = heading_result

                        if heading_level == 'H1':
                            current_hierarchy = [heading_text]
                        elif heading_level == 'H2':
                            current_hierarchy = current_hierarchy[:1] + [heading_text]
                        elif heading_level == 'H3':
                            current_hierarchy = current_hierarchy[:2] + [heading_text]

                        sections.append({
                            "type": "heading",
                            "level": heading_level,
                            "text": heading_text,
                            "page": page_num,
                            "hierarchy": current_hierarchy.copy()
                        })
                        prev_level = heading_level
                    else:
                        sections.append({
                            "type": "content",
                            "text": line,
                            "page": page_num,
                            "hierarchy": current_hierarchy.copy(),
                            "parent_level": prev_level
                        })

        return sections

    def _detect_heading(self, line: str) -> Optional[tuple]:
        for pattern, level in self.heading_patterns:
            match = pattern.match(line)
            if match:
                text = match.groupdict().get('text', line)
                return (level, text.strip())

        if len(line.split()) <= 6 and line[0].isupper():
            return ('H2', line)

        return None


# ----------------------------
# Persona Analyzer
# ----------------------------
class PersonaAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def analyze_persona(self, persona_desc: str, job_desc: str) -> Dict:
        combined = f"{persona_desc}. {job_desc}"
        doc = self.nlp(combined)

        key_terms = set()
        actions = set()
        entities = set()

        for chunk in doc.noun_chunks:
            key_terms.add(chunk.text.strip().lower())

        for token in doc:
            if token.pos_ == "VERB":
                actions.add(token.lemma_.lower())

        for ent in doc.ents:
            entities.add(ent.text.strip().lower())

        doc_type = self._classify_document_type(combined)

        return {
            "key_terms": list(key_terms | entities),
            "actions": list(actions),
            "doc_type": doc_type,
            "persona_type": self._classify_persona(persona_desc),
            "job_type": self._classify_job(job_desc)
        }

    def _classify_persona(self, text: str) -> str:
        text = text.lower()
        if any(t in text for t in ['research', 'phd', 'scientist']):
            return "academic"
        elif 'student' in text:
            return "student"
        elif any(t in text for t in ['analyst', 'investment', 'finance']):
            return "analyst"
        elif 'journalist' in text:
            return "journalist"
        elif 'executive' in text:
            return "executive"
        else:
            return "general"

    def _classify_job(self, text: str) -> str:
        text = text.lower()
        if any(t in text for t in ['review', 'literature', 'survey']):
            return "review"
        elif any(t in text for t in ['analyze', 'trend', 'compare', 'evaluate']):
            return "analysis"
        elif any(t in text for t in ['study', 'exam', 'learn', 'prepare']):
            return "learning"
        elif any(t in text for t in ['summary', 'overview', 'brief']):
            return "summarization"
        else:
            return "information_extraction"

    def _classify_document_type(self, text: str) -> str:
        text = text.lower()
        if any(t in text for t in ['research', 'paper', 'journal']):
            return "academic"
        elif any(t in text for t in ['report', 'annual', 'financial']):
            return "business"
        elif any(t in text for t in ['textbook', 'chapter', 'course']):
            return "educational"
        elif any(t in text for t in ['article', 'news', 'blog']):
            return "media"
        else:
            return "general"


# ----------------------------
# Relevance Scorer
# ----------------------------
class RelevanceScorer:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

    def score_sections(self, documents: List[Dict], profile: Dict) -> List[Dict]:
        all_sections = self._prepare_sections(documents)
        tfidf_scores = self._calculate_tfidf_scores(all_sections, profile['key_terms'])
        semantic_scores = self._calculate_semantic_scores(all_sections, profile)

        scored = []
        for i, sec in enumerate(all_sections):
            score = (
                0.4 * tfidf_scores[i] +
                0.4 * semantic_scores[i] +
                0.1 * self._get_level_weight(sec['level']) +
                0.1 * self._get_doc_type_weight(sec['doc_type'], profile['doc_type'])
            )
            scored.append({**sec, "combined_score": float(score)})
        return scored

    def _prepare_sections(self, documents: List[Dict]) -> List[Dict]:
        all_sections = []
        for doc in documents:
            for sec in doc['sections']:
                if sec['type'] == 'heading':
                    full_text = self._get_section_text(sec, doc['sections'])
                    all_sections.append({
                        **sec,
                        'doc_id': doc['id'],
                        'doc_name': doc['name'],
                        'doc_type': doc['type'],
                        'full_text': full_text
                    })
        return all_sections

    def _get_section_text(self, section: Dict, all_sections: List[Dict]) -> str:
        start_idx = all_sections.index(section)
        section_text = [section['text']]
        current_level = section['level']
        for next_section in all_sections[start_idx + 1:]:
            if next_section['type'] == 'heading':
                if self._is_higher_level(current_level, next_section['level']):
                    break
            section_text.append(next_section['text'])
        return ' '.join(section_text)

    def _is_higher_level(self, current: str, next_: str) -> bool:
        levels = {'H1': 1, 'H2': 2, 'H3': 3}
        return levels.get(next_, 4) <= levels.get(current, 4)

    def _calculate_tfidf_scores(self, sections: List[Dict], key_terms: List[str]) -> np.ndarray:
        texts = [s['full_text'] for s in sections]
        combined_terms = ' '.join(key_terms)
        tfidf_matrix = self.tfidf.fit_transform(texts + [combined_terms])
        text_matrix = tfidf_matrix[:-1]
        query_vector = tfidf_matrix[-1]
        similarities = cosine_similarity(text_matrix, query_vector.reshape(1, -1))
        return similarities.flatten()

    def _calculate_semantic_scores(self, sections: List[Dict], profile: Dict) -> np.ndarray:
        job_desc = f"{profile['persona_type']} needs to {' '.join(profile['actions'])} about {' '.join(profile['key_terms'])}"
        section_texts = [s['full_text'] for s in sections]
        embeddings = self.sentence_model.encode(section_texts + [job_desc])
        section_embeddings = embeddings[:-1]
        job_embedding = embeddings[-1].reshape(1, -1)
        similarities = cosine_similarity(section_embeddings, job_embedding)
        return similarities.flatten()

    def _get_level_weight(self, level: Optional[str]) -> float:
        return {'H1': 1.0, 'H2': 0.7, 'H3': 0.5}.get(level, 0.3)

    def _get_doc_type_weight(self, doc_type: Optional[str], target_type: str) -> float:
        return 1.0 if doc_type == target_type else 0.7


# ----------------------------
# Document Analyzer
# ----------------------------
class DocumentAnalyzer:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.persona_analyzer = PersonaAnalyzer()
        self.relevance_scorer = RelevanceScorer()

    def process_documents(self, pdf_paths: List[str], persona_desc: str, job_desc: str) -> Dict:
        documents = []
        for i, path in enumerate(pdf_paths):
            try:
                sections = self.pdf_processor.extract_text_with_structure(path)
                doc_type = self._guess_document_type(path, sections)
                documents.append({
                    "id": i,
                    "name": os.path.basename(path),
                    "type": doc_type,
                    "sections": sections
                })
            except Exception as e:
                print(f"Error processing {path}: {e}")
        if not documents:
            raise ValueError("No valid documents processed")

        profile = self.persona_analyzer.analyze_persona(persona_desc, job_desc)
        scored_sections = self.relevance_scorer.score_sections(documents, profile)
        ranked = sorted(scored_sections, key=lambda x: x['combined_score'], reverse=True)[:20]

        return {
            "metadata": {
                "input_documents": [os.path.basename(p) for p in pdf_paths],
                "persona": persona_desc,
                "job_to_be_done": job_desc,
                "processing_timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "extracted_sections": [
                {
                    "document": s['doc_name'],
                    "page_number": s['page'],
                    "section_title": s['text'],
                    "confidence_score": round(s['combined_score'], 4)
                } for s in ranked
            ]
        }

    def _guess_document_type(self, path: str, sections: List[Dict]) -> str:
        name = os.path.basename(path).lower()
        if 'research' in name:
            return 'academic'
        if 'report' in name or 'financial' in name:
            return 'business'
        return 'general'


# ----------------------------
# Utility Functions
# ----------------------------
def save_output(output: Dict, output_path: str):
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='input')
    parser.add_argument('--output', '-o', default='output.json')
    parser.add_argument('--persona', '-p', required=True)
    parser.add_argument('--job', '-j', required=True)
    args = parser.parse_args()

    pdf_paths = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith('.pdf')]

    if not pdf_paths:
        print("No PDF files found in the input directory.")
        return

    analyzer = DocumentAnalyzer()
    output = analyzer.process_documents(pdf_paths, args.persona, args.job)
    save_output(output, args.output)
    print(f"✅ Output saved to {args.output}")


if __name__ == "__main__":
    main()
