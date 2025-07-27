import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
import pdfplumber
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ----------------------------
# Utility Classes
# ----------------------------

class PDFProcessor:
    def __init__(self):
        self.heading_patterns = [
            (re.compile(r'^(?P<text>[A-Z][A-Z\s]{3,})$'), 'H1'),
            (re.compile(r'^\d+\.\s+(?P<text>.+)$'), 'H2'),
            (re.compile(r'^\d+\.\d+\.\s+(?P<text>.+)$'), 'H3'),
        ]

    def extract(self, pdf_path: str) -> List[Dict]:
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
                        level, heading_text = heading_result
                        if level == 'H1':
                            current_hierarchy = [heading_text]
                        elif level == 'H2':
                            current_hierarchy = current_hierarchy[:1] + [heading_text]
                        elif level == 'H3':
                            current_hierarchy = current_hierarchy[:2] + [heading_text]
                        sections.append({
                            "type": "heading",
                            "level": level,
                            "text": heading_text,
                            "page": page_num,
                            "hierarchy": current_hierarchy.copy()
                        })
                        prev_level = level
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
                return (level, match.group('text').strip())
        if len(line.split()) < 8 and line[0].isupper():
            return ('H2', line)
        return None


class PersonaAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def analyze(self, persona: str, job: str) -> Dict:
        combined = f"{persona}. {job}"
        doc = self.nlp(combined)
        key_terms = set()
        verbs = set()
        for chunk in doc.noun_chunks:
            key_terms.add(chunk.text.lower().strip())
        for token in doc:
            if token.pos_ == "VERB":
                verbs.add(token.lemma_.lower())
        return {
            "persona": persona,
            "job": job,
            "keywords": list(key_terms),
            "actions": list(verbs)
        }


class RelevanceScorer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def score_sections(self, docs: List[Dict], profile: Dict) -> List[Dict]:
        all_sections = []
        for doc in docs:
            for sec in doc['sections']:
                if sec['type'] == 'heading':
                    sec_text = self._gather_text(sec, doc['sections'])
                    all_sections.append({
                        "doc": doc["name"],
                        "page": sec["page"],
                        "title": sec["text"],
                        "level": sec["level"],
                        "text": sec_text
                    })

        # Semantic scoring
        job_query = f"{profile['persona']} needs to {' '.join(profile['actions'])} about {' '.join(profile['keywords'])}"
        section_texts = [sec['text'] for sec in all_sections]
        embeddings = self.model.encode(section_texts + [job_query])
        section_vecs = embeddings[:-1]
        job_vec = embeddings[-1].reshape(1, -1)
        similarities = cosine_similarity(section_vecs, job_vec).flatten()

        # Add scores
        for i, sec in enumerate(all_sections):
            sec['score'] = float(similarities[i])
        ranked = sorted(all_sections, key=lambda x: x['score'], reverse=True)
        return ranked[:20]

    def _gather_text(self, heading: Dict, all_sections: List[Dict]) -> str:
        idx = all_sections.index(heading)
        content = [heading['text']]
        for sec in all_sections[idx+1:]:
            if sec['type'] == 'heading':
                break
            content.append(sec['text'])
        return ' '.join(content)


def extract_subsections(top_sections: List[Dict], profile: Dict) -> List[Dict]:
    results = []
    for sec in top_sections[:10]:  # top 10
        sentences = re.split(r'(?<=[.!?])\s+', sec['text'])
        scored = []
        for s in sentences:
            score = sum(1 for kw in profile['keywords'] if kw in s.lower())
            if score > 0:
                scored.append((score, s))
        scored.sort(reverse=True)
        for i, (score, sent) in enumerate(scored[:3]):  # top 3 sentences
            results.append({
                "document": sec["doc"],
                "page_number": sec["page"],
                "subsection_title": f"Key Detail from {sec['title']}",
                "refined_text": sent,
                "relevance_to_job": round(sec["score"] * 0.9, 4)
            })
    return results


# ----------------------------
# Main pipeline
# ----------------------------
def run_analysis(input_dir, persona, job, output_path):
    pdf_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    if not pdf_paths:
        raise Exception("No PDFs found.")

    processor = PDFProcessor()
    analyzer = PersonaAnalyzer()
    scorer = RelevanceScorer()

    documents = []
    for path in pdf_paths:
        try:
            sections = processor.extract(path)
            documents.append({"name": os.path.basename(path), "sections": sections})
        except Exception as e:
            print(f"Error in {path}: {e}")

    profile = analyzer.analyze(persona, job)
    top_sections = scorer.score_sections(documents, profile)
    sub_sections = extract_subsections(top_sections, profile)

    output = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in pdf_paths],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.utcnow().isoformat() + "Z"
        },
        "extracted_sections": [
            {
                "document": sec["doc"],
                "page_number": sec["page"],
                "section_title": sec["title"],
                "importance_rank": i+1,
                "confidence_score": round(sec["score"], 4)
            }
            for i, sec in enumerate(top_sections)
        ],
        "sub_section_analysis": sub_sections
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"✅ Output saved to {output_path}")


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input", help="Input directory containing PDFs")
    parser.add_argument("--persona", required=True, help="Persona description")
    parser.add_argument("--job", required=True, help="Job to be done")
    parser.add_argument("--output", default="output.json", help="Output JSON file")
    args = parser.parse_args()

    run_analysis(args.input, args.persona, args.job, args.output)
