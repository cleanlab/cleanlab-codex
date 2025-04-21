import requests
from bs4 import BeautifulSoup
from typing import List, Tuple, Optional, Dict, Any

from cleanlab_tlm import TLM


class TrustworthinessEvaluator:
    def __init__(self, client, model="gpt-4.1", tlm_model="gpt-4o-mini", tlm_preset="base"):
        self.client = client
        self.model = model
        self.tlm = TLM(options={
            "model": tlm_model,
            "quality_preset": tlm_preset,
            "log": ["explanation"]
        })

    def get_response(self, optimized_prompt: str, query: str) -> Any:
        response = self.client.responses.create(
            model=self.model,
            tools=[{"type": "web_search_preview"}],
            input=optimized_prompt.format(query=query)
        )
        return response

    def parse_response(self, response: Any) -> Tuple[str, List[Dict[str, str]]]:
        output_text = ""
        citations = []

        for item in response.output:
            if item.type == "message":
                for content_item in item.content:
                    if content_item.type == "output_text":
                        output_text = content_item.text
                        for annotation in content_item.annotations:
                            if annotation.type == "url_citation":
                                citations.append({
                                    "title": getattr(annotation, "title", "") or None,
                                    "url": annotation.url
                                })
        return output_text, citations

    def get_page_text(self, url: str, max_lines: int = 100) -> str:
        try:
            response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            return "\n".join(text.splitlines()[:max_lines])
        except Exception as e:
            return f"[Error fetching content: {e}]"

    def form_tlm_prompt_from_urls(
        self,
        prompt: str,
        page_titles: List[str],
        urls: List[str]
    ) -> Tuple[str, List[str]]:

        if len(page_titles) != len(urls):
            raise ValueError("page_titles and urls must have same length")

        if not page_titles:
            return f"""Respond to the given request from a user below.\n\n{prompt}""", []

        page_texts = [self.get_page_text(url) for url in urls]

        tlm_prompt = """Respond to the given request from a user below. To help you respond accurately, a web search was run using terms related to the user request. Here are the results of the web search:\n\n"""

        for title, url, text in zip(page_titles, urls, page_texts):
            tlm_prompt += f"""## {title}\nURL: {url}\n\n{text}\n\n"""

        tlm_prompt += f"""Using these web search results, answer the following User Request.\n\nUser Request:\n\n{prompt}"""
        return tlm_prompt, page_texts

    def get_trustworthiness_scores(
        self,
        response: Any,
        optimized_prompt: str,
        query: str
    ) -> Dict[str, Any]:

        response_text, citations = self.parse_response(response)
        titles = [c["title"] for c in citations]
        urls = [c["url"] for c in citations]

        tlm_prompt, page_texts = self.form_tlm_prompt_from_urls(query, titles, urls)

        trustworthiness_score = self.tlm.get_trustworthiness_score(tlm_prompt, response=response_text)

        return {
            "score": trustworthiness_score,
            "text": response_text,
            "citations": citations,
            "contexts": page_texts
        }
