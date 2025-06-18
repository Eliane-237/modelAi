"""
Service Web Agent SIMPLIFIÉ pour LexCam
Version allégée - Recherche Google + Synthèse avec votre LLM local
"""

import asyncio
import aiohttp
import time
import random
import logging
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str

class WebAgentService:
    """Agent web simple pour recherche Google + synthèse locale"""
    
    def __init__(self, llm_service, search_service, embedding_service):
        self.llm_service = llm_service
        self.search_service = search_service
        self.embedding_service = embedding_service
        self.session = None
        
        # Headers simples anti-détection
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'DNT': '1'
        }
        
        # Cache simple (30 minutes)
        self.cache = {}
        self.cache_duration = 1800

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=15)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def needs_web_search(self, query: str) -> bool:
        """Détermine si recherche web nécessaire"""
        query_lower = query.lower()
        
        # Mots-clés indiquant du contenu récent
        recent_keywords = [
            '2024', '2025', 'récent', 'nouveau', 'dernier', 'actualité',
            'prix', 'cours', 'aujourd\'hui', 'maintenant', 'actuel'
        ]
        
        return any(keyword in query_lower for keyword in recent_keywords)

    async def search_google(self, query: str) -> List[WebSearchResult]:
        """Recherche Google simple"""
        
        # Vérifier cache
        cache_key = f"google_{query}"
        if cache_key in self.cache:
            cache_time, results = self.cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                logger.info(f"Cache hit pour: {query}")
                return results
        
        search_url = f"https://www.google.com/search?q={query}&hl=fr&gl=cm&num=5"
        
        try:
            # Délai anti-bot simple
            await asyncio.sleep(random.uniform(1, 2))
            
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    html = await response.text()
                    results = self._parse_google_results(html)
                    
                    # Mettre en cache
                    self.cache[cache_key] = (time.time(), results)
                    
                    logger.info(f"Google: {len(results)} résultats pour '{query}'")
                    return results
                else:
                    logger.warning(f"Google failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Erreur Google: {e}")
        
        return []

    def _parse_google_results(self, html: str) -> List[WebSearchResult]:
        """Parse simple des résultats Google"""
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        # Chercher les divs de résultats
        for div in soup.find_all('div', class_='g')[:5]:
            try:
                # Titre
                title_elem = div.find('h3')
                title = title_elem.get_text(strip=True) if title_elem else ""
                
                # Lien
                link_elem = div.find('a')
                url = link_elem.get('href') if link_elem else ""
                
                # Nettoyer URL Google
                if url and url.startswith('/url?q='):
                    url = url.split('&')[0].replace('/url?q=', '')
                
                # Snippet
                snippet_elem = div.find('span') or div.find('div', class_='VwiC3b')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                if title and url and not url.startswith('http://webcache'):
                    results.append(WebSearchResult(
                        title=title,
                        url=url,
                        snippet=snippet[:200] + "..." if len(snippet) > 200 else snippet
                    ))
                    
            except Exception:
                continue
        
        return results

    async def search_vector_database(self, query: str) -> List[Dict]:
        """Recherche dans votre base vectorielle"""
        try:
            results = self.search_service.search(query, top_k=5)
            logger.info(f"Vector: {len(results)} résultats")
            return results
        except Exception as e:
            logger.error(f"Erreur vector: {e}")
            return []

    async def synthesize_response(self, query: str, web_results: List[WebSearchResult], 
                                 vector_results: List[Dict]) -> str:
        """Synthèse avec votre LLM local"""
        
        # Préparer contexte web
        web_context = ""
        if web_results:
            web_context = "INFORMATIONS WEB RÉCENTES:\n"
            for i, result in enumerate(web_results[:3], 1):
                web_context += f"{i}. {result.title}\n   {result.snippet}\n   Source: {result.url}\n\n"
        
        # Préparer contexte vectoriel
        vector_context = ""
        if vector_results:
            vector_context = "BASE JURIDIQUE CAMEROUNAISE:\n"
            for i, result in enumerate(vector_results[:3], 1):
                content = result.get("content", result.get("text", ""))[:300]
                source = result.get("metadata", {}).get("source", "Document juridique")
                vector_context += f"{i}. {source}\n   {content}...\n\n"
        
        # Prompt simple et efficace
        prompt = f"""Vous êtes LexCam, assistant juridique camerounais.

Question: {query}

{web_context}
{vector_context}

INSTRUCTIONS:
- Répondez en français de manière précise
- Utilisez d'abord la base juridique camerounaise
- Complétez avec les infos web si pertinentes
- Citez vos sources
- Soyez concis et accessible

Réponse:"""

        try:
            response = self.llm_service.generate_response(prompt, max_length=800)
            
            if isinstance(response, dict):
                return response.get("generated_text", response.get("response", ""))
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Erreur LLM: {e}")
            return "Désolé, je n'ai pas pu synthétiser une réponse."

    async def process_query(self, query: str) -> Dict:
        """Point d'entrée principal - SIMPLE"""
        start_time = time.time()
        
        try:
            # 1. Vérifier si web nécessaire
            needs_web = self.needs_web_search(query)
            
            # 2. Recherche vectorielle (toujours)
            vector_results = await self.search_vector_database(query)
            
            # 3. Recherche web si nécessaire
            web_results = []
            if needs_web:
                web_results = await self.search_google(query)
            
            # 4. Synthèse
            response = await self.synthesize_response(query, web_results, vector_results)
            
            return {
                "query": query,
                "response": response,
                "web_search_performed": needs_web,
                "web_sources": len(web_results),
                "vector_sources": len(vector_results),
                "processing_time": time.time() - start_time,
                "sources": {
                    "web": [{"title": r.title, "url": r.url} for r in web_results],
                    "vector": [{"source": r.get("metadata", {}).get("source", "Document")} 
                              for r in vector_results[:3]]
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur process_query: {e}")
            return {
                "query": query,
                "response": f"Erreur: {str(e)}",
                "web_search_performed": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

