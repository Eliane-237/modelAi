"""
Script de test pour l'agent RAG juridique camerounais.
Ce script permet de tester l'interaction conversationnelle avec l'agent
directement depuis la ligne de commande.
"""

import sys
import os
import requests
import json
import time
from typing import Dict, Any, List

# Ajouter le chemin du projet au PYTHONPATH
project_root = "/home/mea/Documents/modelAi"
sys.path.append(project_root)

# Configuration
API_BASE_URL = "http://localhost:8000/api"
API_AGENT_ENDPOINT = "/agent/chat"

def print_colored(text, color="white", bold=False):
    """Affiche du texte coloré dans le terminal."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }
    
    bold_code = "\033[1m" if bold else ""
    reset_code = "\033[0m"
    
    color_code = colors.get(color.lower(), colors["white"])
    print(f"{color_code}{bold_code}{text}{reset_code}")

def format_source(source: Dict[str, Any]) -> str:
    """Formate une source pour l'affichage."""
    text = source.get("text", "")[:150] + "..." if len(source.get("text", "")) > 150 else source.get("text", "")
    metadata = source.get("metadata", {})
    citation = source.get("citation", "")
    
    return f"{text}\n  📄 {citation}"

def test_agent_conversation():
    """Test de conversation avec l'agent."""
    session_id = None
    
    print_colored("\n=== Assistant Juridique LexCam - Test de conversation ===\n", "cyan", bold=True)
    print_colored("Tapez 'exit' ou 'quit' pour quitter, 'reset' pour réinitialiser la conversation.\n", "yellow")
    
    while True:
        # Demander une question à l'utilisateur
        query = input("\n🧑 Votre question: ")
        
        if query.lower() in ["exit", "quit"]:
            print_colored("\nAu revoir! 👋", "cyan")
            break
            
        if query.lower() == "reset":
            # Réinitialiser la conversation
            if session_id:
                try:
                    reset_url = f"{API_BASE_URL}/agent/reset"
                    reset_params = {"session_id": session_id}
                    response = requests.post(reset_url, json=reset_params)
                    response.raise_for_status()
                    session_id = response.json().get("new_session_id")
                    print_colored(f"✅ Conversation réinitialisée. Nouvelle session: {session_id}", "green")
                except Exception as e:
                    print_colored(f"❌ Erreur lors de la réinitialisation: {e}", "red")
            else:
                print_colored("⚠️ Aucune session active à réinitialiser.", "yellow")
            continue
            
        if not query.strip():
            continue
            
        try:
            # Préparer la requête
            request_data = {
                "query": query,
                "session_id": session_id,
                "streaming": False
            }
            
            # Mesurer le temps de réponse
            start_time = time.time()
            
            # Envoyer la requête
            response = requests.post(f"{API_BASE_URL}{API_AGENT_ENDPOINT}", json=request_data)
            response.raise_for_status()
            
            # Calculer le temps de réponse
            response_time = time.time() - start_time
            
            # Analyser la réponse
            result = response.json()
            
            # Mettre à jour l'ID de session
            session_id = result.get("session_id")
            
            # Afficher la réponse
            print_colored(f"\n🤖 LexCam ({response_time:.2f}s):", "blue", bold=True)
            print(result.get("response", ""))
            
            # Afficher les sources
            sources = result.get("source_documents", [])
            if sources:
                print_colored("\n📚 Sources:", "magenta")
                for i, source in enumerate(sources[:3], 1):  # Limiter à 3 sources
                    print_colored(f"  {i}. {format_source(source)}", "white")
                
                if len(sources) > 3:
                    print_colored(f"  ... et {len(sources) - 3} autres sources", "white")
            
            # Afficher les domaines identifiés
            domains = result.get("domains", [])
            if domains:
                domain_str = ", ".join(domains)
                print_colored(f"\n🏛️ Domaines identifiés: {domain_str}", "green")
            
        except Exception as e:
            print_colored(f"\n❌ Erreur: {e}", "red")

def test_agent_tool(tool_name, query):
    """Test d'un outil spécifique de l'agent."""
    try:
        tool_url = f"{API_BASE_URL}/agent/tool/{tool_name}"
        params = {"query": query}
        
        response = requests.post(tool_url, json=params)
        response.raise_for_status()
        
        result = response.json()
        
        print_colored(f"\n=== Test de l'outil '{tool_name}' ===\n", "cyan", bold=True)
        print_colored(f"Requête: {query}", "yellow")
        print_colored("\nRésultat:", "green")
        print(result.get("result", ""))
        
    except Exception as e:
        print_colored(f"\n❌ Erreur: {e}", "red")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test de l'agent RAG juridique camerounais")
    parser.add_argument("--tool", help="Tester un outil spécifique")
    parser.add_argument("--query", help="Requête à utiliser avec l'outil")
    
    args = parser.parse_args()
    
    if args.tool and args.query:
        test_agent_tool(args.tool, args.query)
    else:
        test_agent_conversation()