# Filename: 4_generate_prescriptive_playbook.py
import json
import asyncio

async def generate_playbook(xai_findings): 
    prompt = f""" 
    As a SOC Manager, your task is to create a simple, step-by-step incident response playbook for a Tier 1 analyst. 
    The playbook should be based on the provided alert details and the explanation from our AI model. 
    
    Do not explain the AI model; only provide the prescriptive actions. The playbook must be a numbered list of 3-4 clear, concise steps. 
    
    **Alert Details & AI Explanation:** 
    {xai_findings} 
    """ 
    
    try: 
        chatHistory = [{"role": "user", "parts": [{"text": prompt}]}] 
        payload = {"contents": chatHistory} 
        apiKey = ""  # Leave as-is
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}" 
        response = await fetch(apiUrl, { 
            'method': 'POST', 
            'headers': {'Content-Type': 'application/json'}, 
            'body': JSON.stringify(payload) 
        }) 

        result = await response.json() 
        if result.get('candidates'): 
            return result['candidates'][0]['content']['parts'][0]['text'] 
        else: 
            return "Error: Could not generate playbook. Response was: " + json.dumps(result) 
    
    except Exception as e: 
        return f"An error occurred: {e}"


# Simulate the findings from our DGA model and SHAP lab
findings = """- **Alert:** Potential DGA domain detected in DNS logs.
- **Domain:** `kq3v9z7j1x5f8g2h.info`
- **Source IP:** `10.1.1.50` (Workstation-1337)
- **AI Model Explanation (from SHAP):** The model flagged this domain with 99.8% confidence primarily due to its very high character entropy and long length, which are strong indicators of an algorithmically generated domain."""


# Run the generation
playbook = asyncio.run(generate_playbook(findings))