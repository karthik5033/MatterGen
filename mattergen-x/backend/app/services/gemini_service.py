
import os
import google.generativeai as genai
from typing import Dict, Any

class GeminiService:
    def __init__(self):
        self.api_keys = []
        self._load_keys()
        
        if not self.api_keys:
            print("WARNING: No Gemini API keys found.")

    def _load_keys(self):
        # Specific path provided by user
        env_path = r"d:\coding_files\Projects\matterGen\material dataset\.env.local"
        try:
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.startswith("GEMINI_API_KEY"):
                            # Handle Key=Value properly, clean whitespace
                            parts = line.strip().split('=')
                            if len(parts) >= 2:
                                self.api_keys.append(parts[1].strip())
                print(f"Loaded {len(self.api_keys)} Gemini API keys.")
            else:
                print(f"Key file not found at {env_path}")
        except Exception as e:
            print(f"Error loading keys: {e}")

    async def analyze_material(self, formula: str, properties: Dict[str, Any]) -> Dict[str, str]:
        # Try Gemini first if keys exist
        if self.api_keys:
            prompt = f"""
            Act as a Senior Principal Materials Scientist at a top-tier research institute (like MIT or Max Planck). 
            Generate a comprehensive TECHNICAL REPORT for the newly discovered material '{formula}' with the following predicted properties:
            {properties}

            You must hallucinate/predict plausible scientific details based on these properties (e.g., if band gap is 0, it's a metal; if >3, it's an insulator).

            Provide a STRICT JSON response with the following keys:
            
            1. "executive_summary": A detailed professional executive essay (approx 150-200 words). Discuss the strategic importance, potential market disruption, and high-level performance characteristic. Do not hold back on complexity.
            2. "scientific_deep_dive": An extensive scientific essay (approx 400-500 words). This must be detailed and broken into clear paragraphs. Discuss:
               - Electronic Band Structure Analysis: Direct vs Indirect gaps, carrier mobility implications.
               - Crystallographic Stability: Lattice energies, phonon density of states (DOS), and thermal stability.
               - Bonding Mechanisms: Ionic/Covalent/Metallic character ratios and orbital hybridization.
               - Synthesis Pathways: Challenges (metastable phases) and proposed synthesis techniques (e.g., solvothermal, sputtering).
            3. "industrial_applications": A list of 3 complex objects, where each object has:
               - "title": Name of application (e.g. "Solid Oxide Fuel Cell Cathode")
               - "description": Why it works here.
               - "performance_metric": A specific, impressive number (e.g. "Conductivity > 0.5 S/cm at 600°C").
            4. "future_tech_lore": A creative, epic sci-fi description for a game/movie database. Give it a codename (e.g. "Aetherium-9"). Describe its role in faster-than-light travel, energy shielding, or terraforming.
            5. "ratings": An object containing integer scores (0-100):
               - "commercial_viability"
               - "sustainability_index"
               - "manufacturing_complexity"
            6. "synthesis_guide": Object with keys:
               - "method": "Solid State / Sol-gel / CVD..."
               - "precursors": ["List", "of", "chemicals"]
               - "equipment": ["List", "of", "machines"]
               - "detailed_procedure": ["Step 1...", "Step 2...", "Step 3..."]
               - "challenges": "Key difficulty..."
            7. "risk_profile": Object with keys:
               - "flammability": "Low/High..."
               - "toxicity": "Detailed description..."
               - "handling_precautions": ["Wear gloves", "Fume hood..."]
               - "disposal": "Protocol..."
            8. "economic_outlook": Object with keys:
               - "estimated_cost": "$X per kg"
               - "scalability_verdict": "High/Low..."
               - "supply_chain_risks": ["Rare earth shortage", "Geopolitical..."]
               - "potential_market_sectors": ["sector 1", "sector 2"]

            Format the output purely as valid JSON. Do not include markdown formatting.
            """
            
            for key in self.api_keys:
                try:
                    genai.configure(api_key=key)
                    # Try standards in order
                    for model_name in ['gemini-1.5-flash', 'gemini-pro']:
                        try:
                            model = genai.GenerativeModel(model_name)
                            response = await model.generate_content_async(prompt)
                            
                            # Clean/Parse
                            import json
                            import re
                            text = response.text
                            match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
                            json_str = match.group(1) if match else text
                            return json.loads(json_str)
                        except Exception:
                            continue # Try next model
                except Exception as e:
                    print(f"Key failed: {e}")
                    continue

        # Fallback if no keys or all failed
        print("Falling back to local simulation.")
        return self._generate_mock_fallback(formula, properties)

    def _generate_mock_fallback(self, formula: str, props: Dict[str, Any]) -> Dict[str, Any]:
        """Generate plausible-looking data locally when AI is offline."""
        import random
        
        # Parse elements roughly
        import re
        elements = re.findall(r'[A-Z][a-z]*', formula)
        main_el = elements[0] if elements else "Unknown"
        
        # Templates
        scifi_names = [f"Hyper-{main_el}", f"Quantum {formula}", f"{main_el}-Flux Alloy", "Starforged Crystal", "Void-Resistant Lattice"]
        scifi_techs = ["warp drive stabilizers", "plasma containment fields", "quantum computing cores", "interstellar hull plating", "neural interface links"]
        
        bg = props.get('band_gap', 0)
        is_metal = bg < 0.1
        
        # Scientific Mock
        scientific = (
            f"The crystal structure of {formula} exhibits unique electronic properties. "
            f"With a band gap of {bg:.2f} eV, it behaves as a {'promising semiconductor' if not is_metal else 'highly conductive metal'}. "
            f"Density functional theory calculations indicate strong orbital hybridization, contributing to its high thermodynamic stability. "
            f"Phonon dispersion analysis suggests dynamic stability at elevated temperatures, making it a candidate for extreme environments. "
            f"The electronic band structure reveals a {'direct' if random.random() > 0.5 else 'indirect'} gap, influencing its optoelectronic performance."
        )

        return {
            "executive_summary": f"{formula} represents a significant breakthrough in materials science, offering a unique combination of stability and performance. Its properties suggest immediate applicability in next-generation electronic and energy systems, potentially disrupting current market standards for high-efficiency components.",
            "scientific_deep_dive": scientific,
            "industrial_applications": [
                {
                   "title": "Next-gen Thermoelectrics",
                   "description": "Utilizing the material's low thermal conductivity and high electronic transport.",
                   "performance_metric": "ZT > 2.5 at 800K"
                },
                {
                   "title": "High-frequency Transistors" if not is_metal else "Superconducting Interconnects",
                   "description": "Leveraging high carrier mobility for rapid switching applications.",
                   "performance_metric": "Cutoff frequency > 100 GHz"
                },
                {
                   "title": "Extreme-environment Sensors",
                   "description": "Stable operation in high-pressure and corrosive environments.",
                   "performance_metric": "Operating range: -200°C to 1000°C"
                }
            ],
            "future_tech_lore": f"{random.choice(scifi_names)}. A material harvested from the hearts of dying stars. Primarily used in {random.choice(scifi_techs)} due to its ability to withstand chroniton radiation and subspace distortion.",
            "ratings": {
                "commercial_viability": random.randint(60, 95),
                "sustainability_index": random.randint(40, 90),
                "manufacturing_complexity": random.randint(30, 80)
            },
            "synthesis_guide": {
                "method": "High-Temperature Solid State Reaction",
                "precursors": [f"{main_el}2O3 (99.9% purity)", "Li2CO3", "TiO2 nanopowder"],
                "equipment": ["Planetary Ball Mill", "Tube Furnace (1500°C)", "Argon Glovebox"],
                "detailed_procedure": [
                    f"Stoichiometric mixing of {main_el} and secondary precursors.",
                    "High-energy ball milling for 12 hours at 400rpm.",
                    "Pelletization at 200 MPa.",
                    "Sintering at 1250°C for 24 hours under flowing Argon."
                ],
                "challenges": "Controlling the volatile components during the high-temperature sintering phase."
            },
            "risk_profile": {
                "flammability": "Non-flammable in bulk state. Powder form may present dust explosion hazard.",
                "toxicity": "Low toxicity. Avoid inhalation of precursor dusts. Long-term accumulation effects unknown.",
                "handling_precautions": ["Use N95 respirator or better", "Handle in well-ventilated area", "Standard nitrile gloves"],
                "disposal": "Dispose of in accordance with local chemical waste regulations. Do not flush down drains."
            },
            "economic_outlook": {
                "estimated_cost": "$120 - $180 per kg (Lab Scale)",
                "scalability_verdict": "Moderate. Batch processing currently limits throughput.",
                "supply_chain_risks": ["Fluctuating prices of Rare Earth precursors", "Single-source dependency for high-purity reagents"],
                "potential_market_sectors": ["Aerospace Defense", "Consumer Electronics", "Renewable Energy Grid"]
            }
        }


    async def chat_with_assistant(self, message: str, context: dict = None) -> dict:
        """
        Chat with Aether, the discovery assistant. 
        Translates raw intent into refined scientific prompts and weights.
        """
        if not self.api_keys:
            return {
                "response": "Discovery Assistant 'Aether' is currently offline. Please use manual specifications.",
                "suggested_prompt": message,
                "suggested_weights": None
            }

        prompt = f"""
        Act as Aether, a proprietary AI discovery assistant for the MatterGen project.
        Your goal is to help research scientists refine their material search.
        
        User Intent: "{message}"
        Current Context: {context}

        Rules:
        1. Be professional, slightly futuristic, and concise.
        2. Analyze the user's intent to extract target material class (e.g. Perovskite, Spinel) and properties (e.g. High conductivity, stability).
        3. Provide a refined 'Scientific Prompt' that would work better for a generative ML model.
        4. Suggest values (0.0 to 1.0) for the optimization sliders: 
           - density
           - stability
           - band_gap
           - shear_modulus
           - thermal_conductivity
           - refractive_index

        Provide a STRICT JSON response:
        {{
          "response": "Your friendly guidance message here.",
          "suggested_prompt": "Refined scientific prompt here",
          "suggested_weights": {{
             "density": 0.5,
             "stability": 0.8,
             "band_gap": 0.5,
             "shear_modulus": 0.5,
             "thermal_conductivity": 0.5,
             "refractive_index": 0.5
          }}
        }}
        """

        # Prefer 3rd key if available, otherwise rotate
        ordered_keys = self.api_keys.copy()
        if len(self.api_keys) >= 3:
            # Move 3rd key to front
            key3 = ordered_keys.pop(2)
            ordered_keys.insert(0, key3)

        for key in ordered_keys:
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = await model.generate_content_async(prompt)
                
                import json
                import re
                text = response.text
                match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
                json_str = match.group(1) if match else text
                return json.loads(json_str)
            except Exception as e:
                print(f"Chat failed with key: {e}")
                continue

        # If all keys fail, use the robust local fallback (simulated intelligence)
        print("All API keys failed. Engaging local fallback engine.")
        return self._local_chat_fallback(message)

    def _local_chat_fallback(self, message: str) -> dict:
        """
        A rule-based fallback engine to maintain chat functionality 
        when the LLM is unreachable.
        """
        msg_lower = message.lower()
        
        # 1. Handle Greetings
        if any(w in msg_lower for w in ['hi', 'hello', 'hey', 'start']):
            return {
                "response": "Systems online. I am ready to configure your material search.",
                "suggested_prompt": "Stable inorganic perovskite for photovoltaic applications",
                "suggested_weights": {"stability": 0.6, "band_gap": 0.8}
            }

        # 2. Handle Batteries/Energy
        if any(w in msg_lower for w in ['battery', 'storage', 'energy', 'lithium', 'ion']):
            return {
                "response": "I've detected an intent for Energy Storage. Optimizing for high Density and Stability.",
                "suggested_prompt": "High-capacity lithium-ion cathode material with garnet-type structure",
                "suggested_weights": {"density": 0.9, "stability": 0.7, "band_gap": 0.3}
            }

        # 3. Handle Solar/PV
        if any(w in msg_lower for w in ['solar', 'sun', 'light', 'photovol', 'kv']):
            return {
                "response": "Optimizing search parameters for Photovoltaic Efficiency. Prioritizing Band Gap targeting 1.3-1.5 eV.",
                "suggested_prompt": "Lead-free double perovskite with direct band gap for solar absorption",
                "suggested_weights": {"band_gap": 1.0, "stability": 0.5, "refractive_index": 0.6}
            }

        # 4. Handle Strength/Structural
        if any(w in msg_lower for w in ['strong', 'hard', 'steel', 'alloy', 'structure']):
            return {
                "response": "Structural integrity prioritized. Maximizing Shear Modulus and Density.",
                "suggested_prompt": "Ultra-high strength multi-principal element alloy (High Entropy Alloy)",
                "suggested_weights": {"shear_modulus": 1.0, "density": 0.6, "thermal_conductivity": 0.4}
            }

        # 5. Default Fallback
        return {
            "response": "I am operating in offline mode. I have prepared a general exploratory configuration for you.",
            "suggested_prompt": f"Novel material structure based on: {message}",
            "suggested_weights": {
                "stability": 0.5, 
                "density": 0.5, 
                "band_gap": 0.5,
                "shear_modulus": 0.5,
                "thermal_conductivity": 0.5,
                "refractive_index": 0.5
            }
        }


gemini_service = GeminiService()
