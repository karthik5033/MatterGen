import sys
import os
# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from app.services.structure_generator import structure_generator
    print("Import successful")
    
    data = structure_generator.generate_batch(1, ["battery"])
    print("Generation successful")
    print(data[0]["formula"])
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
